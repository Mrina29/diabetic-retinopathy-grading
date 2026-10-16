"""
model.py
Uncertainty-Aware Ordinal DR Grading Model
Architecture: EfficientNet-B4 + MC Dropout + Frank-Hall Ordinal Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class FrankHallOrdinalHead(nn.Module):
    """
    Frank-Hall ordinal regression head.
    4 binary sigmoid outputs for 5 DR grades (0-4).
    Grade k is encoded as [1,1,...,1,0,...,0] with k ones.
    """
    def __init__(self, in_features: int, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes - 1  # 4 binary classifiers
        self.fc = nn.Linear(in_features, self.num_outputs)

    def forward(self, x):
        # Raw logits → sigmoid for each binary threshold
        logits = self.fc(x)
        probs = torch.sigmoid(logits)  # shape: (B, 4)
        return probs

    def decode_grade(self, probs):
        """
        Convert cumulative probabilities to grade.
        Grade = number of thresholds exceeded (>0.5).
        """
        binary = (probs > 0.5).float()
        grade = binary.sum(dim=-1).long()
        return grade


class DRGradingModel(nn.Module):
    """
    Full model: EfficientNet-B4 backbone + MC Dropout + Ordinal Head
    """
    def __init__(self, dropout_rate: float = 0.3, pretrained: bool = False):
        super().__init__()
        self.dropout_rate = dropout_rate

        if TIMM_AVAILABLE:
            # Real EfficientNet-B4 backbone
            self.backbone = timm.create_model(
                'efficientnet_b4',
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                global_pool='avg'
            )
            in_features = self.backbone.num_features  # 1792 for B4
        else:
            # Fallback lightweight backbone for demo
            self.backbone = self._build_simple_backbone()
            in_features = 256

        # MC Dropout — applied at inference too
        self.mc_dropout = nn.Dropout(p=dropout_rate)

        # Frank-Hall Ordinal Head
        self.ordinal_head = FrankHallOrdinalHead(in_features, num_classes=5)

        # Grad-CAM hook storage
        self.gradients = None
        self.activations = None

    def _build_simple_backbone(self):
        """Lightweight CNN backbone for demo when timm is unavailable."""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU()
        )

    def forward(self, x):
        # Extract features
        if TIMM_AVAILABLE:
            features = self.backbone(x)
        else:
            features = self.backbone(x)

        # MC Dropout (active during both train AND inference)
        features = self.mc_dropout(features)

        # Ordinal head
        probs = self.ordinal_head(features)
        return probs

    def predict_with_uncertainty(self, x, n_passes: int = 30):
        """
        Monte Carlo Dropout inference.
        Run n_passes forward passes with dropout active.
        Returns: grade, uncertainty score, per-grade probabilities
        """
        self.train()  # Keep dropout active

        all_probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                probs = self.forward(x)  # (B, 4)
                all_probs.append(probs.unsqueeze(0))

        all_probs = torch.cat(all_probs, dim=0)  # (n_passes, B, 4)

        # Mean prediction
        mean_probs = all_probs.mean(dim=0)  # (B, 4)

        # Uncertainty = variance across passes
        uncertainty = all_probs.var(dim=0).mean(dim=-1)  # (B,)
        uncertainty_normalized = torch.clamp(uncertainty * 50, 0, 1)  # Scale to [0,1]

        # Decode grade
        grade = self.ordinal_head.decode_grade(mean_probs)

        # Convert ordinal probs to per-grade distribution
        grade_probs = self._ordinal_to_grade_probs(mean_probs)

        return grade, uncertainty_normalized, grade_probs

    def _ordinal_to_grade_probs(self, cumul_probs):
        """
        Convert Frank-Hall cumulative probs to per-grade probability distribution.
        P(grade=0) = 1 - P(>0)
        P(grade=k) = P(>k-1) - P(>k)   for k=1,2,3
        P(grade=4) = P(>3)
        """
        B = cumul_probs.shape[0]
        p = cumul_probs  # (B, 4)

        grade_probs = torch.zeros(B, 5, device=p.device)
        grade_probs[:, 0] = 1 - p[:, 0]
        grade_probs[:, 1] = p[:, 0] - p[:, 1]
        grade_probs[:, 2] = p[:, 1] - p[:, 2]
        grade_probs[:, 3] = p[:, 2] - p[:, 3]
        grade_probs[:, 4] = p[:, 3]

        # Clamp negatives (can occur from non-monotonic predictions)
        grade_probs = torch.clamp(grade_probs, min=0)
        grade_probs = grade_probs / grade_probs.sum(dim=-1, keepdim=True)

        return grade_probs


def get_model(dropout_rate=0.3, pretrained=False):
    return DRGradingModel(dropout_rate=dropout_rate, pretrained=pretrained)
