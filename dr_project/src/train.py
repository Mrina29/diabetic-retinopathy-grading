"""
train.py
2-stage training pipeline for DR grading model.

Stage 1: Freeze backbone, train ordinal head only (5 epochs)
Stage 2: Unfreeze all, fine-tune at low LR with cosine annealing

Dataset expected structure:
  data/
    train/
      images/  *.png
      labels.csv  (image_id, diagnosis)
    val/
      images/  *.png
      labels.csv

Usage:
  python src/train.py --data_dir data/ --epochs_s1 5 --epochs_s2 20
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from sklearn.metrics import cohen_kappa_score

from model import get_model
from preprocessing import preprocess_pil_image, get_train_transform, frank_hall_encode


class DRDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['id_code']}.png"
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # Frank-Hall ordinal encoding
        grade = int(row['diagnosis'])
        label = torch.tensor(frank_hall_encode(grade), dtype=torch.float32)

        return img, label, grade


def ordinal_loss(pred_probs, target_labels, class_weights=None):
    """
    Binary cross-entropy per ordinal threshold.
    Naturally penalises distant-grade errors more.
    """
    bce = nn.BCELoss(reduction='none')
    loss = bce(pred_probs, target_labels)

    if class_weights is not None:
        loss = loss * class_weights.unsqueeze(0)

    return loss.mean()


def compute_qwk(preds, targets):
    """Quadratic Weighted Kappa — main evaluation metric."""
    return cohen_kappa_score(targets, preds, weights='quadratic')


def train_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for imgs, labels, grades in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        probs = model(imgs)
        loss = ordinal_loss(probs, labels, class_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred_grades = model.ordinal_head.decode_grade(probs)
        all_preds.extend(pred_grades.cpu().numpy())
        all_targets.extend(grades.numpy())

    qwk = compute_qwk(all_preds, all_targets)
    return total_loss / len(loader), qwk


def validate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, labels, grades in loader:
            imgs = imgs.to(device)
            probs = model(imgs)
            pred_grades = model.ordinal_head.decode_grade(probs)
            all_preds.extend(pred_grades.cpu().numpy())
            all_targets.extend(grades.numpy())

    return compute_qwk(all_preds, all_targets)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # APTOS class distribution: Grade 0=49%, Grade 4=5%
    # Inverse frequency weights for BCE loss
    class_weights = torch.tensor([1.0, 3.0, 2.5, 4.0], device=device)

    from preprocessing import get_train_transform, get_inference_transform
    train_ds = DRDataset(f"{args.data_dir}/train/labels.csv",
                          f"{args.data_dir}/train/images",
                          transform=get_train_transform())
    val_ds = DRDataset(f"{args.data_dir}/val/labels.csv",
                        f"{args.data_dir}/val/images",
                        transform=get_inference_transform())

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    model = get_model(dropout_rate=0.3, pretrained=True).to(device)

    # ── STAGE 1: Freeze backbone, train head only ───────────────────────────
    print("\n=== Stage 1: Head-only training ===")
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.ordinal_head.parameters(), lr=1e-3)

    for epoch in range(args.epochs_s1):
        loss, qwk = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_qwk = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs_s1} | Loss: {loss:.4f} | Train QWK: {qwk:.4f} | Val QWK: {val_qwk:.4f}")

    # ── STAGE 2: Unfreeze all, fine-tune ────────────────────────────────────
    print("\n=== Stage 2: Full fine-tuning ===")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_s2, eta_min=1e-7)

    best_qwk = 0
    for epoch in range(args.epochs_s2):
        loss, qwk = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_qwk = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs_s2} | Loss: {loss:.4f} | Train QWK: {qwk:.4f} | Val QWK: {val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  → Saved best model (QWK: {best_qwk:.4f})")

    print(f"\nTraining complete. Best Val QWK: {best_qwk:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--epochs_s1', type=int, default=5)
    parser.add_argument('--epochs_s2', type=int, default=20)
    args = parser.parse_args()
    main(args)
