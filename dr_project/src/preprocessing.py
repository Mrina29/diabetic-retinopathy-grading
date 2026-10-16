"""
preprocessing.py
Ben Graham preprocessing pipeline for retinal fundus images.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def ben_graham_preprocess(img: np.ndarray, image_size: int = 380) -> np.ndarray:
    """
    Ben Graham's retinal image preprocessing:
    1. Crop black borders using circular mask
    2. CLAHE contrast enhancement
    3. Subtract local average colour to reduce illumination bias

    This makes Indian (APTOS) and European (Messidor-2) images
    more similar by removing scanner-specific lighting differences.
    """
    if img is None or img.size == 0:
        raise ValueError("Empty image provided")

    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # Step 1: Resize keeping aspect ratio, pad to square
    h, w = img.shape[:2]
    scale = image_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    # Pad to square
    top = (image_size - new_h) // 2
    bottom = image_size - new_h - top
    left = (image_size - new_w) // 2
    right = image_size - new_w - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Step 2: Circular mask — crops black border typical in fundus images
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cx, cy = image_size // 2, image_size // 2
    radius = int(image_size * 0.47)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    # Step 3: CLAHE on L channel (illumination normalization)
    lab = cv2.cvtColor(img_padded, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    # Step 4: Subtract local average (Ben Graham's key trick)
    # Reduces scanner-specific global illumination
    img_blur = cv2.GaussianBlur(img_clahe, (0, 0), image_size / 30)
    img_processed = cv2.addWeighted(img_clahe, 4, img_blur, -4, 128)

    # Apply circular mask
    img_processed = cv2.bitwise_and(img_processed, img_processed,
                                     mask=mask)

    return img_processed.astype(np.uint8)


def get_inference_transform(image_size: int = 380):
    """Transform pipeline for model inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_train_transform(image_size: int = 380):
    """Augmented transform pipeline for training."""
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_pil_image(pil_img: Image.Image,
                          image_size: int = 380) -> torch.Tensor:
    """
    Full pipeline: PIL Image → preprocessed tensor ready for model.
    """
    # Convert PIL to numpy
    img_np = np.array(pil_img.convert('RGB'))

    # Ben Graham preprocessing
    img_processed = ben_graham_preprocess(img_np, image_size)

    # Convert back to PIL for transforms
    img_pil = Image.fromarray(img_processed)

    # Apply inference transform
    transform = get_inference_transform(image_size)
    tensor = transform(img_pil)

    return tensor.unsqueeze(0)  # Add batch dimension


def frank_hall_encode(grade: int, num_classes: int = 5) -> list:
    """
    Encode a DR grade as Frank-Hall cumulative binary label.
    Grade 0 → [0, 0, 0, 0]
    Grade 1 → [1, 0, 0, 0]
    Grade 2 → [1, 1, 0, 0]
    Grade 3 → [1, 1, 1, 0]
    Grade 4 → [1, 1, 1, 1]
    """
    label = [1 if i < grade else 0 for i in range(num_classes - 1)]
    return label
