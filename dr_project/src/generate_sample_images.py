"""
generate_sample_images.py
Generates synthetic retinal fundus-like images for demo purposes.
Each image simulates DR features for grades 0-4.
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path


def draw_retinal_base(size=380):
    """Create realistic retinal background with circular fundus."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    radius = int(size * 0.46)

    # Dark reddish fundus background
    cv2.circle(img, (cx, cy), radius, (80, 40, 30), -1)

    # Add texture gradient
    for r in range(radius, 0, -10):
        alpha = r / radius
        color = (int(80 + 40 * alpha), int(40 + 20 * alpha), int(30 + 10 * alpha))
        cv2.circle(img, (cx, cy), r, color, 8)

    # Optic disc (bright yellowish circle)
    od_x = cx + int(radius * 0.35)
    od_y = cy - int(radius * 0.05)
    cv2.circle(img, (od_x, od_y), int(size * 0.06), (200, 180, 100), -1)
    cv2.circle(img, (od_x, od_y), int(size * 0.04), (220, 210, 160), -1)

    # Macula (darker central area)
    cv2.circle(img, (cx, cy), int(size * 0.05), (60, 30, 25), -1)

    # Blood vessels (main branches)
    vessel_color = (40, 20, 15)
    pts = np.array([[od_x, od_y], [cx - 30, cy - 20], [cx - 80, cy + 30]], dtype=np.int32)
    cv2.polylines(img, [pts], False, vessel_color, 2)
    pts2 = np.array([[od_x, od_y], [cx - 20, cy + 30], [cx - 70, cy + 80]], dtype=np.int32)
    cv2.polylines(img, [pts2], False, vessel_color, 2)
    pts3 = np.array([[od_x, od_y], [cx + 40, cy - 60]], dtype=np.int32)
    cv2.polylines(img, [pts3], False, vessel_color, 2)

    # Slight noise for realism
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def add_microaneurysms(img, count=5, size=380):
    """Small red dots — first sign of DR (Grade 1)."""
    cx, cy, radius = size // 2, size // 2, int(size * 0.4)
    for _ in range(count):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.1, 0.7) * radius
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))
        dot_size = np.random.randint(2, 4)
        cv2.circle(img, (x, y), dot_size, (180, 60, 60), -1)
    return img


def add_hard_exudates(img, count=8, size=380):
    """Bright yellow-white deposits — Grade 2."""
    cx, cy, radius = size // 2, size // 2, int(size * 0.4)
    for _ in range(count):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.1, 0.65) * radius
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))
        size_ex = np.random.randint(3, 8)
        cv2.circle(img, (x, y), size_ex, (220, 210, 150), -1)
    return img


def add_haemorrhages(img, count=6, size=380):
    """Dark red blotches — Grade 3."""
    cx, cy, radius = size // 2, size // 2, int(size * 0.4)
    for _ in range(count):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.05, 0.6) * radius
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))
        w_h = np.random.randint(6, 16)
        h_h = np.random.randint(4, 12)
        cv2.ellipse(img, (x, y), (w_h, h_h), np.random.randint(0, 180),
                    0, 360, (140, 30, 30), -1)
    return img


def add_neovascularization(img, count=4, size=380):
    """New abnormal blood vessels — Grade 4 (proliferative)."""
    cx, cy, radius = size // 2, size // 2, int(size * 0.4)
    for _ in range(count):
        x = np.random.randint(cx - radius // 2, cx + radius // 2)
        y = np.random.randint(cy - radius // 2, cy + radius // 2)
        for branch in range(np.random.randint(3, 7)):
            angle = np.random.uniform(0, 2 * np.pi)
            length = np.random.randint(15, 35)
            ex = int(x + length * np.cos(angle))
            ey = int(y + length * np.sin(angle))
            cv2.line(img, (x, y), (ex, ey), (200, 80, 80), 1)
        cv2.circle(img, (x, y), 3, (220, 100, 100), -1)
    return img


GRADE_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}


def generate_grade_image(grade: int, size: int = 380, seed: int = 42) -> np.ndarray:
    """Generate a synthetic retinal image for a given DR grade."""
    np.random.seed(seed)
    img = draw_retinal_base(size)

    if grade >= 1:
        img = add_microaneurysms(img, count=8 + grade * 3, size=size)
    if grade >= 2:
        img = add_hard_exudates(img, count=6 + grade * 2, size=size)
    if grade >= 3:
        img = add_haemorrhages(img, count=5 + grade * 3, size=size)
    if grade >= 4:
        img = add_neovascularization(img, count=6, size=size)

    # Slight blur for realism
    img = cv2.GaussianBlur(img, (3, 3), 0.5)

    return img


def save_sample_images(output_dir: str = "sample_images"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for grade in range(5):
        for variant in range(3):
            img_np = generate_grade_image(grade, seed=grade * 10 + variant)
            pil_img = Image.fromarray(img_np)
            fname = f"grade_{grade}_{GRADE_LABELS[grade].replace(' ', '_')}_v{variant+1}.png"
            fpath = str(Path(output_dir) / fname)
            pil_img.save(fpath)
            if grade not in paths:
                paths[grade] = fpath
    print(f"Saved {5 * 3} sample images to {output_dir}/")
    return paths


if __name__ == '__main__':
    save_sample_images("sample_images")
