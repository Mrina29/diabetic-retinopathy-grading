"""
evaluate.py
Cross-dataset evaluation script for DR grading model.
Evaluates trained model on Messidor-2 (cross-domain generalisation).

Usage:
  python src/evaluate.py --model_path best_model.pth --data_dir data/messidor2/
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                              classification_report)

from model import get_model
from preprocessing import preprocess_pil_image


def load_dataset(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    return df, Path(img_dir)


def predict_batch(model, img_paths, device, n_passes=30):
    """Run MC Dropout inference on a list of image paths."""
    all_grades, all_uncertainties, all_grade_probs = [], [], []

    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        tensor = preprocess_pil_image(img).to(device)

        grade, uncertainty, grade_probs = model.predict_with_uncertainty(
            tensor, n_passes=n_passes)

        all_grades.append(grade.item())
        all_uncertainties.append(uncertainty.item())
        all_grade_probs.append(grade_probs.cpu().numpy())

    return all_grades, all_uncertainties, all_grade_probs


def compute_ece(grade_probs_list, true_grades, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Measures how well predicted probabilities match actual accuracy.
    Lower is better. Well-calibrated model → ECE < 0.05.
    """
    confidences = []
    accuracies = []

    for grade_probs, true_grade in zip(grade_probs_list, true_grades):
        probs = grade_probs[0]  # shape (5,)
        pred_grade = np.argmax(probs)
        conf = probs[pred_grade]
        correct = int(pred_grade == true_grade)
        confidences.append(conf)
        accuracies.append(correct)

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)

    ece /= len(confidences)
    return ece


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = get_model(dropout_rate=0.3, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")

    # Load dataset
    df, img_dir = load_dataset(
        f"{args.data_dir}/labels.csv",
        f"{args.data_dir}/images"
    )

    img_paths = [img_dir / f"{row['id_code']}.png" for _, row in df.iterrows()]
    true_grades = df['diagnosis'].tolist()

    print(f"Evaluating on {len(img_paths)} images...")

    # Predict
    pred_grades, uncertainties, grade_probs = predict_batch(
        model, img_paths, device, n_passes=args.n_passes)

    # Metrics
    qwk = cohen_kappa_score(true_grades, pred_grades, weights='quadratic')
    ece = compute_ece(grade_probs, true_grades)

    print(f"\n{'='*50}")
    print(f"Dataset: {args.data_dir}")
    print(f"{'='*50}")
    print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Mean Uncertainty: {np.mean(uncertainties):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        true_grades, pred_grades,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']))

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(true_grades, pred_grades)
    print(cm)

    # High-uncertainty cases
    high_unc = [(i, u) for i, u in enumerate(uncertainties) if u > args.unc_threshold]
    print(f"\nHigh-uncertainty cases (>{args.unc_threshold}): {len(high_unc)}/{len(pred_grades)}")

    # Save results
    results_df = pd.DataFrame({
        'image': [p.name for p in img_paths],
        'true_grade': true_grades,
        'pred_grade': pred_grades,
        'uncertainty': uncertainties
    })
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nResults saved to evaluation_results.csv")

    return qwk, ece


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='best_model.pth')
    parser.add_argument('--data_dir', default='data/messidor2/')
    parser.add_argument('--n_passes', type=int, default=30,
                        help='MC Dropout passes for uncertainty')
    parser.add_argument('--unc_threshold', type=float, default=0.3,
                        help='Uncertainty threshold for flagging')
    args = parser.parse_args()
    main(args)
