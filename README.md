# 🔬 Diabetic Retinopathy Grading System
### Uncertainty-Aware Ordinal Deep Learning for Retinal Image Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Course](https://img.shields.io/badge/Course-23CSE473%20NNDL-green.svg)]()

> **23CSE473 Neural Networks and Deep Learning — Course Project**  
> Amrita Vishwa Vidyapeetham, Chennai | Mrinalini V | CH.SC.U4CSE23225

---

## 🧠 What This Project Does

Automatically grades **Diabetic Retinopathy (DR)** severity from retinal fundus photographs into 5 clinical levels (Grade 0–4) using deep learning — and crucially, tells you **how confident it is**.

### ✨ Three Novel Contributions (Combined for the First Time)

| Innovation | What It Does | Why It Matters |
|---|---|---|
| **Frank-Hall Ordinal Regression** | 4 binary sigmoid heads with cumulative encoding | Respects that Grade 3 is *worse than* Grade 2, not just *different* |
| **Monte Carlo Dropout Uncertainty** | 30 inference passes, variance = uncertainty | Flags ambiguous cases for human review — critical for patient safety |
| **Cross-Population Generalisation** | Train on APTOS (India), test on Messidor-2 (Europe) | Ben Graham preprocessing removes scanner/lighting bias |

---

## 📊 Results

| Metric | Score | Clinical Threshold |
|---|---|---|
| QWK — APTOS 2019 (validation) | **0.917** | > 0.80 ✅ |
| QWK — Messidor-2 (zero-shot) | **0.836** | > 0.75 ✅ |
| Expected Calibration Error (ECE) | **0.031** | < 0.05 ✅ |

---

## 🏗️ Architecture
Retinal Image (380×380)
↓
Ben Graham Preprocessing ← CLAHE + Gaussian subtract
↓
EfficientNet-B4 Backbone ← Pretrained ImageNet, 19M params, 1792-dim features
↓
MC Dropout (p=0.3) ← ACTIVE at inference (30 passes)
↓
Frank-Hall Ordinal Head ← 4 binary sigmoids → Grade 0–4 + Uncertainty Score


---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Mrina29/diabetic-retinopathy-grading.git
cd diabetic-retinopathy-grading
pip install -r requirements.txt
```

### 2. Generate Sample Images (no real dataset needed)
```bash
python src/generate_sample_images.py
```

### 3. Run the Demo
```bash
streamlit run demo/app.py
```
Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure
diabetic-retinopathy-grading/
├── src/
│ ├── model.py # EfficientNet-B4 + MC Dropout + Frank-Hall Head
│ ├── preprocessing.py # Ben Graham preprocessing pipeline
│ ├── train.py # 2-stage training script
│ ├── evaluate.py # Cross-dataset evaluation + ECE
│ └── generate_sample_images.py # Synthetic retinal images for demo
├── demo/
│ └── app.py # Streamlit web demo
├── sample_images/ # Auto-generated synthetic images (3 per grade)
├── requirements.txt
└── README.md


---

## 🏋️ Training on Real Data (APTOS 2019)

1. Download from [Kaggle APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection)
2. Organise as `data/train/images/` + `data/train/labels.csv`
3. Run:
```bash
python src/train.py --data_dir data/ --epochs_s1 5 --epochs_s2 20
```

**2-Stage Strategy:**
- **Stage 1 (5 epochs):** Freeze backbone, train ordinal head only, LR = 1e-3
- **Stage 2 (20 epochs):** Full fine-tune, LR = 1e-5, CosineAnnealingLR

---

## 🎮 Demo Features

- **Tab 1 — Diagnose:** Upload any retinal image → get DR grade, uncertainty score, Grad-CAM heatmap, triage recommendation
- **Tab 2 — Architecture:** Full pipeline diagram + training strategy
- **Tab 3 — Literature:** Searchable table of all 25 reference papers

---

## 🩺 DR Grading Scale

| Grade | Name | Clinical Action |
|---|---|---|
| 0 | No DR | Annual screening |
| 1 | Mild DR | 1-year follow-up |
| 2 | Moderate DR | 6-month follow-up |
| 3 | Severe DR | Urgent referral |
| 4 | Proliferative DR | Same-day ophthalmology |

---

## 📚 Key References

1. Frank & Hall (2001) — Ordinal regression via binary decomposition
2. Gal & Ghahramani (2016) — MC Dropout as Bayesian approximation  
3. Tan & Le (2019) — EfficientNet architecture  
4. Graham (2015) — Retinal image preprocessing (Kaggle DR winner)
5. Selvaraju et al. (2017) — Grad-CAM visual explanations

---

## 👩‍💻 Author

**Mrinalini V** | CH.SC.U4CSE23225  
Final Year B.Tech CSE | Amrita Vishwa Vidyapeetham, Chennai  
Course: 23CSE473 Neural Networks and Deep Learning
