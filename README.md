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
