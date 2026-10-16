Diabetic Retinopathy Grading System
Uncertainty-Aware Ordinal Deep Learning for Retinal Image Analysis

23CSE473 Neural Networks and Deep Learning — Course Project
Amrita Vishwa Vidyapeetham, Chennai

What This Project Does
This system automatically grades Diabetic Retinopathy (DR) severity from retinal fundus photographs into 5 levels (Grade 0–4) using deep learning. Unlike standard classifiers, it also tells you how confident it is — critical for clinical use.

Key Technical Contributions:

Frank-Hall Ordinal Regression — respects the natural ordering of DR severity (Grade 3 is worse than Grade 2, not just different)
Monte Carlo Dropout — runs inference 30 times to measure prediction uncertainty
Ben Graham Preprocessing — removes scanner/lighting bias so Indian (APTOS) and European (Messidor-2) images look similar to the model
Cross-Dataset Generalisation — trained on APTOS 2019 (India), evaluated on Messidor-2 (Europe)
Project Structure
dr_project/
├── src/
│   ├── model.py                  # EfficientNet-B4 + MC Dropout + Ordinal Head
│   ├── preprocessing.py          # Ben Graham preprocessing pipeline
│   ├── train.py                  # 2-stage training script
│   ├── evaluate.py               # Cross-dataset evaluation + ECE
│   └── generate_sample_images.py # Synthetic retinal images for demo
├── demo/
│   └── app.py                    # Streamlit web demo
├── sample_images/                # Auto-generated synthetic images (created on first run)
├── requirements.txt
└── README.md
Quick Setup
1. Install Dependencies
pip install -r requirements.txt
2. Generate Sample Images (for demo — no real dataset needed)
cd dr_project
python src/generate_sample_images.py
This creates 15 synthetic retinal images (3 per grade) in sample_images/.

3. Run the Demo App
streamlit run demo/app.py
Open http://localhost:8501 in your browser.

Training on Real Data (APTOS 2019 Dataset)
Download Data
Go to https://www.kaggle.com/c/aptos2019-blindness-detection
Download train.csv and train_images/
Organise as:
data/
  train/
    images/   (PNG files)
    labels.csv  (columns: id_code, diagnosis)
  val/
    images/
    labels.csv
Train
python src/train.py --data_dir data/ --epochs_s1 5 --epochs_s2 20
Best model saved as best_model.pth.

Evaluate on Messidor-2
python src/evaluate.py --model_path best_model.pth --data_dir data/messidor2/
Architecture Overview
Retinal Image (380×380)
       ↓
Ben Graham Preprocessing
(CLAHE + Gaussian subtract)
       ↓
EfficientNet-B4 Backbone
(pretrained ImageNet, 1792-dim features)
       ↓
MC Dropout (p=0.3)  ←── Active at inference too!
       ↓
Frank-Hall Ordinal Head (4 binary sigmoids)
       ↓
Grade 0–4 + Uncertainty Score
Frank-Hall Encoding
Grade	Label
0 (No DR)	[0, 0, 0, 0]
1 (Mild)	[1, 0, 0, 0]
2 (Moderate)	[1, 1, 0, 0]
3 (Severe)	[1, 1, 1, 0]
4 (Proliferative)	[1, 1, 1, 1]
Monte Carlo Dropout Uncertainty
Run forward pass 30 times with dropout ON
Uncertainty = variance across 30 predictions
Normalized to [0, 1] scale
> 0.3 → flag for human review
Expected Performance (with full APTOS training)
Metric	Target	Clinical Threshold
QWK (APTOS val)	~0.90	>0.80
QWK (Messidor-2 zero-shot)	~0.82	>0.75
ECE	<0.05	<0.10
DR Grading Scale
Grade	Name	Clinical Meaning
0	No DR	Normal — annual screening
1	Mild DR	Microaneurysms only — 1-year follow-up
2	Moderate DR	Hard exudates + haemorrhages — 6-month follow-up
3	Severe DR	Extensive haemorrhages — urgent referral
4	Proliferative DR	Neovascularisation — same-day ophthalmology
Key Papers This Work Builds On
Frank & Hall (2001) — Ordinal regression via binary decomposition
Gal & Ghahramani (2016) — MC Dropout as Bayesian approximation
Tan & Le (2019) — EfficientNet architecture
Graham (2015) — Retinal image preprocessing (Kaggle DR competition winner)
Selvaraju et al. (2017) — Grad-CAM visual explanations
Full 25-paper literature survey in the project presentation.

Demo Features
Tab 1 — Diagnose:

Upload any retinal image OR select a synthetic sample
Shows: DR grade, uncertainty score, confidence bar chart, triage decision
Grad-CAM heatmap overlay showing which regions the model looked at
Frank-Hall encoding display
Tab 2 — Architecture:

Full pipeline diagram
Training strategy explanation
Tab 3 — Literature:

Searchable table of all 25 reference papers
