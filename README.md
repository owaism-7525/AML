# AML Transaction Monitoring System

**SAML-D Dataset | FAST NUCES Lahore | Spring 2026**

A machine learning system for detecting suspicious (money laundering) transactions, built on the SAML-D dataset. Includes a full training pipeline and a 4-page Streamlit web application.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Setup & Installation](#4-setup--installation)
5. [Pipeline — Step by Step](#5-pipeline--step-by-step)
6. [Experiment Log & Results](#6-experiment-log--results)
7. [Final Model Results](#7-final-model-results)
8. [Streamlit App](#8-streamlit-app)
9. [What We Learned](#9-what-we-learned)
10. [Potential Next Steps](#10-potential-next-steps)

---

## 1. Project Overview

This project builds an end-to-end Anti-Money Laundering (AML) detection system:

- **Data:** Stratified sample of the SAML-D dataset (9.4M transactions)
- **Goal:** Classify each transaction as Normal (0) or Suspicious / Laundering (1)
- **Challenge:** Extreme class imbalance — only ~0.104% of transactions are suspicious
- **Approach:** 8 ML models trained with SMOTE oversampling + per-model threshold tuning
- **Deployment:** Interactive Streamlit app for single and batch prediction

---

## 2. Dataset

**Source file:** `SAML-D.csv` (~950 MB, 9,504,852 rows)

| Column | Description |
|--------|-------------|
| Time | Transaction time (HH:MM:SS) |
| Date | Transaction date (YYYY-MM-DD) |
| Sender_account | Sender account ID |
| Receiver_account | Receiver account ID |
| Amount | Transaction amount |
| Payment_currency | Currency used to pay |
| Received_currency | Currency received |
| Sender_bank_location | Country of sender's bank |
| Receiver_bank_location | Country of receiver's bank |
| Payment_type | Type of payment method |
| Is_laundering | **Target** — 0 = Normal, 1 = Suspicious |
| Laundering_type | Type of laundering scheme (dropped — data leakage) |

**Full dataset class distribution:**
- Normal: 9,494,979 (99.896%)
- Suspicious: 9,873 (0.104%)

---

## 3. Project Structure

```
aml-assessment/
├── data/
│   ├── sample_dataset.py     # Stratified sampling from SAML-D.csv
│   ├── dataset.csv           # 400k-row sampled dataset (generated)
│   └── cleaned.csv           # Preprocessed features (generated)
├── model/
│   ├── preprocess.py         # Feature engineering + label encoding
│   ├── train_model.py        # 8-model training with SMOTE + threshold tuning
│   ├── encoders.pkl          # Saved LabelEncoders (generated)
│   ├── aml_model.pkl         # Best trained model (generated)
│   ├── metrics.pkl           # Best model metrics + threshold (generated)
│   └── all_metrics.pkl       # All 8 model metrics (generated)
├── app/
│   └── app.py                # Streamlit 4-page web application
├── notebooks/                # (reserved for exploration)
└── requirements.txt          # Python dependencies
```

---

## 4. Setup & Installation

**Prerequisites:** Python 3.14 managed via `uv`

```powershell
# From the project root (d:\Aml_assement\)

# Create virtual environment
uv venv aml-assessment\.venv

# Install all dependencies
uv pip install -r aml-assessment\requirements.txt
```

**Dependencies (`requirements.txt`):**
```
pandas
scikit-learn
imbalanced-learn
streamlit
matplotlib
seaborn
joblib
numpy
xgboost
```

---

## 5. Pipeline — Step by Step

Run all commands from `d:\Aml_assement\`:

### Step 1 — Sample the Dataset
```powershell
aml-assessment\.venv\Scripts\python.exe aml-assessment\data\sample_dataset.py
```
Reads `SAML-D.csv`, takes a stratified 400,000-row sample, saves to `data/dataset.csv`.

### Step 2 — Preprocess
```powershell
aml-assessment\.venv\Scripts\python.exe aml-assessment\model\preprocess.py
```
Cleans data, engineers features, label-encodes categoricals, saves `cleaned.csv` and `encoders.pkl`.

### Step 3 — Train Models
```powershell
aml-assessment\.venv\Scripts\python.exe aml-assessment\model\train_model.py
```
Trains 8 models with SMOTE + threshold tuning. Saves `aml_model.pkl`, `metrics.pkl`, `all_metrics.pkl`.

### Step 4 — Launch App
```powershell
aml-assessment\.venv\Scripts\streamlit.exe run aml-assessment\app\app.py
```
Opens at **http://localhost:8501**

---

## 6. Experiment Log & Results

### Experiment 1 — Baseline: 50,000 rows, no oversampling

**Problem:** Only 52 suspicious rows in 50k sample. Tree-based models predicted all-Normal.

| Model | Accuracy | F1 | Recall |
|-------|----------|----|--------|
| Logistic Regression | 0.6848 | 0.0025 | 0.40 |
| SVM | 0.6244 | 0.0021 | 0.40 |
| Random Forest | 0.9990 | 0.0000 | 0.00 |
| XGBoost | 0.9979 | 0.0000 | 0.00 |
| Decision Tree | 0.9976 | 0.0000 | 0.00 |
| Gradient Boosting | 0.9974 | 0.0000 | 0.00 |
| Naive Bayes | 0.9451 | 0.0000 | 0.00 |
| KNN | 0.9990 | 0.0000 | 0.00 |

**Root cause:** 52 suspicious samples is far too few. Models exploit the 99.9% majority to maximise accuracy while ignoring the minority class entirely.

**Changes made:**
- Increased sample size to **400,000 rows** → ~415 suspicious
- Added **SMOTE** oversampling on training data only
- Replaced slow **SVM** with **LinearSVC** (wrapped in CalibratedClassifierCV)
- Added **per-model threshold tuning** using precision-recall curves

---

### Experiment 2 — 400k rows + SMOTE + threshold tuning

SMOTE balanced the training set from 332 suspicious → 319,668 suspicious (matching majority class). Threshold tuning found the optimal decision boundary per model.

| Model | Threshold | F1 | Recall | ROC AUC |
|-------|-----------|-----|--------|---------|
| **XGBoost** | 1.000 | **0.0920** | 0.048 | 0.631 |
| Gradient Boosting | 0.960 | 0.0899 | 0.048 | **0.767** |
| Random Forest | 0.850 | 0.0667 | 0.048 | 0.627 |
| Logistic Regression | 0.986 | 0.0260 | 0.024 | 0.661 |
| Linear SVC | 0.979 | 0.0260 | 0.024 | 0.661 |
| Naive Bayes | 1.000 | 0.0244 | 0.036 | 0.647 |
| Decision Tree | 1.000 | 0.0171 | 0.048 | 0.522 |
| KNN | 1.000 | 0.0076 | 0.108 | 0.567 |

**Best model:** XGBoost (F1 = 0.0920)
**Observation:** F1 improved but recall stayed very low (~5%). Gradient Boosting had the best ROC AUC (0.767).

---

### Experiment 3 — SMOTENC (switched from SMOTE)

**Why tried:** SMOTE interpolates between label-encoded integers for categorical features (e.g., averaging "US dollar"=5 and "UK pounds"=3 gives 4.0, a currency that doesn't exist). SMOTENC uses mode-picking for categoricals instead.

**Categorical feature indices:** `[1, 2, 3, 4, 5]` → Payment_currency, Received_currency, Sender_bank_location, Receiver_bank_location, Payment_type

| Model | F1 | Recall | ROC AUC |
|-------|----|--------|---------|
| Naive Bayes | 0.0244 | 0.036 | 0.461 |
| Random Forest | 0.0205 | 0.072 | 0.608 |
| Gradient Boosting | 0.0135 | 0.181 | 0.615 |
| XGBoost | 0.0123 | 0.048 | 0.630 |

**Result:** SMOTENC performed **worse** than SMOTE across all models. F1 dropped from 0.092 to 0.024.

**Why:** Mode-picking for categoricals produced synthetic minority samples with very little diversity — most ended up as near-identical copies of existing suspicious rows, giving models no new signal to learn from.

**Decision:** Reverted to SMOTE (better performer on this dataset).

---

### Bugs Fixed During Development

| Bug | Cause | Fix |
|-----|-------|-----|
| `KeyError: 'Is_laundering'` in sampling | `groupby.apply` in pandas 2.x drops the groupby key column | Switched to `train_test_split(stratify=...)` |
| 60% of rows dropped in preprocessing | `dayfirst=True` in `pd.to_datetime` failed for `YYYY-MM-DD` format | Fixed to `format="%Y-%m-%d %H:%M:%S"` |
| `ValueError: feature names mismatch` in app | Feature column order in app differed from training order | Added `FEATURE_COLS` constant matching training order |
| `use_container_width` deprecation warning | Streamlit 1.57 removed this parameter | Replaced with `width=` |

---

## 7. Final Model Results

**Configuration:** 400,000 rows · SMOTE · Threshold tuning · XGBoost selected

**Training set after SMOTE:**
- Normal: 319,668
- Suspicious: 319,668 (synthetically balanced)
- Total: 639,336 rows

**Test set (original distribution, no SMOTE):**
- Normal: 79,917
- Suspicious: 83

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC AUC |
|-------|-----------|----------|-----------|--------|----|---------|
| **XGBoost** ✅ | 1.000 | 0.9990 | 1.0000 | 0.0482 | **0.0920** | 0.6314 |
| Gradient Boosting | 0.960 | 0.9990 | 0.6667 | 0.0482 | 0.0899 | **0.7667** |
| Random Forest | 0.850 | 0.9986 | 0.1081 | 0.0482 | 0.0667 | 0.6266 |
| Logistic Regression | 0.986 | 0.9981 | 0.0282 | 0.0241 | 0.0260 | 0.6608 |
| Linear SVC | 0.979 | 0.9981 | 0.0282 | 0.0241 | 0.0260 | 0.6606 |
| Naive Bayes | 1.000 | 0.9970 | 0.0184 | 0.0361 | 0.0244 | 0.6466 |
| Decision Tree | 1.000 | 0.9942 | 0.0104 | 0.0482 | 0.0171 | 0.5217 |
| KNN | 1.000 | 0.9706 | 0.0039 | 0.1084 | 0.0076 | 0.5667 |

**XGBoost Confusion Matrix (test set):**
```
              Predicted Normal  Predicted Suspicious
Actual Normal       79,917              0
Actual Suspicious       79              4
```

> **Note:** F1 scores are low due to extreme class imbalance (0.104% suspicious). Accuracy is misleading here — a model that predicts everything as Normal gets 99.9% accuracy. F1 and Recall are the metrics that matter.

---

## 8. Streamlit App

**Launch:**
```powershell
aml-assessment\.venv\Scripts\streamlit.exe run aml-assessment\app\app.py
```
Opens at **http://localhost:8501**

### Page 1 — Home
- Project title and institution info
- Dataset statistics (9.4M transactions, 12 features, 0.104% suspicious)
- Active model name, decision threshold, and F1 score

### Page 2 — Single Prediction
- Input form: Amount, Payment Type, Currencies, Bank Locations
- Outputs: SUSPICIOUS / NORMAL verdict with confidence bar
- Uses the tuned decision threshold (not default 0.5)

### Page 3 — Batch Prediction
- Upload a CSV file of transactions
- Encodes and scores all rows using the trained model
- Highlights suspicious rows in red
- Download results as CSV

### Page 4 — Model Performance
- Full comparison table of all 8 models ranked by F1
- Best model row highlighted in green
- Bar charts: F1 Score, Recall, ROC AUC per model
- Confusion matrix heatmap
- Feature importance chart (XGBoost supports this)

---

## 9. What We Learned

1. **Accuracy is misleading for imbalanced data.** A 99.9% accuracy model that detects zero suspicious transactions is useless. F1 and Recall are the right metrics.

2. **Oversampling alone is not enough.** SMOTE improved F1 from ~0 to ~0.09 but recall stayed at ~5%. The models are still missing 95% of suspicious transactions.

3. **SMOTENC ≠ always better than SMOTE.** Despite being theoretically correct for categorical features, SMOTENC underperformed because it produced low-diversity synthetic samples.

4. **The features are the real bottleneck.** With only Amount, currencies, locations, payment type, and time — the models have limited discriminative power. The ROC AUC scores (0.43–0.77) show many models are close to random.

5. **Threshold tuning matters.** Default 0.5 threshold is wrong for imbalanced problems. Per-model tuning using precision-recall curves lets each model find its optimal operating point.

---

## 10. Potential Next Steps

To significantly improve recall (the key metric for AML), the following improvements are recommended in priority order:

### A. Feature Engineering (highest impact)
Add derived features before training:

| Feature | Code | Why |
|---------|------|-----|
| Same currency flag | `Payment_currency == Received_currency` | Laundering often involves currency conversion |
| Cross-border flag | `Sender_bank_location != Receiver_bank_location` | High-risk indicator |
| Round amount flag | `Amount % 1000 == 0` | Structuring / smurfing pattern |
| High-risk country flag | Known FATF grey-list countries | Regulatory risk signal |
| Amount z-score | Per payment type deviation | Statistical anomaly detection |

### B. Use Full Dataset
Training on all 9.4M rows (with ~9,873 suspicious) gives models far more minority-class signal. Requires removing SVM/KNN due to scaling issues.

### C. Focal Loss / Cost-Sensitive Learning
Use asymmetric loss functions (XGBoost supports this via `scale_pos_weight`) that penalise false negatives more heavily than false positives.

### D. Isolation Forest / Anomaly Detection
Frame suspicious detection as one-class classification — train only on normal transactions and flag anomalies. No class imbalance problem at all.
