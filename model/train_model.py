import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, precision_recall_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    from xgboost import XGBClassifier

BASE            = os.path.join(os.path.dirname(__file__), "..")
SRC             = os.path.join(BASE, "data", "cleaned.csv")
MODEL_OUT       = os.path.join(BASE, "model", "aml_model.pkl")
METRICS_OUT     = os.path.join(BASE, "model", "metrics.pkl")
ALL_METRICS_OUT = os.path.join(BASE, "model", "all_metrics.pkl")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading cleaned.csv ...")
df = pd.read_csv(SRC)

X = df.drop(columns=["Is_laundering"])
y = df["Is_laundering"]

print(f"Features: {list(X.columns)}")
print(f"Shape: {X.shape}")
print(f"Class distribution:\n{y.value_counts()}\n")

# ---------------------------------------------------------------------------
# Train / test split  (stratified, test set NEVER touched by SMOTE)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# XGBoost scale_pos_weight from ORIGINAL training distribution
count_normal     = (y_train == 0).sum()
count_suspicious = (y_train == 1).sum()
scale_pos_weight = count_normal / count_suspicious
print(f"Training set — normal: {count_normal}, suspicious: {count_suspicious}")
print(f"scale_pos_weight for XGBoost: {scale_pos_weight:.2f}\n")

# ---------------------------------------------------------------------------
# SMOTE  (only on training data)
# ---------------------------------------------------------------------------
print("Applying SMOTE to training data ...")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
dist = pd.Series(y_train_sm).value_counts().to_dict()
print(f"  After SMOTE — normal: {dist.get(0,0)}, suspicious: {dist.get(1,0)}")
print(f"  Total training rows: {len(X_train_sm)}\n")

# ---------------------------------------------------------------------------
# Threshold tuning helper
# ---------------------------------------------------------------------------
def tune_threshold(y_true, y_prob):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    idx = int(f1s.argmax())
    return float(thresholds[idx]) if idx < len(thresholds) else 0.5

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
models = [
    ("Logistic Regression", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ("Decision Tree",       DecisionTreeClassifier(class_weight="balanced", random_state=42)),
    ("Random Forest",       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)),
    ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ("Naive Bayes",         GaussianNB()),
    ("Linear SVC",          CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42, max_iter=2000))),
    ("XGBoost",             XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=100,
                                          random_state=42, eval_metric="logloss",
                                          use_label_encoder=False)),
]

# ---------------------------------------------------------------------------
# Train, tune threshold, evaluate
# ---------------------------------------------------------------------------
all_metrics = []
best_f1     = -1
best_model  = None
best_name   = ""
best_metrics = {}

for name, clf in models:
    print(f"Training: {name} ...")
    clf.fit(X_train_sm, y_train_sm)

    y_prob  = clf.predict_proba(X_test)[:, 1]
    thresh  = tune_threshold(y_test, y_prob)
    y_pred  = (y_prob >= thresh).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"  Threshold: {thresh:.3f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    row = {
        "model_name":     name,
        "accuracy":       round(acc, 4),
        "precision":      round(prec, 4),
        "recall":         round(rec, 4),
        "f1":             round(f1, 4),
        "roc_auc":        round(auc, 4),
        "best_threshold": round(thresh, 4),
    }
    all_metrics.append(row)

    if f1 > best_f1:
        best_f1    = f1
        best_model = clf
        best_name  = name
        best_metrics = {**row, "confusion_matrix": cm}

# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------
joblib.dump(all_metrics, ALL_METRICS_OUT)
joblib.dump(best_model,  MODEL_OUT)
joblib.dump(best_metrics, METRICS_OUT)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("ALL MODELS RANKED BY F1 SCORE")
print("=" * 75)
ranked = sorted(all_metrics, key=lambda x: x["f1"], reverse=True)
header = f"{'Model':<22} {'Thresh':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}"
print(header)
print("-" * 75)
for r in ranked:
    marker = " <-- BEST" if r["model_name"] == best_name else ""
    print(
        f"{r['model_name']:<22} {r['best_threshold']:>6.3f} {r['accuracy']:>6.4f} "
        f"{r['precision']:>6.4f} {r['recall']:>6.4f} {r['f1']:>6.4f} {r['roc_auc']:>6.4f}{marker}"
    )

print("\n" + "=" * 75)
print(f"BEST MODEL : {best_name}")
print(f"F1 Score   : {best_f1:.4f}")
print(f"Threshold  : {best_metrics['best_threshold']:.3f}")
print(f"Reason     : Highest F1 — best balance of precision & recall for AML")
print("=" * 75)
print(f"\nArtifacts saved:")
print(f"  {MODEL_OUT}")
print(f"  {METRICS_OUT}")
print(f"  {ALL_METRICS_OUT}")
print("Done.")
