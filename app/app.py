import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH      = os.path.join(BASE, "model", "aml_model.pkl")
ENCODERS_PATH   = os.path.join(BASE, "model", "encoders.pkl")
METRICS_PATH    = os.path.join(BASE, "model", "metrics.pkl")
ALL_METRICS_PATH = os.path.join(BASE, "model", "all_metrics.pkl")

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model       = joblib.load(MODEL_PATH)
    encoders    = joblib.load(ENCODERS_PATH)
    metrics     = joblib.load(METRICS_PATH)
    all_metrics = joblib.load(ALL_METRICS_PATH)
    return model, encoders, metrics, all_metrics

model, encoders, metrics, all_metrics = load_artifacts()
THRESHOLD = metrics.get("best_threshold", 0.5)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAYMENT_TYPES = [
    "Credit card", "Debit card", "Cash Deposit",
    "Cash Withdrawal", "ACH", "Cheque", "Cross-border",
]
CURRENCIES = [
    "UK pounds", "US dollar", "Euro", "Indian rupee", "Mexican Peso",
    "Dirham", "Turkish lira", "Naira", "Pakistani rupee",
    "Swiss franc", "Albanian lek",
]
BANK_LOCATIONS = [
    "UK", "USA", "UAE", "France", "Spain",
    "Morocco", "Turkey", "Mexico", "Nigeria", "Albania", "Switzerland",
]
CAT_COLS = [
    "Payment_type", "Payment_currency", "Received_currency",
    "Sender_bank_location", "Receiver_bank_location",
]

def encode_value(col, val):
    le = encoders.get(col)
    if le is None:
        return 0
    if val in le.classes_:
        return int(le.transform([val])[0])
    return -1

# Must match the exact column order from cleaned.csv / training
FEATURE_COLS = [
    "Amount", "Payment_currency", "Received_currency",
    "Sender_bank_location", "Receiver_bank_location", "Payment_type",
    "Year", "Month", "Day", "Hour",
]

def build_feature_row(amount, payment_type, payment_currency,
                       received_currency, sender_loc, receiver_loc):
    now = datetime.now()
    row = {
        "Amount":                 amount,
        "Payment_currency":       encode_value("Payment_currency", payment_currency),
        "Received_currency":      encode_value("Received_currency", received_currency),
        "Sender_bank_location":   encode_value("Sender_bank_location", sender_loc),
        "Receiver_bank_location": encode_value("Receiver_bank_location", receiver_loc),
        "Payment_type":           encode_value("Payment_type", payment_type),
        "Year":  now.year,
        "Month": now.month,
        "Day":   now.day,
        "Hour":  now.hour,
    }
    return pd.DataFrame([row])[FEATURE_COLS]

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to",
    ["Home", "Single Prediction", "Batch Prediction", "Model Performance"],
)

# ===========================================================================
# PAGE 1 — Home
# ===========================================================================
if page == "Home":
    st.title("AML Transaction Monitoring System")
    st.subheader("SAML-D Dataset | FAST NUCES Lahore | Spring 2026")

    st.markdown("""
    This application uses machine learning to detect **suspicious (money laundering)**
    transactions in real-time. It was trained on the SAML-D dataset and supports
    single transaction analysis as well as bulk batch screening.
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", "9.4 Million")
    col2.metric("Features", "12")
    col3.metric("Suspicious Rate", "0.124%")

    st.markdown("---")
    st.markdown("""
    **How to use:**
    - **Single Prediction** — Enter transaction details and get an instant verdict
    - **Batch Prediction** — Upload a CSV file to screen multiple transactions at once
    - **Model Performance** — Compare all 8 trained models and view evaluation metrics
    """)

    best_name = metrics.get("model_name", "N/A")
    best_f1   = metrics.get("f1", 0)
    st.info(f"Active model: **{best_name}** | F1 Threshold: **{THRESHOLD:.3f}** | F1 Score: **{best_f1:.4f}**")

# ===========================================================================
# PAGE 2 — Single Prediction
# ===========================================================================
elif page == "Single Prediction":
    st.title("Single Transaction Prediction")
    st.markdown("Enter the transaction details below.")

    with st.form("prediction_form"):
        amount          = st.number_input("Amount", min_value=0.0, value=1000.0, step=0.01)
        payment_type    = st.selectbox("Payment Type", PAYMENT_TYPES)
        payment_curr    = st.selectbox("Payment Currency", CURRENCIES)
        received_curr   = st.selectbox("Received Currency", CURRENCIES)
        sender_loc      = st.selectbox("Sender Bank Location", BANK_LOCATIONS)
        receiver_loc    = st.selectbox("Receiver Bank Location", BANK_LOCATIONS)
        submitted       = st.form_submit_button("Predict")

    if submitted:
        X_input    = build_feature_row(
            amount, payment_type, payment_curr,
            received_curr, sender_loc, receiver_loc,
        )
        proba      = model.predict_proba(X_input)[0]
        prediction = int(proba[1] >= THRESHOLD)
        confidence = float(proba[prediction])

        st.markdown("---")
        if prediction == 1:
            st.error("🔴 SUSPICIOUS TRANSACTION")
        else:
            st.success("🟢 NORMAL TRANSACTION")

        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
        st.progress(confidence)

        col1, col2 = st.columns(2)
        col1.metric("Normal Probability",     f"{proba[0] * 100:.2f}%")
        col2.metric("Suspicious Probability", f"{proba[1] * 100:.2f}%")

# ===========================================================================
# PAGE 3 — Batch Prediction
# ===========================================================================
elif page == "Batch Prediction":
    st.title("Batch Transaction Prediction")
    st.markdown("Upload a CSV file with transaction data to screen multiple records at once.")
    st.markdown(
        "**Expected columns:** Amount, Payment_type, Payment_currency, Received_currency, "
        "Sender_bank_location, Receiver_bank_location *(Date/Time optional)*"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.subheader("Preview (first 5 rows)")
        st.dataframe(df_batch.head())

        if st.button("Run Predictions"):
            df_proc = df_batch.copy()

            # Encode categorical columns
            for col in CAT_COLS:
                if col in df_proc.columns:
                    le = encoders.get(col)
                    if le:
                        known = set(le.classes_)
                        df_proc[col] = df_proc[col].apply(
                            lambda v: int(le.transform([v])[0]) if v in known else -1
                        )

            # Fill datetime features
            now = datetime.now()
            for col, default in [("Year", now.year), ("Month", now.month),
                                  ("Day", now.day), ("Hour", now.hour)]:
                if col not in df_proc.columns:
                    df_proc[col] = default
                elif df_proc[col].dtype == object:
                    df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce").fillna(default)

            present = [c for c in FEATURE_COLS if c in df_proc.columns]
            X_batch = df_proc[present]

            probas = model.predict_proba(X_batch)[:, 1]
            preds  = (probas >= THRESHOLD).astype(int)

            df_batch["Prediction"]  = preds
            df_batch["Confidence"]  = (probas * 100).round(2)
            df_batch["Label"]       = df_batch["Prediction"].map({0: "Normal", 1: "Suspicious"})

            st.subheader("Results")
            n_sus = (preds == 1).sum()
            col1, col2 = st.columns(2)
            col1.metric("Total Transactions", len(preds))
            col2.metric("Suspicious Detected", n_sus)

            def highlight_suspicious(row):
                color = "background-color: #ffcccc" if row["Prediction"] == 1 else ""
                return [color] * len(row)

            st.dataframe(df_batch.style.apply(highlight_suspicious, axis=1))

            csv_out = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results as CSV",
                data=csv_out,
                file_name="aml_predictions.csv",
                mime="text/csv",
            )

# ===========================================================================
# PAGE 4 — Model Performance
# ===========================================================================
elif page == "Model Performance":
    st.title("Model Performance Comparison")
    st.info(
        "Recall and F1 Score are prioritized over Accuracy due to severe class imbalance "
        "(0.124% suspicious transactions)"
    )

    # Build comparison dataframe
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics = df_metrics.rename(columns={
        "model_name": "Model",
        "accuracy":   "Accuracy",
        "precision":  "Precision",
        "recall":     "Recall",
        "f1":         "F1 Score",
        "roc_auc":    "ROC AUC",
    })
    df_metrics = df_metrics.sort_values("F1 Score", ascending=False).reset_index(drop=True)

    best_model_name = metrics.get("model_name", "")

    def highlight_best(row):
        if row["Model"] == best_model_name:
            return ["background-color: #90EE90"] * len(row)
        return [""] * len(row)

    st.subheader("All Models — Ranked by F1 Score")
    st.dataframe(df_metrics.style.apply(highlight_best, axis=1), width=1200)

    # Bar charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric_col in zip(axes, ["F1 Score", "Recall", "ROC AUC"]):
        colors = [
            "#2ecc71" if m == best_model_name else "#3498db"
            for m in df_metrics["Model"]
        ]
        ax.barh(df_metrics["Model"], df_metrics[metric_col], color=colors)
        ax.set_xlabel(metric_col)
        ax.set_title(f"{metric_col} by Model")
        ax.set_xlim(0, 1)
        for i, v in enumerate(df_metrics[metric_col]):
            ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Confusion matrix
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        st.subheader(f"Confusion Matrix — {best_model_name}")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
            xticklabels=["Normal", "Suspicious"],
            yticklabels=["Normal", "Suspicious"],
        )
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)
        plt.close(fig2)

    # Feature importance (if supported)
    best_clf = joblib.load(MODEL_PATH)
    if hasattr(best_clf, "feature_importances_"):
        st.subheader(f"Feature Importance — {best_model_name}")
        feature_names = FEATURE_COLS
        importance = best_clf.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        fi_df = fi_df.sort_values("Importance", ascending=True)

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.barh(fi_df["Feature"], fi_df["Importance"], color="#3498db")
        ax3.set_xlabel("Importance")
        ax3.set_title("Feature Importance")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
