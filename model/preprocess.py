import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

BASE    = os.path.join(os.path.dirname(__file__), "..")
SRC     = os.path.join(BASE, "data", "dataset.csv")
DST     = os.path.join(BASE, "data", "cleaned.csv")
ENC_OUT = os.path.join(BASE, "model", "encoders.pkl")

CAT_COLS = [
    "Payment_type",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
]

DROP_COLS = ["Time", "Date", "Sender_account", "Receiver_account", "Laundering_type"]

CHUNK_SIZE = 100_000

# --- Pass 1: collect all unique values per categorical column ---
print("Pass 1: scanning unique values for label encoders ...")
unique_vals = {col: set() for col in CAT_COLS}

for chunk in pd.read_csv(SRC, chunksize=CHUNK_SIZE):
    for col in CAT_COLS:
        if col in chunk.columns:
            unique_vals[col].update(chunk[col].dropna().unique())

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    le.fit(sorted(unique_vals[col]))
    encoders[col] = le
    print(f"  {col}: {len(le.classes_)} classes")

joblib.dump(encoders, ENC_OUT)
print(f"Encoders saved to {ENC_OUT}")

# --- Pass 2: process chunks ---
print("\nPass 2: processing chunks ...")
chunks_out = []

for i, chunk in enumerate(pd.read_csv(SRC, chunksize=CHUNK_SIZE)):
    chunk = chunk.drop_duplicates()
    chunk = chunk.dropna()

    # Datetime feature engineering
    chunk["datetime"] = pd.to_datetime(
        chunk["Date"].astype(str) + " " + chunk["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    chunk["Year"]  = chunk["datetime"].dt.year
    chunk["Month"] = chunk["datetime"].dt.month
    chunk["Day"]   = chunk["datetime"].dt.day
    chunk["Hour"]  = chunk["datetime"].dt.hour

    chunk = chunk.drop(columns=DROP_COLS + ["datetime"], errors="ignore")
    chunk = chunk.dropna()  # drop rows where datetime parsing produced NaN

    # Label encode categorical columns
    for col in CAT_COLS:
        if col in chunk.columns:
            le = encoders[col]
            known = set(le.classes_)
            chunk[col] = chunk[col].apply(
                lambda v: le.transform([v])[0] if v in known else -1
            )

    chunks_out.append(chunk)
    print(f"  Chunk {i+1}: {len(chunk)} rows")

df_clean = pd.concat(chunks_out, ignore_index=True)

print(f"\nSaving cleaned data to {DST} ...")
df_clean.to_csv(DST, index=False)

print(f"\nFinal shape: {df_clean.shape}")
print("Class distribution of Is_laundering:")
print(df_clean["Is_laundering"].value_counts())
print(df_clean["Is_laundering"].value_counts(normalize=True).round(6))
print("Done.")
