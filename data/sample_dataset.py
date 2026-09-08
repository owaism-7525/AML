import pandas as pd
import os
from sklearn.model_selection import train_test_split

SOURCE = os.path.join(os.path.dirname(__file__), "..", "..", "SAML-D.csv")
DEST   = os.path.join(os.path.dirname(__file__), "dataset.csv")

print("Loading full dataset from SAML-D.csv ...")
df = pd.read_csv(SOURCE)

print(f"\nOriginal shape: {df.shape}")
print("Original class distribution:")
print(df["Is_laundering"].value_counts())
print(df["Is_laundering"].value_counts(normalize=True).round(6))

SAMPLE_SIZE = 400_000

# Stratified sample using train_test_split — keeps class ratio intact
_, sampled = train_test_split(
    df,
    test_size=SAMPLE_SIZE,
    stratify=df["Is_laundering"],
    random_state=42,
)

sampled = sampled.reset_index(drop=True)

print(f"\nNew shape: {sampled.shape}")
print("New class distribution:")
print(sampled["Is_laundering"].value_counts())
print(sampled["Is_laundering"].value_counts(normalize=True).round(6))

print(f"\nSaving to {DEST} ...")
sampled.to_csv(DEST, index=False)

del df, sampled

# Final confirmation
final = pd.read_csv(DEST)
print("\n--- Final Confirmation ---")
print(f"Shape: {final.shape}")
print("Class distribution:")
print(final["Is_laundering"].value_counts())
size_mb = os.path.getsize(DEST) / (1024 * 1024)
print(f"File size: {size_mb:.1f} MB")
print("Done.")
