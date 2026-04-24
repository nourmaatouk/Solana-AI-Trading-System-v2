from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sol_usdt_15m_features_h4.csv"

OUTPUT_X = PROJECT_ROOT / "data" / "processed" / "X.npy"
OUTPUT_Y = PROJECT_ROOT / "data" / "processed" / "y.npy"
OUTPUT_SCALER = PROJECT_ROOT / "data" / "processed" / "scaler.pkl"
OUTPUT_CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "sol_usdt_15m_features_h4_clean.csv"

SEQUENCE_LENGTH = 32


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_PATH)

print("Loaded shape:", df.shape)


# =========================================================
# SELECT FEATURES
# =========================================================
DROP_COLUMNS = [
    "timestamp",
    "datetime",
    "target_direction",
]

feature_columns = [col for col in df.columns if col not in DROP_COLUMNS]

print("Number of features before cleaning:", len(feature_columns))

X_df = df[feature_columns].copy()
y_series = df["target_direction"].copy()


# =========================================================
# CHECK FOR INF / NAN
# =========================================================
print("\nChecking invalid values...")

# Count inf values by column
inf_counts = np.isinf(X_df.select_dtypes(include=[np.number])).sum()

bad_inf_cols = inf_counts[inf_counts > 0]
if len(bad_inf_cols) > 0:
    print("\nColumns containing INF values:")
    print(bad_inf_cols.sort_values(ascending=False))
else:
    print("No INF values found before replacement.")

# Replace inf / -inf with NaN
X_df = X_df.replace([np.inf, -np.inf], np.nan)

# Count NaN values by column after replacement
nan_counts = X_df.isna().sum()
bad_nan_cols = nan_counts[nan_counts > 0]

if len(bad_nan_cols) > 0:
    print("\nColumns containing NaN values:")
    print(bad_nan_cols.sort_values(ascending=False))
else:
    print("No NaN values found.")

# Keep only rows where all selected features are valid
valid_mask = X_df.notna().all(axis=1)

rows_before = len(X_df)
rows_after = valid_mask.sum()
rows_removed = rows_before - rows_after

print(f"\nRows before cleaning: {rows_before}")
print(f"Rows after cleaning : {rows_after}")
print(f"Rows removed        : {rows_removed}")

X_df = X_df.loc[valid_mask].reset_index(drop=True)
y_series = y_series.loc[valid_mask].reset_index(drop=True)

# Optional: save cleaned dataframe for later reuse
clean_df = pd.concat([X_df, y_series], axis=1)
clean_df.to_csv(OUTPUT_CLEAN_CSV, index=False)
print(f"Cleaned CSV saved to: {OUTPUT_CLEAN_CSV}")


# =========================================================
# CONVERT TO NUMPY
# =========================================================
X_raw = X_df.values.astype(np.float64)
y_raw = y_series.values.astype(np.int64)

print("\nFinal tabular shape after cleaning:")
print("X_raw:", X_raw.shape)
print("y_raw:", y_raw.shape)


# =========================================================
# SCALE FEATURES
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

joblib.dump(scaler, OUTPUT_SCALER)
print(f"Scaler saved to: {OUTPUT_SCALER}")


# =========================================================
# BUILD SEQUENCES
# =========================================================
X = []
y = []

for i in range(SEQUENCE_LENGTH, len(X_scaled)):
    X.append(X_scaled[i - SEQUENCE_LENGTH:i])
    y.append(y_raw[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("\nSequence shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)


# =========================================================
# SAVE
# =========================================================
np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)

print("\nSaved:")
print(OUTPUT_X)
print(OUTPUT_Y)


# =========================================================
# TARGET DISTRIBUTION
# =========================================================
unique, counts = np.unique(y, return_counts=True)
distribution = dict(zip(unique, counts))

print("\nTarget distribution after sequence build:")
print(distribution)