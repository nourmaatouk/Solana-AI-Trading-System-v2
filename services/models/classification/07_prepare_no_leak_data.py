from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = DATA_DIR / "sol_usdt_15m_features_h4_clean.csv"

SEQUENCE_LENGTH = 32

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios must sum to 1."


# =========================================================
# HELPERS
# =========================================================
def build_sequences_from_split(
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2D tabular split into 3D sequences.

    Input:
        X_2d shape = (n_rows, n_features)
        y_1d shape = (n_rows,)

    Output:
        X_seq shape = (n_samples, seq_len, n_features)
        y_seq shape = (n_samples,)
    """
    X_seq = []
    y_seq = []

    for i in range(seq_len, len(X_2d)):
        X_seq.append(X_2d[i - seq_len:i])
        y_seq.append(y_1d[i])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int64)

    return X_seq, y_seq


def print_distribution(name: str, y: np.ndarray) -> None:
    unique, counts = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"{name} distribution: {dist}")


# =========================================================
# LOAD CLEAN TABULAR DATA
# =========================================================
df = pd.read_csv(INPUT_PATH)

print("Loaded clean tabular dataset:")
print(df.shape)

# Important:
# this file currently contains only features + target_direction
# because in step 3 we saved the cleaned CSV from X_df + y_series.
# So timestamp/datetime are already absent, which is fine.

TARGET_COL = "target_direction"
if TARGET_COL not in df.columns:
    raise ValueError(f"Missing target column: {TARGET_COL}")

feature_columns = [c for c in df.columns if c != TARGET_COL]

X_all = df[feature_columns].values.astype(np.float64)
y_all = df[TARGET_COL].values.astype(np.int64)

print("Tabular shapes:")
print("X_all:", X_all.shape)
print("y_all:", y_all.shape)
print("Number of features:", len(feature_columns))


# =========================================================
# CHRONOLOGICAL TABULAR SPLIT
# =========================================================
n = len(df)

train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

X_train_raw = X_all[:train_end]
y_train_raw = y_all[:train_end]

X_val_raw = X_all[train_end:val_end]
y_val_raw = y_all[train_end:val_end]

X_test_raw = X_all[val_end:]
y_test_raw = y_all[val_end:]

print("\nChronological tabular split:")
print("X_train_raw:", X_train_raw.shape, "y_train_raw:", y_train_raw.shape)
print("X_val_raw  :", X_val_raw.shape, "y_val_raw  :", y_val_raw.shape)
print("X_test_raw :", X_test_raw.shape, "y_test_raw :", y_test_raw.shape)

print_distribution("Train raw", y_train_raw)
print_distribution("Val raw", y_val_raw)
print_distribution("Test raw", y_test_raw)


# =========================================================
# FIT SCALER ONLY ON TRAIN
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

scaler_path = DATA_DIR / "scaler_no_leak.pkl"
joblib.dump(scaler, scaler_path)

print(f"\nScaler saved to: {scaler_path}")


# =========================================================
# BUILD SEQUENCES SEPARATELY FOR EACH SPLIT
# =========================================================
X_train, y_train = build_sequences_from_split(X_train_scaled, y_train_raw, SEQUENCE_LENGTH)
X_val, y_val = build_sequences_from_split(X_val_scaled, y_val_raw, SEQUENCE_LENGTH)
X_test, y_test = build_sequences_from_split(X_test_scaled, y_test_raw, SEQUENCE_LENGTH)

print("\nSequence shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)

print_distribution("Train seq", y_train)
print_distribution("Val seq", y_val)
print_distribution("Test seq", y_test)


# =========================================================
# SAVE NO-LEAK ARRAYS
# =========================================================
np.save(DATA_DIR / "X_train_no_leak.npy", X_train)
np.save(DATA_DIR / "y_train_no_leak.npy", y_train)

np.save(DATA_DIR / "X_val_no_leak.npy", X_val)
np.save(DATA_DIR / "y_val_no_leak.npy", y_val)

np.save(DATA_DIR / "X_test_no_leak.npy", X_test)
np.save(DATA_DIR / "y_test_no_leak.npy", y_test)

print("\nSaved no-leak arrays:")
print(DATA_DIR / "X_train_no_leak.npy")
print(DATA_DIR / "y_train_no_leak.npy")
print(DATA_DIR / "X_val_no_leak.npy")
print(DATA_DIR / "y_val_no_leak.npy")
print(DATA_DIR / "X_test_no_leak.npy")
print(DATA_DIR / "y_test_no_leak.npy")