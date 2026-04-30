"""
07_prepare_no_leak_data.py
Prepares leaked-filtered combined dataset sequences for modeling.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# =========================================================
# CONFIG
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "processed"
INPUT_PATH = DATA_DIR / "sol_usdt_15m_features_h4_with_sentiment.csv"
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
    X_seq = []
    y_seq = []
    for i in range(seq_len, len(X_2d)):
        X_seq.append(X_2d[i - seq_len:i])
        y_seq.append(y_1d[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)

def print_distribution(name: str, y: np.ndarray) -> None:
    unique, counts = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"{name} distribution: {dist}")

# =========================================================
# LOAD AND PROCESS DATA
# =========================================================
def process_data(input_path: Path, data_dir: Path) -> dict:
    df = pd.read_csv(input_path)
    print("Loaded clean tabular dataset:")
    print(df.shape)

    n_before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    n_after = len(df)
    print(f"Dropped {n_before - n_after} rows due to NaN/inf")

    TARGET_COL = "target_direction"
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    y_all = df[TARGET_COL].values.astype(np.int64)

    LEAKAGE_SUBSTRINGS = ["future_return", "target", "lead", "future", "t+1"]

    cols_to_drop = set()
    for c in df.columns:
        c_lower = str(c).lower()
        if c == TARGET_COL:
            cols_to_drop.add(c)
        elif any(sub in c_lower for sub in LEAKAGE_SUBSTRINGS):
            cols_to_drop.add(c)
        elif c_lower in ["timestamp", "datetime", "date", "time", "ts"]:
            cols_to_drop.add(c)

    feature_columns = [c for c in df.columns if c not in cols_to_drop]

    sentiment_keywords = ["sentiment", "news", "gdelt", "rss", "positive_", "negative_", "neutral_", "max_conf"]
    num_sentiment = sum(1 for c in feature_columns if any(k in str(c).lower() for k in sentiment_keywords))

    X_all = df[feature_columns].values.astype(np.float64)

    print("Tabular shapes:")
    print("X_all:", X_all.shape)
    print("y_all:", y_all.shape)
    print("Number of total features:", len(feature_columns))
    print("Number of sentiment features included:", num_sentiment)

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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    data_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = data_dir / "scaler_no_leak.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler saved to: {scaler_path}")

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

    np.save(data_dir / "X_train_no_leak.npy", X_train)
    np.save(data_dir / "y_train_no_leak.npy", y_train)

    np.save(data_dir / "X_val_no_leak.npy", X_val)
    np.save(data_dir / "y_val_no_leak.npy", y_val)

    np.save(data_dir / "X_test_no_leak.npy", X_test)
    np.save(data_dir / "y_test_no_leak.npy", y_test)

    print("\nSaved no-leak arrays to:", data_dir)
    
    # Save feature columns for real-time alignment
    import json
    feature_cols_path = data_dir / "feature_columns.json"
    with open(feature_cols_path, "w") as f:
        json.dump(feature_columns, f, indent=2)
    print(f"Saved feature columns to: {feature_cols_path}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_columns": feature_columns
    }

def main():
    print(f"INPUT_PATH resolving to: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} does not exist. Run 06_merge_sentiment_features.py first.")
        sys.exit(1)
    process_data(INPUT_PATH, DATA_DIR)

if __name__ == "__main__":
    main()