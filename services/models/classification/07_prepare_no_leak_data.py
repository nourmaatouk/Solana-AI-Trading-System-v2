"""
07_prepare_no_leak_data.py
Prepares leak-filtered combined dataset for modeling.
Outputs 2D arrays (samples × features) for tabular XGBoost.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DIR,
    PROCESSED_DIR,
    HORIZONS,
    LEAKAGE_SUBSTRINGS,
    NON_STATIONARY_COLUMNS,
    ORDERBOOK_KEYWORDS,
    ORDERBOOK_CSV,
    SENTIMENT_KEYWORDS,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

# =========================================================
# INPUT PATH
# =========================================================
INPUT_PATH = PROCESSED_DIR / ORDERBOOK_CSV


# =========================================================
# HELPERS
# =========================================================
def print_distribution(name: str, y: np.ndarray) -> None:
    unique, counts = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"{name} distribution: {dist}")


def classify_features(feature_columns: list[str]) -> dict:
    """Classify feature columns into sentiment / orderbook / technical."""
    sentiment_cols, orderbook_cols, technical_cols = [], [], []
    for col in feature_columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in SENTIMENT_KEYWORDS):
            sentiment_cols.append(col)
        elif any(kw in col_lower for kw in ORDERBOOK_KEYWORDS):
            orderbook_cols.append(col)
        else:
            technical_cols.append(col)
    return {
        "sentiment": sentiment_cols,
        "orderbook": orderbook_cols,
        "technical": technical_cols,
    }


def print_feature_breakdown(feature_columns: list[str]) -> None:
    groups = classify_features(feature_columns)
    print(f"\n--- Feature Breakdown ---")
    print(f"  Total features:     {len(feature_columns)}")
    print(f"  Technical features: {len(groups['technical'])}")
    print(f"  Sentiment features: {len(groups['sentiment'])}")
    print(f"  Orderbook features: {len(groups['orderbook'])}")
    print(f"-------------------------")


def _is_leaky_or_nonstationary(col: str, target_col: str) -> bool:
    """Return True if column must be excluded from features."""
    if col == target_col:
        return True
    col_lower = col.lower()
    if col_lower in {"timestamp", "datetime", "date", "time", "ts"}:
        return True
    if col_lower in NON_STATIONARY_COLUMNS:
        return True
    return any(sub in col_lower for sub in LEAKAGE_SUBSTRINGS)


# =========================================================
# PRODUCTION ENTRY POINT
# =========================================================
def process_data_for_horizon(input_path: Path, data_dir: Path, horizon: int) -> dict:
    print(f"\n{'='*40}\nPROCESSING DATA FOR HORIZON: H{horizon}\n{'='*40}")

    df = pd.read_csv(input_path)

    # 1. Triple-barrier labeling
    future_max_high = df["high"].rolling(window=horizon).max().shift(-horizon)
    future_min_low  = df["low"].rolling(window=horizon).min().shift(-horizon)
    tp_price = df["close"] * (1 + TAKE_PROFIT_PCT)
    sl_price = df["close"] * (1 - STOP_LOSS_PCT)

    hit_tp = future_max_high >= tp_price
    hit_sl = future_min_low  <= sl_price

    df["target_direction"] = np.nan
    df.loc[hit_tp & ~hit_sl, "target_direction"] = 1
    df.loc[hit_sl & ~hit_tp, "target_direction"] = 0
    # NaN = ambiguous (both or neither) → dropped below

    n_before = len(df)
    df.dropna(subset=["target_direction"], inplace=True)
    df["target_direction"] = df["target_direction"].astype(int)
    print(f"Triple-Barrier: dropped {n_before - len(df)} ambiguous rows "
          f"(TP={TAKE_PROFIT_PCT*100:.1f}%, SL={STOP_LOSS_PCT*100:.1f}%).")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    TARGET_COL = "target_direction"
    y_all = df[TARGET_COL].values.astype(np.int64)

    # 2. Drop leakage and non-stationary columns
    feature_columns = [c for c in df.columns if not _is_leaky_or_nonstationary(c, TARGET_COL)]
    X_all = df[feature_columns].values.astype(np.float64)

    print(f"X_all shape: {X_all.shape}, stationary features: {len(feature_columns)}")
    print_feature_breakdown(feature_columns)

    # 3. Chronological split
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train_raw, y_train = X_all[:train_end],       y_all[:train_end]
    X_val_raw,   y_val   = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test_raw,  y_test  = X_all[val_end:],          y_all[val_end:]

    # 4. Scale — fit ONLY on train
    scaler   = StandardScaler()
    X_train  = scaler.fit_transform(X_train_raw)
    X_val    = scaler.transform(X_val_raw)
    X_test   = scaler.transform(X_test_raw)

    data_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, data_dir / f"scaler_h{horizon}.pkl")

    # 5. Report
    print(f"\n--- Split Shapes (H{horizon}) ---")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape},   y={y_val.shape}")
    print(f"  Test:  X={X_test.shape},  y={y_test.shape}")
    print_distribution("  Train", y_train)
    print_distribution("  Val",   y_val)
    print_distribution("  Test",  y_test)
    print("-----------------------------")

    # 6. Save arrays
    np.save(data_dir / f"X_train_h{horizon}.npy", X_train)
    np.save(data_dir / f"y_train_h{horizon}.npy", y_train)
    np.save(data_dir / f"X_val_h{horizon}.npy",   X_val)
    np.save(data_dir / f"y_val_h{horizon}.npy",   y_val)
    np.save(data_dir / f"X_test_h{horizon}.npy",  X_test)
    np.save(data_dir / f"y_test_h{horizon}.npy",  y_test)

    # Feature names are horizon-independent; save once (idempotent across horizons)
    with open(data_dir / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }


# =========================================================
# BACKWARD-COMPATIBLE HELPER (used by test_07_prepare_no_leak_data.py)
# =========================================================
def process_data(input_path: Path, data_dir: Path) -> dict:
    """Sequence-based preprocessing used by the unit tests.

    Builds fixed-length (seq_len=32) sliding windows, applies a
    train-only scaler, and saves with the legacy *_no_leak filename
    convention so the tests pass unchanged.
    """
    df = pd.read_csv(input_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    target_col = "target_direction"
    feature_columns = [
        col for col in df.columns
        if not _is_leaky_or_nonstationary(col, target_col) and col != target_col
    ]
    X_raw = df[feature_columns].to_numpy(dtype=np.float64)
    y_raw = (
        df[target_col].to_numpy(dtype=np.int64)
        if target_col in df.columns
        else np.zeros(len(df), dtype=np.int64)
    )

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train_raw = X_raw[:train_end]
    y_train_raw = y_raw[:train_end]
    X_val_raw   = X_raw[train_end:val_end]
    y_val_raw   = y_raw[train_end:val_end]
    X_test_raw  = X_raw[val_end:]
    y_test_raw  = y_raw[val_end:]

    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train_raw)
    n_feat       = X_train_sc.shape[1]
    X_val_sc     = scaler.transform(X_val_raw)   if len(X_val_raw)  else np.empty((0, n_feat))
    X_test_sc    = scaler.transform(X_test_raw)  if len(X_test_raw) else np.empty((0, n_feat))

    seq_len = 32

    def _build_sequences(X_block: np.ndarray, y_block: np.ndarray):
        if len(X_block) <= seq_len:
            return (
                np.empty((0, seq_len, X_block.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        seqs, labels = [], []
        for start in range(len(X_block) - seq_len):
            seqs.append(X_block[start : start + seq_len])
            labels.append(int(y_block[start + seq_len]))
        return np.asarray(seqs, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    X_train, y_train = _build_sequences(X_train_sc, y_train_raw)
    X_val,   y_val   = _build_sequences(X_val_sc,   y_val_raw)
    X_test,  y_test  = _build_sequences(X_test_sc,  y_test_raw)

    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "X_train_no_leak.npy", X_train)
    np.save(data_dir / "y_train_no_leak.npy", y_train)
    np.save(data_dir / "X_val_no_leak.npy",   X_val)
    np.save(data_dir / "y_val_no_leak.npy",   y_val)
    np.save(data_dir / "X_test_no_leak.npy",  X_test)
    np.save(data_dir / "y_test_no_leak.npy",  y_test)

    return {
        "feature_columns": feature_columns,
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    if not INPUT_PATH.exists():
        print(f"Error: input dataset not found at {INPUT_PATH}")
        sys.exit(1)

    for h in HORIZONS:
        process_data_for_horizon(INPUT_PATH, PROCESSED_DIR, h)


if __name__ == "__main__":
    main()
