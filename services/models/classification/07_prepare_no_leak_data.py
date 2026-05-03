"""
07_prepare_no_leak_data.py
Prepares leaked-filtered combined dataset for modeling.
Outputs 2D arrays (samples x features) for tabular tree models like XGBoost.
"""
import json
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

# Dataset paths
INPUT_PATH = DATA_DIR / "sol_usdt_15m_features_h4_with_sentiment_orderbook.csv"

# Known feature group identifiers
SENTIMENT_KEYWORDS = [
    "sentiment", "news_volume", "positive_ratio", "negative_ratio",
    "neutral_ratio", "strong_positive", "strong_negative", "max_confidence",
    "event_count", "negative_event", "positive_event", "high_impact",
    "relevance_weighted", "event_signal", "gdelt", "rss",
    "sentiment_momentum", "news_volume_change", "sentiment_spike",
]
ORDERBOOK_KEYWORDS = [
    "imbalance", "spread_mean", "mid_price", "bid_volume",
    "ask_volume", "pressure_indicator",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios must sum to 1."

# =========================================================
# HELPERS
# =========================================================
def print_distribution(name: str, y: np.ndarray) -> None:
    unique, counts = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"{name} distribution: {dist}")

def classify_features(feature_columns: list[str]) -> dict:
    """Classify feature columns into sentiment / orderbook / technical."""
    sentiment_cols = []
    orderbook_cols = []
    technical_cols = []

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
    """Print a breakdown of features by category."""
    groups = classify_features(feature_columns)
    print(f"\n--- Feature Breakdown ---")
    print(f"  Total features:     {len(feature_columns)}")
    print(f"  Technical features: {len(groups['technical'])}")
    print(f"  Sentiment features: {len(groups['sentiment'])}")
    print(f"  Orderbook features: {len(groups['orderbook'])}")
    print(f"-------------------------")

# =========================================================
# LOAD AND PROCESS DATA
# =========================================================
def process_data_for_horizon(input_path: Path, data_dir: Path, horizon: int) -> dict:
    print(f"\n" + "="*40)
    print(f"PROCESSING DATA FOR HORIZON: H{horizon}")
    print("="*40)
    
    df = pd.read_csv(input_path)
    
    # 1. Create Target using Triple-Barrier Labeling
    # Instead of naive close vs close, we calculate if it hits a Take Profit or Stop Loss
    pt_pct = 0.02   # 2% Take Profit
    sl_pct = 0.015  # 1.5% Stop Loss
    
    # Vectorized look-forward for max high and min low in the upcoming horizon
    future_max_high = df["high"].rolling(window=horizon).max().shift(-horizon)
    future_min_low = df["low"].rolling(window=horizon).min().shift(-horizon)
    
    tp_price = df["close"] * (1 + pt_pct)
    sl_price = df["close"] * (1 - sl_pct)
    
    hit_tp = future_max_high >= tp_price
    hit_sl = future_min_low <= sl_price
    
    df["target_direction"] = np.nan
    
    # 1 if it hits Take Profit cleanly without hitting Stop Loss
    df.loc[hit_tp & ~hit_sl, "target_direction"] = 1
    # 0 if it hits Stop Loss cleanly without hitting Take Profit
    df.loc[hit_sl & ~hit_tp, "target_direction"] = 0
    # Ambiguous hits (both) or time expirations (neither) remain NaN
    
    n_before_drop = len(df)
    df.dropna(subset=["target_direction"], inplace=True)
    df["target_direction"] = df["target_direction"].astype(int)
    print(f"Triple-Barrier Labeling: Dropped {n_before_drop - len(df)} non-actionable rows (no trend or volatile chop).")
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    TARGET_COL = "target_direction"
    y_all = df[TARGET_COL].values.astype(np.int64)

    # 2. Prevent Leakage & Non-Stationarity
    LEAKAGE_SUBSTRINGS = ["future_return", "target", "lead", "future", "t+1"]
    
    # We must explicitly ban NON-STATIONARY values from tree-estimators
    NON_STATIONARY = ["open", "high", "low", "close", "volume", "sma_10", "sma_20", "sma_50", "ema_10", "ema_20", "ema_50"]

    cols_to_drop = set()
    for c in df.columns:
        c_lower = str(c).lower()
        if c == TARGET_COL:
            cols_to_drop.add(c)
        elif any(sub in c_lower for sub in LEAKAGE_SUBSTRINGS):
            cols_to_drop.add(c)
        elif c_lower in NON_STATIONARY:
            cols_to_drop.add(c)
        elif c_lower in ["timestamp", "datetime", "date", "time", "ts"]:
            cols_to_drop.add(c)

    feature_columns = [c for c in df.columns if c not in cols_to_drop]
    X_all = df[feature_columns].values.astype(np.float64)

    print(f"X_all shape: {X_all.shape}, Cleaned Stationary Features: {len(feature_columns)}")
    print_feature_breakdown(feature_columns)

    # 3. Split
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train_raw = X_all[:train_end]
    y_train = y_all[:train_end]

    X_val_raw = X_all[train_end:val_end]
    y_val = y_all[train_end:val_end]

    X_test_raw = X_all[val_end:]
    y_test = y_all[val_end:]

    # 4. Scaling (Only fit on train!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Save horizon-specific scaler
    data_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = data_dir / f"scaler_h{horizon}.pkl"
    joblib.dump(scaler, scaler_path)

    # 5. Output single rows directly (No Sequence Logic)
    # The arrays are now standard flat 2D arrays: (samples, features)

    # Print split shapes
    print(f"\n--- Split Shapes (H{horizon}) ---")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape},  y={y_val.shape}")
    print(f"  Test:  X={X_test.shape},  y={y_test.shape}")
    print_distribution(f"  Train", y_train)
    print_distribution(f"  Val", y_val)
    print_distribution(f"  Test", y_test)
    print(f"-----------------------------")

    # 6. Save
    np.save(data_dir / f"X_train_h{horizon}.npy", X_train)
    np.save(data_dir / f"y_train_h{horizon}.npy", y_train)
    np.save(data_dir / f"X_val_h{horizon}.npy", X_val)
    np.save(data_dir / f"y_val_h{horizon}.npy", y_val)
    np.save(data_dir / f"X_test_h{horizon}.npy", X_test)
    np.save(data_dir / f"y_test_h{horizon}.npy", y_test)

    # Save feature names
    feature_cols_path = data_dir / "feature_columns.json"
    with open(feature_cols_path, "w") as f:
        json.dump(feature_columns, f, indent=2)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

def main():
    if not INPUT_PATH.exists():
        print(f"Error: No input dataset found at {INPUT_PATH}")
        sys.exit(1)

    for h in [4, 8, 16]:
        process_data_for_horizon(INPUT_PATH, DATA_DIR, h)

if __name__ == "__main__":
    main()