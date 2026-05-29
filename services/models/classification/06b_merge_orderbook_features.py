"""
06b_merge_orderbook_features.py

Task:
- Load sol_usdt_15m_features_h4_with_sentiment.csv
- Load orderbook_features.csv from data/realtime/
- Convert timestamps to UTC datetime
- Resample orderbook data to 15-minute intervals (mean aggregation)
- Left join order book features onto the existing market+sentiment dataset
- Fill missing order book features with 0
- Remove leakage columns
- Save: data/processed/sol_usdt_15m_features_h4_with_sentiment_orderbook.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import LEAKAGE_SUBSTRINGS, ORDERBOOK_CSV, PROCESSED_DIR, REALTIME_DIR, SENTIMENT_CSV
from pipeline_utils import find_ts_col, resample_orderbook_15m

# =========================================================
# PATHS
# =========================================================
MARKET_SENT_PATH = PROCESSED_DIR / SENTIMENT_CSV
ORDERBOOK_PATH   = REALTIME_DIR / "orderbook_features.csv"
OUTPUT_PATH      = PROCESSED_DIR / ORDERBOOK_CSV


def merge_orderbook(
    market_sent_path: Path,
    orderbook_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Merge orderbook features onto market+sentiment dataset."""
    print(f"Loading market+sentiment features from {market_sent_path}...")
    df_market = pd.read_csv(market_sent_path)

    print(f"Loading orderbook features from {orderbook_path}...")
    df_ob_raw = pd.read_csv(orderbook_path)

    # --- Standardise timestamps to 'timestamp' column, UTC ---
    ts_market = find_ts_col(df_market)
    if ts_market != "timestamp":
        df_market.rename(columns={ts_market: "timestamp"}, inplace=True)

    df_market["timestamp"] = pd.to_datetime(
        df_market["timestamp"], errors="coerce", utc=True,
    )
    df_market.dropna(subset=["timestamp"], inplace=True)

    # --- Resample orderbook to 15-min buckets ---
    print("Resampling orderbook data to 15-minute intervals...")
    df_ob = resample_orderbook_15m(df_ob_raw)

    ob_cols = [c for c in df_ob.columns if c != "timestamp"]
    n_before = len(df_market)

    # --- Left join ---
    print("Merging orderbook features onto market+sentiment data...")
    df_merged = pd.merge(df_market, df_ob, on="timestamp", how="left")

    assert len(df_merged) == n_before, (
        f"Row count changed after merge: {n_before} -> {len(df_merged)}"
    )

    # --- Fill missing orderbook values with 0 ---
    print("Filling missing orderbook values with 0...")
    for c in ob_cols:
        if c in df_merged.columns:
            df_merged[c] = df_merged[c].fillna(0)

    # --- Remove leakage columns (keep target_direction) ---
    print("Removing leakage columns...")
    cols_to_drop = []
    for c in df_merged.columns:
        c_lower = str(c).lower()
        if c_lower == "target_direction":
            continue
        if any(sub in c_lower for sub in LEAKAGE_SUBSTRINGS):
            cols_to_drop.append(c)

    if cols_to_drop:
        print(f"  Dropping: {cols_to_drop}")
        df_merged.drop(columns=cols_to_drop, inplace=True)

    # --- Replace inf with NaN, then fill with 0 ---
    df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_merged.fillna(0, inplace=True)

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    print(f"Saved merged dataset to {output_path}")
    print(f"  Shape: {df_merged.shape}")
    print(f"  Orderbook columns added: {ob_cols}")

    return df_merged


# =========================================================
# MAIN
# =========================================================
def main() -> int:
    if not MARKET_SENT_PATH.exists():
        print(f"ERROR: Market+sentiment CSV missing at {MARKET_SENT_PATH}")
        return 1
    if not ORDERBOOK_PATH.exists():
        print(f"ERROR: Orderbook CSV missing at {ORDERBOOK_PATH}")
        return 1

    merge_orderbook(MARKET_SENT_PATH, ORDERBOOK_PATH, OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
