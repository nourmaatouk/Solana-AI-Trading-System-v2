from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
NLP_DIR = PROJECT_ROOT / "nlp"
TECH_PATH = BASE_DIR / "data" / "realtime" / "latest_sol_15m_features.csv"
SENTIMENT_PATH = BASE_DIR / "data" / "realtime" / "latest_sentiment_15m.csv"
ORDERBOOK_PATH = BASE_DIR / "data" / "realtime" / "orderbook_features.csv"
OUTPUT_PATH = BASE_DIR / "data" / "realtime" / "latest_features_with_sentiment.csv"

sys.path.append(str(NLP_DIR))

from src.contracts.layer2_contract import normalize_layer2_bucket_frame


def normalize_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def resample_orderbook_15m(df_orderbook: pd.DataFrame) -> pd.DataFrame:
    df_ob = df_orderbook.copy()
    if "timestamp" not in df_ob.columns:
        raise ValueError("Orderbook data must contain a timestamp column.")

    df_ob["timestamp"] = normalize_timestamp(df_ob["timestamp"])
    df_ob.dropna(subset=["timestamp"], inplace=True)
    if df_ob.empty:
        return pd.DataFrame(columns=["timestamp"])

    df_ob["timestamp"] = df_ob["timestamp"].dt.floor("15min")
    numeric_cols = [
        col for col in df_ob.columns
        if col != "timestamp" and pd.api.types.is_numeric_dtype(df_ob[col])
    ]

    return df_ob.groupby("timestamp", as_index=False)[numeric_cols].mean()


def merge_realtime_data(
    df_tech: pd.DataFrame,
    df_sent: pd.DataFrame,
    df_orderbook: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merges technical, sentiment, and optional orderbook features for real-time inference.
    Uses 'timestamp' as the join key after ensuring UTC normalization.
    """
    df_tech = df_tech.copy()
    df_sent = normalize_layer2_bucket_frame(df_sent.copy())
    
    # 1. Standardize timestamps to UTC Datetime
    # Tech data usually has Unix ms and maybe 'datetime' col
    if "timestamp" in df_tech.columns:
        df_tech["timestamp"] = normalize_timestamp(df_tech["timestamp"])
    elif "datetime" in df_tech.columns:
        df_tech["timestamp"] = normalize_timestamp(df_tech["datetime"])
        
    if "timestamp" in df_sent.columns:
        df_sent["timestamp"] = normalize_timestamp(df_sent["timestamp"])
        
    print(f"Tech samples: {len(df_tech)}")
    print(f"Sentiment samples: {len(df_sent)}")

    # 2. Left join sentiment onto technical features
    # Keeping all market rows as requested
    df_merged = pd.merge(df_tech, df_sent, on="timestamp", how="left")
    
    # 3. Fill missing sentiment columns with 0
    # Identify sentiment columns from df_sent (excluding join key)
    sent_cols = [c for c in df_sent.columns if c != "timestamp"]
    for col in sent_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0.0)

    if df_orderbook is not None and not df_orderbook.empty:
        df_ob = resample_orderbook_15m(df_orderbook)
        ob_cols = [c for c in df_ob.columns if c != "timestamp"]
        print(f"Orderbook buckets: {len(df_ob)}")

        if ob_cols:
            df_merged = pd.merge(df_merged, df_ob, on="timestamp", how="left")
            matched_mask = df_merged[ob_cols].notna().any(axis=1)
            df_merged["orderbook_data_available"] = matched_mask.astype(int)
            df_merged[ob_cols] = df_merged[ob_cols].fillna(0.0)
    else:
        print("Orderbook samples: 0")
            
    print(f"Merged shape: {df_merged.shape}")
    
    return df_merged


def main():
    print("=" * 60)
    print("Real-time Feature Merger Starting")
    print("=" * 60)

    if not TECH_PATH.exists():
        print(f"Error: Technical features not found at {TECH_PATH}")
        sys.exit(1)
    if not SENTIMENT_PATH.exists():
        print(f"Warning: Sentiment features not found at {SENTIMENT_PATH}. Using empty fallback.")
        df_sent = pd.DataFrame(columns=["timestamp"])
    else:
        df_sent = pd.read_csv(SENTIMENT_PATH)
    if not ORDERBOOK_PATH.exists():
        print(f"Warning: Orderbook features not found at {ORDERBOOK_PATH}.")
        df_orderbook = None
    else:
        df_orderbook = pd.read_csv(ORDERBOOK_PATH)

    df_tech = pd.read_csv(TECH_PATH)
    
    # Merge
    df_final = merge_realtime_data(df_tech, df_sent, df_orderbook)
    
    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Merged features saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
