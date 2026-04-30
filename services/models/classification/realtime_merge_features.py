from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
TECH_PATH = BASE_DIR / "data" / "realtime" / "latest_sol_15m_features.csv"
SENTIMENT_PATH = BASE_DIR / "data" / "realtime" / "latest_sentiment_15m.csv"
OUTPUT_PATH = BASE_DIR / "data" / "realtime" / "latest_features_with_sentiment.csv"


def merge_realtime_data(df_tech: pd.DataFrame, df_sent: pd.DataFrame) -> pd.DataFrame:
    """
    Merges technical and sentiment features for real-time inference.
    Uses 'timestamp' as the join key after ensuring UTC normalization.
    """
    df_tech = df_tech.copy()
    df_sent = df_sent.copy()
    
    # 1. Standardize timestamps to UTC Datetime
    # Tech data usually has Unix ms and maybe 'datetime' col
    if "timestamp" in df_tech.columns:
        df_tech["timestamp"] = pd.to_datetime(df_tech["timestamp"], unit="ms", utc=True)
    elif "datetime" in df_tech.columns:
        df_tech["timestamp"] = pd.to_datetime(df_tech["datetime"], utc=True)
        
    if "timestamp" in df_sent.columns:
        df_sent["timestamp"] = pd.to_datetime(df_sent["timestamp"], utc=True)
        
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

    df_tech = pd.read_csv(TECH_PATH)
    
    # Merge
    df_final = merge_realtime_data(df_tech, df_sent)
    
    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Merged features saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
