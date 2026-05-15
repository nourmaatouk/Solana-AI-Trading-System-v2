"""
06_merge_sentiment_features.py

Task:
- Load the technical features CSV from classification/data/processed/
- Convert timestamp column to UTC datetime
- Remove leakage columns if present:
  future_return, target, target_up, lead, future, t+1
- Save output as classification/data/processed/sol_usdt_15m_features_h4_with_sentiment.csv
  (Name kept identical so downstream '07_prepare_no_leak_data.py' doesn't break)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from config import LEAKAGE_SUBSTRINGS, PROCESSED_DIR, TECH_FEATURES_CSV, SENTIMENT_CSV
from pipeline_utils import find_ts_col

TECH_CSV_PATH   = PROCESSED_DIR / TECH_FEATURES_CSV
OUTPUT_CSV_PATH = PROCESSED_DIR / SENTIMENT_CSV


def process_features(tech_path: Path, output_path: Path) -> pd.DataFrame:
    print(f"Loading technical features from {tech_path}...")
    df_tech = pd.read_csv(tech_path)

    ts_tech = find_ts_col(df_tech)

    # standardize to 'timestamp'
    if ts_tech != "timestamp":
        df_tech.rename(columns={ts_tech: "timestamp"}, inplace=True)

    print("Converting timestamps to UTC...")
    df_tech["timestamp"] = pd.to_datetime(df_tech["timestamp"], errors="coerce", utc=True)

    # drop rows with unparseable timestamps just in case
    df_tech.dropna(subset=["timestamp"], inplace=True)

    print("Removing leakage columns...")
    cols_to_drop = []
    for c in df_tech.columns:
        c_lower = str(c).lower()
        if c_lower == "target_direction":
            continue
        if any(sub in c_lower for sub in LEAKAGE_SUBSTRINGS):
            cols_to_drop.append(c)

    if cols_to_drop:
        print(f"Dropping leakage columns: {cols_to_drop}")
        df_tech.drop(columns=cols_to_drop, inplace=True)

    # Create parent directories
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving processed technical data to {output_path}...")
    df_tech.to_csv(output_path, index=False)
    print("Done!")

    return df_tech


def merge_features(tech_path: Path, sent_path: Path, output_path: Path) -> pd.DataFrame:
    """Backward-compatible merge helper used by the older tests.

    The current pipeline stores technical features and sentiment separately,
    then merges them on timestamp and drops leakage columns.
    """
    df_tech = pd.read_csv(tech_path)
    df_sent = pd.read_csv(sent_path)

    tech_ts = find_ts_col(df_tech)
    sent_ts = find_ts_col(df_sent)

    if tech_ts != "timestamp":
        df_tech.rename(columns={tech_ts: "timestamp"}, inplace=True)
    if sent_ts != "timestamp":
        df_sent.rename(columns={sent_ts: "timestamp"}, inplace=True)

    df_tech["timestamp"] = pd.to_datetime(df_tech["timestamp"], errors="coerce", utc=True).astype("datetime64[ns, UTC]")
    df_sent["timestamp"] = pd.to_datetime(df_sent["timestamp"], errors="coerce", utc=True).astype("datetime64[ns, UTC]")
    df_tech.dropna(subset=["timestamp"], inplace=True)
    df_sent.dropna(subset=["timestamp"], inplace=True)

    def _drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower == "target_direction":
                continue
            if any(sub in col_lower for sub in LEAKAGE_SUBSTRINGS):
                drop_cols.append(col)
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df

    df_tech = _drop_leakage(df_tech)
    df_sent = _drop_leakage(df_sent)

    merged = pd.merge(df_tech, df_sent, on="timestamp", how="left")

    sentiment_cols = [c for c in merged.columns if c.lower().startswith("sentiment")]
    if sentiment_cols:
        merged[sentiment_cols] = merged[sentiment_cols].fillna(0.0)

    merged.fillna(0.0, inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def main() -> int:
    # Do not execute if files do not exist
    if not TECH_CSV_PATH.exists():
        print(f"ERROR: Tech CSV missing at {TECH_CSV_PATH}")
        return 1

    process_features(TECH_CSV_PATH, OUTPUT_CSV_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
