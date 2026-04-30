"""
06_merge_sentiment_features.py

Task:
- Load the technical features CSV from classification/data/processed/
- Load sentiment_features_15m.csv from services/models/nlp/data/processed/
- Convert both timestamp columns to UTC datetime
- Left join sentiment onto technical features using timestamp
- Fill missing sentiment columns with 0
- Remove leakage columns if present:
  future_return, target, target_up, lead, future, t+1
- Save output as classification/data/processed/sol_usdt_15m_features_h4_with_sentiment.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
CLASSIFICATION_DIR = SCRIPT_DIR
NLP_DIR = MODELS_DIR / "nlp"

TECH_CSV_PATH = CLASSIFICATION_DIR / "data" / "processed" / "sol_usdt_15m_features_h4.csv"
SENTIMENT_CSV_PATH = NLP_DIR / "data" / "processed" / "sentiment_features_15m.csv"
OUTPUT_CSV_PATH = CLASSIFICATION_DIR / "data" / "processed" / "sol_usdt_15m_features_h4_with_sentiment.csv"

LEAKAGE_SUBSTRINGS = [
    "future",
    "lead",
    "t+1",
]

EXACT_LEAKAGE = [
    "target",
    "target_up",
]


def find_ts_col(df: pd.DataFrame) -> str:
    """Find the timestamp column."""
    candidates = ["timestamp", "datetime", "date", "time", "ts"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if c.lower().endswith(("_time", "_timestamp", "_date", "_ts")):
            return c
    raise ValueError(f"No timestamp column found in columns: {df.columns.tolist()}")


def merge_features(tech_path: Path, sent_path: Path, output_path: Path) -> pd.DataFrame:
    print(f"Loading technical features from {tech_path}...")
    df_tech = pd.read_csv(tech_path)

    print(f"Loading sentiment features from {sent_path}...")
    df_sent = pd.read_csv(sent_path)

    ts_tech = find_ts_col(df_tech)
    ts_sent = find_ts_col(df_sent)

    # standardize to 'timestamp'
    if ts_tech != "timestamp":
        df_tech.rename(columns={ts_tech: "timestamp"}, inplace=True)
    if ts_sent != "timestamp":
        df_sent.rename(columns={ts_sent: "timestamp"}, inplace=True)

    print("Converting timestamps to UTC...")
    df_tech["timestamp"] = pd.to_datetime(df_tech["timestamp"], errors="coerce", utc=True)
    df_sent["timestamp"] = pd.to_datetime(df_sent["timestamp"], errors="coerce", utc=True)

    # drop rows with unparseable timestamps just in case
    df_tech.dropna(subset=["timestamp"], inplace=True)
    df_sent.dropna(subset=["timestamp"], inplace=True)

    # record original sentiment columns except timestamp
    sent_cols = [c for c in df_sent.columns if c != "timestamp"]

    print("Merging sentiment onto technical features...")
    df_merged = pd.merge(df_tech, df_sent, on="timestamp", how="left")

    print("Filling missing sentiment values with 0...")
    for c in sent_cols:
        if c in df_merged.columns:
            df_merged[c] = df_merged[c].fillna(0)

    print("Removing leakage columns...")
    cols_to_drop = []
    for c in df_merged.columns:
        c_lower = str(c).lower()
        if c_lower == "target_direction":
            continue
        if any(sub in c_lower for sub in LEAKAGE_SUBSTRINGS):
            cols_to_drop.append(c)
        elif c_lower in EXACT_LEAKAGE:
            cols_to_drop.append(c)

    if cols_to_drop:
        print(f"Dropping leakage columns: {cols_to_drop}")
        df_merged.drop(columns=cols_to_drop, inplace=True)

    # Create parent directories
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged data to {output_path}...")
    df_merged.to_csv(output_path, index=False)
    print("Done!")

    return df_merged


def main() -> int:
    # Do not execute if files do not exist
    if not TECH_CSV_PATH.exists():
        print(f"ERROR: Tech CSV missing at {TECH_CSV_PATH}")
        return 1
    if not SENTIMENT_CSV_PATH.exists():
        print(f"ERROR: Sentiment CSV missing at {SENTIMENT_CSV_PATH}")
        return 1

    merge_features(TECH_CSV_PATH, SENTIMENT_CSV_PATH, OUTPUT_CSV_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
