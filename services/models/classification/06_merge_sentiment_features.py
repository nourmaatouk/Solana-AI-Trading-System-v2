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

# Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
CLASSIFICATION_DIR = SCRIPT_DIR

TECH_CSV_PATH = CLASSIFICATION_DIR / "data" / "processed" / "sol_usdt_15m_features_h4.csv"
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
        elif c_lower in EXACT_LEAKAGE:
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


def main() -> int:
    # Do not execute if files do not exist
    if not TECH_CSV_PATH.exists():
        print(f"ERROR: Tech CSV missing at {TECH_CSV_PATH}")
        return 1

    process_features(TECH_CSV_PATH, OUTPUT_CSV_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
