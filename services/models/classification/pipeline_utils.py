"""
Shared pipeline utilities used by batch merge scripts.

Centralises helpers that were previously duplicated across
06_merge_sentiment_features.py and 06b_merge_orderbook_features.py.
"""
from __future__ import annotations

import pandas as pd


def find_ts_col(df: pd.DataFrame) -> str:
    """Return the name of the timestamp column in *df*.

    Checks canonical names first (timestamp, datetime, …), then falls back to
    any column whose name ends with a time-related suffix.
    """
    candidates = ["timestamp", "datetime", "date", "time", "ts"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if c.lower().endswith(("_time", "_timestamp", "_date", "_ts")):
            return c
    raise ValueError(f"No timestamp column found in columns: {df.columns.tolist()}")
