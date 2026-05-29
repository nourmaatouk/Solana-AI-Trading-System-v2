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


from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Recursively convert non-JSON-serialisable types for json.dumps()."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def resample_orderbook_15m(df_ob: pd.DataFrame) -> pd.DataFrame:
    """Resample sub-minute order-book snapshots into 15-minute mean buckets."""
    ts_col = find_ts_col(df_ob)
    df = df_ob.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df.dropna(subset=[ts_col], inplace=True)
    df["bucket"] = df[ts_col].dt.floor("15min")
    numeric_cols = [c for c in df.columns if c not in [ts_col, "bucket"]]
    agg_dict = {c: "mean" for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])}
    df_resampled = df.groupby("bucket").agg(agg_dict).reset_index()
    df_resampled.rename(columns={"bucket": "timestamp"}, inplace=True)
    return df_resampled
