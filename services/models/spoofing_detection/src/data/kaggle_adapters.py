"""Normalization adapters for Kaggle spoofing detection data."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def normalize_kaggle_crypto_frame(df: pd.DataFrame, n_levels: int = 15) -> pd.DataFrame:
    """Normalize raw Kaggle LOB rows into the canonical loader schema.

    The Kaggle CSVs use 0-based depth columns such as ``bids_notional_0`` and
    ``asks_cancel_notional_14``. The existing feature loader expects
    1-based aliases like ``bids_limit_notional_1`` and ``bids_cancel_notional_1``.
    This adapter fills those aliases, coerces numerics, and standardizes the
    timestamp column so downstream feature extraction can stay simple.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()

    if "timestamp" in normalized.columns:
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True)
    elif "datetime" in normalized.columns:
        normalized["timestamp"] = pd.to_datetime(normalized["datetime"], errors="coerce", utc=True)

    if "midpoint" not in normalized.columns and "mid_price" in normalized.columns:
        normalized["midpoint"] = normalized["mid_price"]
    if "spread" not in normalized.columns and {"ask_price_1", "bid_price_1"}.issubset(normalized.columns):
        normalized["spread"] = normalized["ask_price_1"] - normalized["bid_price_1"]

    aliases = (
        ("bids_notional", "bids_limit_notional"),
        ("asks_notional", "asks_limit_notional"),
        ("bids_cancel_notional", "bids_cancel_notional"),
        ("asks_cancel_notional", "asks_cancel_notional"),
    )

    for raw_prefix, alias_prefix in aliases:
        for raw_index in range(n_levels):
            raw_col = f"{raw_prefix}_{raw_index}"
            alias_col = f"{alias_prefix}_{raw_index + 1}"
            if raw_col in normalized.columns:
                normalized[alias_col] = _to_numeric_series(normalized[raw_col])
            elif alias_col not in normalized.columns:
                normalized[alias_col] = 0.0
            else:
                normalized[alias_col] = _to_numeric_series(normalized[alias_col])

    for column in ["midpoint", "spread", "buys", "sells"]:
        if column in normalized.columns:
            normalized[column] = _to_numeric_series(normalized[column])

    return normalized