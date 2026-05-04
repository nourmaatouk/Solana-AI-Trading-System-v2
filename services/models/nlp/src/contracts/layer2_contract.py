"""Shared Layer 2 contracts for NLP and realtime feature exchange."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

ARTICLE_DEFAULTS: Dict[str, Any] = {
    "timestamp": pd.NaT,
    "source": "unknown",
    "feed_source": "",
    "title": "",
    "url": "",
    "cleaned_text": "",
    "relevance_score": 0.0,
    "event_type": "general",
    "event_weight": 0.5,
    "event_signal": 0.0,
    "high_impact_event": 0,
    "negative_prob": 0.0,
    "neutral_prob": 0.0,
    "positive_prob": 0.0,
    "sentiment_score": 0.0,
    "confidence": 0.0,
    "sentiment_label": "neutral",
}

BUCKET_DEFAULTS: Dict[str, Any] = {
    "timestamp": pd.NaT,
    "source": "unknown",
    "event_type": "general",
    "sentiment_score": 0.0,
    "sentiment_label": "neutral",
    "confidence": 0.0,
    "relevance_score": 0.0,
    "event_signal": 0.0,
    "sentiment_mean": 0.0,
    "sentiment_std": 0.0,
    "news_volume": 0.0,
    "positive_ratio": 0.0,
    "negative_ratio": 0.0,
    "neutral_ratio": 0.0,
    "strong_positive_ratio": 0.0,
    "strong_negative_ratio": 0.0,
    "max_confidence": 0.0,
    "event_count": 0.0,
    "negative_event_count": 0.0,
    "positive_event_count": 0.0,
    "high_impact_event_count": 0.0,
    "high_impact_event": 0,
    "relevance_weighted_sentiment": 0.0,
    "event_signal_mean": 0.0,
    "event_signal_sum": 0.0,
    "gdelt_volume": 0.0,
    "rss_volume": 0.0,
    "gdelt_sentiment": 0.0,
    "rss_sentiment": 0.0,
    "sentiment_momentum": 0.0,
    "news_volume_change": 0.0,
    "sentiment_spike": 0.0,
}


def _coerce_timestamp_utc(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


def _apply_defaults(frame: pd.DataFrame, defaults: Dict[str, Any]) -> pd.DataFrame:
    normalized = frame.copy()
    for column, default_value in defaults.items():
        if column not in normalized.columns:
            normalized[column] = default_value
        elif default_value is not pd.NaT:
            normalized[column] = normalized[column].fillna(default_value)
    return normalized


def normalize_layer2_article_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single article-level NLP record to the shared contract."""
    normalized = dict(ARTICLE_DEFAULTS)
    normalized.update({key: value for key, value in record.items() if value is not None})
    if normalized.get("high_impact_event") in (None, 0):
        event_type = str(normalized.get("event_type", "general"))
        event_weight = float(normalized.get("event_weight", 0.5) or 0.5)
        normalized["high_impact_event"] = int(event_weight >= 2.0 or event_type in {"hack/security", "outage/network"})
    if not normalized.get("event_signal") and normalized.get("sentiment_score") is not None:
        normalized["event_signal"] = float(normalized.get("sentiment_score", 0.0)) * float(normalized.get("event_weight", 0.5) or 0.5)
    return normalized


def normalize_layer2_article_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an article-level NLP dataframe to the shared contract."""
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = _apply_defaults(df, ARTICLE_DEFAULTS)
    normalized = _coerce_timestamp_utc(normalized)
    derived_event_signal = normalized["sentiment_score"].astype(float) * normalized["event_weight"].astype(float)
    if "event_signal" in df.columns:
        normalized["event_signal"] = normalized["event_signal"].fillna(derived_event_signal)
    else:
        normalized["event_signal"] = derived_event_signal
    derived_high_impact = (
        (normalized["event_weight"] >= 2.0)
        | normalized["event_type"].isin(["hack/security", "outage/network"])
    ).astype(int)
    normalized["high_impact_event"] = np.maximum(normalized["high_impact_event"].astype(int), derived_high_impact)
    ordered_columns = list(ARTICLE_DEFAULTS.keys()) + [c for c in normalized.columns if c not in ARTICLE_DEFAULTS]
    return normalized[ordered_columns]


def normalize_layer2_bucket_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a 15m aggregated sentiment dataframe to the shared contract."""
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    if "timestamp" not in normalized.columns and "bucket" in normalized.columns:
        normalized = normalized.rename(columns={"bucket": "timestamp"})

    if "max_confidence" not in normalized.columns and "confidence" in normalized.columns:
        normalized["max_confidence"] = normalized["confidence"]
    if "sentiment_mean" not in normalized.columns and "sentiment_score" in normalized.columns:
        normalized["sentiment_mean"] = normalized["sentiment_score"]
    if "event_signal_mean" not in normalized.columns and "event_signal" in normalized.columns:
        normalized["event_signal_mean"] = normalized["event_signal"]
    if "event_signal_sum" not in normalized.columns and "event_signal" in normalized.columns:
        normalized["event_signal_sum"] = normalized["event_signal"]

    normalized = _apply_defaults(normalized, BUCKET_DEFAULTS)
    normalized = _coerce_timestamp_utc(normalized)

    if "high_impact_event" not in normalized.columns:
        normalized["high_impact_event"] = (normalized["high_impact_event_count"] > 0).astype(int)
    else:
        normalized["high_impact_event"] = normalized["high_impact_event"].fillna(
            (normalized["high_impact_event_count"] > 0).astype(int)
        )

    ordered_columns = list(BUCKET_DEFAULTS.keys()) + [c for c in normalized.columns if c not in BUCKET_DEFAULTS]
    return normalized[ordered_columns]
