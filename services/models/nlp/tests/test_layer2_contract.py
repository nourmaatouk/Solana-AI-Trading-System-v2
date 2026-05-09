import pandas as pd

from src.contracts.layer2_contract import (
    normalize_layer2_article_frame,
    normalize_layer2_bucket_frame,
)


def test_normalize_layer2_bucket_frame_adds_missing_columns():
    df = pd.DataFrame([
        {
            "timestamp": "2026-04-30T10:00:00Z",
            "sentiment_mean": 0.4,
            "news_volume": 3,
            "confidence": 0.8,
        }
    ])

    normalized = normalize_layer2_bucket_frame(df)

    assert not normalized.empty
    assert "max_confidence" in normalized.columns
    assert "event_signal_mean" in normalized.columns
    assert "high_impact_event" in normalized.columns
    assert pd.api.types.is_datetime64_any_dtype(normalized["timestamp"])
    assert normalized.iloc[0]["max_confidence"] == 0.8
    assert normalized.iloc[0]["sentiment_mean"] == 0.4


def test_normalize_layer2_article_frame_adds_defaults():
    df = pd.DataFrame([
        {
            "timestamp": "2026-04-30T10:05:00Z",
            "source": "rss",
            "title": "Solana partners with a major provider",
            "event_type": "partnership",
            "sentiment_score": 0.6,
            "event_weight": 2.0,
        }
    ])

    normalized = normalize_layer2_article_frame(df)

    assert not normalized.empty
    assert "event_signal" in normalized.columns
    assert "high_impact_event" in normalized.columns
    assert normalized.iloc[0]["high_impact_event"] == 1
    assert normalized.iloc[0]["event_signal"] == 1.2
