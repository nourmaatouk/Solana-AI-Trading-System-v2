import pytest
import pandas as pd
from src.aggregation.feature_aggregator import aggregate_sentiment_15m


def test_empty_df_returns_empty():
    """Ensure aggregation pipeline nullifies safely against empty dataframes."""
    df = pd.DataFrame()
    agg_df = aggregate_sentiment_15m(df)
    assert agg_df.empty


def test_aggregation_computes_structurally():
    """Test 15-minute bucketing mapping layout, volume scaling, and explicit ratios alignment limits."""
    raw_data = [
        # Bucket 1: 10:00 - 10:15
        {"timestamp": "2026-04-29T10:05:00Z", "source": "gdelt", "sentiment_score": 0.8, "confidence": 0.9},
        {"timestamp": "2026-04-29T10:12:00Z", "source": "rss", "sentiment_score": 0.2, "confidence": 0.6},
        # Bucket 2: 10:15 - 10:30
        {"timestamp": "2026-04-29T10:20:00Z", "source": "rss", "sentiment_score": -0.5, "confidence": 0.8},
        # Bucket 3: 10:30 - 10:45
        {"timestamp": "2026-04-29T10:35:00Z", "source": "gdelt", "sentiment_score": 0.05, "confidence": 0.4},
    ]
    df = pd.DataFrame(raw_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    agg_df = aggregate_sentiment_15m(df)
    
    assert not agg_df.empty
    assert len(agg_df) == 3
    
    # Assert temporal sequence floors map sequentially explicitly
    assert agg_df.iloc[0]["timestamp"] == pd.to_datetime("2026-04-29T10:00:00Z", utc=True)
    assert agg_df.iloc[1]["timestamp"] == pd.to_datetime("2026-04-29T10:15:00Z", utc=True)
    assert agg_df.iloc[2]["timestamp"] == pd.to_datetime("2026-04-29T10:30:00Z", utc=True)
    
    row_1000 = agg_df.iloc[0]
    
    # Validate volume allocations cleanly assigned
    assert row_1000["news_volume"] == 2
    assert row_1000["gdelt_volume"] == 1
    assert row_1000["rss_volume"] == 1
    
    # Validation testing on explicitly bounded sentiment score ratios mathematically
    assert row_1000["positive_ratio"] == 1.0 # 0.8 and 0.2 both trigger bounding limit requirements uniquely above Config bounds
    assert row_1000["negative_ratio"] == 0.0
    assert row_1000["neutral_ratio"] == 0.0
    
    # Sum bounds exactly equals 1.0 natively across bounds 
    assert row_1000["positive_ratio"] + row_1000["negative_ratio"] + row_1000["neutral_ratio"] == 1.0
    
    assert row_1000["max_confidence"] == 0.9
    assert row_1000["sentiment_mean"] == 0.5 
    assert "sentiment_spike" in row_1000
    
    # Validate neutral allocation mapped natively backwards structurally
    row_1030 = agg_df.iloc[2]
    assert row_1030["neutral_ratio"] == 1.0 # 0.05 falls under default bound +/-0.10 limits


def test_spike_computation():
    """Test deviation metric maps spikes tracking effectively from moving average history logic."""
    raw_data = [
        {"timestamp": "2026-04-29T10:05:00Z", "source": "rss", "sentiment_score": 0.5, "confidence": 0.9},
        {"timestamp": "2026-04-29T10:20:00Z", "source": "rss", "sentiment_score": 0.5, "confidence": 0.9},
        {"timestamp": "2026-04-29T10:35:00Z", "source": "rss", "sentiment_score": 0.5, "confidence": 0.9},
        {"timestamp": "2026-04-29T10:50:00Z", "source": "rss", "sentiment_score": 0.5, "confidence": 0.9},
        {"timestamp": "2026-04-29T11:05:00Z", "source": "rss", "sentiment_score": -0.9, "confidence": 0.9}, # Heavy negative burst.
    ]
    df = pd.DataFrame(raw_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    agg_df = aggregate_sentiment_15m(df)
    
    assert len(agg_df) == 5
    
    # Continual mapping implies near 0 bounds logic structurally
    assert pytest.approx(agg_df.iloc[3]["sentiment_spike"], 0.01) == 0.0
    
    # Immediate disruption effectively scales rolling means dynamically alerting the flags mapping.
    assert agg_df.iloc[4]["sentiment_spike"] > 0.0
