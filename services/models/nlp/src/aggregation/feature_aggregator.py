import pandas as pd
import numpy as np
from src.config import get_settings


def aggregate_sentiment_15m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups processed sentiment DataFrame into 15-minute intervals.
    Computes means, spreads, ratios, and standardizes source metrics safely.
    """
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
        
    df = df.copy()
        
    settings = get_settings()
    threshold = settings.SENTIMENT_THRESHOLD

    # Defensively ensure timestamp bounds mapped accurately just in case upstream didn't
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        
    if df.empty:
        return pd.DataFrame()

    # Bucketing definition floor
    df["bucket"] = df["timestamp"].dt.floor("15min")
    
    # Classification mapped explicit flags
    df["is_positive"] = (df["sentiment_score"] > threshold).astype(int)
    df["is_negative"] = (df["sentiment_score"] < -threshold).astype(int)
    df["is_neutral"] = ((df["sentiment_score"] >= -threshold) & (df["sentiment_score"] <= threshold)).astype(int)
    df["is_strong_pos"] = (df["sentiment_score"] > 0.8).astype(int)
    df["is_strong_neg"] = (df["sentiment_score"] < -0.8).astype(int)

    # Source flags
    df["is_gdelt"] = (df["source"] == "gdelt").astype(int)
    df["is_rss"] = (df["source"] == "rss").astype(int)

    # Event Mapping
    # Consider hack, outage, and market_move as potentially high impact
    df["is_event"] = (df["event_type"] != "general").astype(int)
    df["is_negative_event"] = ((df["is_event"] == 1) & (df["sentiment_label"] == "negative")).astype(int)
    df["is_positive_event"] = ((df["is_event"] == 1) & (df["sentiment_label"] == "positive")).astype(int)
    df["is_critical"] = df["event_type"].isin(["hack/security", "outage/network"]).astype(int)

    # Weights
    df["rel_weighted"] = df["sentiment_score"] * df["relevance_score"]
    df["ev_weighted"] = df["sentiment_score"] * (1 + df["is_event"])

    # Core group continuous dimension aggregations
    grouped = df.groupby("bucket")
    
    agg_df = pd.DataFrame()
    agg_df["sentiment_mean"] = grouped["sentiment_score"].mean()
    agg_df["sentiment_std"] = grouped["sentiment_score"].std().fillna(0)
    
    news_volume = grouped.size()
    agg_df["news_volume"] = news_volume
    
    agg_df["positive_ratio"] = grouped["is_positive"].sum() / news_volume
    agg_df["negative_ratio"] = grouped["is_negative"].sum() / news_volume
    agg_df["neutral_ratio"] = grouped["is_neutral"].sum() / news_volume
    agg_df["strong_positive_ratio"] = grouped["is_strong_pos"].sum() / news_volume
    agg_df["strong_negative_ratio"] = grouped["is_strong_neg"].sum() / news_volume
    
    agg_df["max_confidence"] = grouped["confidence"].max()
    
    # Event Features
    agg_df["event_count"] = grouped["is_event"].sum()
    agg_df["negative_event_count"] = grouped["is_negative_event"].sum()
    agg_df["positive_event_count"] = grouped["is_positive_event"].sum()
    agg_df["high_impact_event_count"] = grouped["high_impact_event"].sum()
    agg_df["high_impact_event"] = grouped["high_impact_event"].max()

    # Weighted Sentiments
    agg_df["relevance_weighted_sentiment"] = grouped["rel_weighted"].mean()
    agg_df["event_signal_mean"] = grouped["event_signal"].mean()
    agg_df["event_signal_sum"] = grouped["event_signal"].sum()

    # Extracted volumes arrays
    agg_df["gdelt_volume"] = grouped["is_gdelt"].sum()
    agg_df["rss_volume"] = grouped["is_rss"].sum()

    # Source specific sentiments mapped iteratively and populated back onto main bounds
    gdelt_subset = df[df["source"] == "gdelt"].groupby("bucket")["sentiment_score"].mean()
    rss_subset = df[df["source"] == "rss"].groupby("bucket")["sentiment_score"].mean()
    
    agg_df["gdelt_sentiment"] = gdelt_subset
    agg_df["rss_sentiment"] = rss_subset
    
    # Coalescing missing overlaps to strict zeroes
    agg_df["gdelt_sentiment"] = agg_df["gdelt_sentiment"].fillna(0)
    agg_df["rss_sentiment"] = agg_df["rss_sentiment"].fillna(0)
    
    # Sort chronological boundaries
    agg_df = agg_df.sort_index()

    # Time-series features (Momentum and Change)
    agg_df["sentiment_momentum"] = agg_df["sentiment_mean"].diff().fillna(0)
    agg_df["news_volume_change"] = agg_df["news_volume"].diff().fillna(0)

    # Compute trailing 4-interval derivation map (deviation from tracking simple moving average)
    rolling_4_mean = agg_df["sentiment_mean"].rolling(window=4, min_periods=1).mean()
    agg_df["sentiment_spike"] = (agg_df["sentiment_mean"] - rolling_4_mean).abs().fillna(0)

    # Clear categorical grouping index logic and finalize explicit DataFrame structures
    agg_df = agg_df.reset_index().rename(columns={"bucket": "timestamp"})
    
    return agg_df
