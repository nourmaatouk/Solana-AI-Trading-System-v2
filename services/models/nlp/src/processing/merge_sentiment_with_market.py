import pandas as pd
from pathlib import Path


def merge_sentiment_features(
    market_path: str = "data/processed/sol_usdt_15m_features.csv",
    sentiment_path: str = "data/processed/sentiment_features_15m.csv",
    output_path: str = "data/processed/sol_usdt_15m_with_sentiment.csv"
) -> pd.DataFrame:
    """Merge market price data with advanced temporal NLP sentiment features."""
    
    market_df = pd.read_csv(market_path)
    sentiment_df = pd.read_csv(sentiment_path)
    
    # 1. Convert to UTC datetime
    market_df["timestamp"] = pd.to_datetime(market_df["timestamp"], utc=True)
    sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"], utc=True)
    
    # 2. Left join
    merged_df = pd.merge(market_df, sentiment_df, on="timestamp", how="left")
    
    # Sort to enforce strict chronological order before computing ffill/lags
    merged_df = merged_df.sort_values("timestamp")
    
    # 3. Forward-fill sentiment values to preserve past data, then fillna(0) for leading voids
    sentiment_cols = [col for col in sentiment_df.columns if col != "timestamp"]
    for col in sentiment_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].ffill().fillna(0)
    
    # 4. Add advanced temporal/event-based features
    if "sentiment_mean" in merged_df.columns and "news_volume" in merged_df.columns:
        
        # 1. Weighted sentiment
        merged_df["sentiment_weighted"] = merged_df["sentiment_mean"] * merged_df["news_volume"]
        
        # Bonus: Lagged weighted metrics (1-3)
        for lag in range(1, 4):
            merged_df[f"sentiment_weighted_lag_{lag}"] = merged_df["sentiment_weighted"].shift(lag)
            
        # 2. News presence signal
        merged_df["has_news"] = (merged_df["news_volume"] > 0).astype(int)
        
        # 3. Strong sentiment spike (z-score deviation from 8-period rolling mean)
        rolling_mean_8 = merged_df["sentiment_mean"].rolling(8).mean()
        rolling_std_8 = merged_df["sentiment_mean"].rolling(8).std()
        merged_df["sentiment_spike_strong"] = (
            (merged_df["sentiment_mean"] - rolling_mean_8) / (rolling_std_8 + 1e-6)
        )
        
        # Weighted volatility
        merged_df["sentiment_weighted_volatility"] = merged_df["sentiment_weighted"].rolling(4).std()
        
        # News shock
        merged_df["news_jump"] = merged_df["news_volume"].diff().fillna(0)
        
        # Extreme sentiment flag
        merged_df["extreme_sentiment"] = (
            (merged_df["sentiment_mean"] > 0.3) | 
            (merged_df["sentiment_mean"] < -0.3)
        ).astype(int)
        
        # Informative news event signal
        merged_df["high_volume_news"] = (merged_df["news_volume"] >= 3).astype(int)
        merged_df["strong_sentiment"] = (merged_df["sentiment_mean"].abs() > 0.2).astype(int)
        merged_df["event_signal"] = merged_df["high_volume_news"] * merged_df["strong_sentiment"]
        
        # Interaction with market regime (if technicals are present)
        if "ema_20" in merged_df.columns and "sma_20" in merged_df.columns:
            merged_df["trend_regime"] = (merged_df["ema_20"] > merged_df["sma_20"]).astype(int)
            merged_df["sentiment_x_trend"] = merged_df["sentiment_mean"] * merged_df["trend_regime"]
        
        # 4. Directional sentiment pressure
        if "positive_ratio" in merged_df.columns and "negative_ratio" in merged_df.columns:
            merged_df["positive_pressure"] = merged_df["positive_ratio"] - merged_df["negative_ratio"]
            
        # 5. Delayed sentiment (market reaction lag)
        merged_df["sentiment_delayed"] = merged_df["sentiment_mean"].shift(2)
        
        # Preserving original temporal features
        for lag in range(1, 5):
            merged_df[f"sentiment_mean_lag_{lag}"] = merged_df["sentiment_mean"].shift(lag)
            
        merged_df["sentiment_rolling_mean_4"] = merged_df["sentiment_mean"].rolling(window=4).mean()
        merged_df["sentiment_rolling_std_4"] = merged_df["sentiment_mean"].rolling(window=4).std()
        merged_df["sentiment_trend"] = merged_df["sentiment_mean"].diff(1)

    # 5. Fill remaining NaNs (e.g. from early rolling windows)
    merged_df = merged_df.ffill().fillna(0)
            
    # 6. Save output
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    return merged_df


if __name__ == "__main__":
    print("Starting market and sentiment merge...")
    try:
        merged = merge_sentiment_features()
        print(f"Merge complete. Output saved with {len(merged)} rows.")
    except Exception as e:
        print(f"Merge error: {e}")
