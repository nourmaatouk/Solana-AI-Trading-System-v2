from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

# Add the parent nlp directory to sys.path to allow importing from src
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
NLP_DIR = PROJECT_ROOT / "nlp"
sys.path.append(str(NLP_DIR))

try:
    from src.fetchers.gdelt_fetcher import GdeltFetcher
    from src.fetchers.rss_fetcher import RssFetcher
    from src.processing.sentiment_pipeline import process_articles
    from src.models.finbert_wrapper import FinBertSentimentAnalyzer
    from src.aggregation.feature_aggregator import aggregate_sentiment_15m
    from src.contracts.layer2_contract import normalize_layer2_bucket_frame, BUCKET_DEFAULTS
except ImportError:
    # If standard imports fail due to path structure, try relative to models
    sys.path.append(str(PROJECT_ROOT / "nlp"))
    # (re-imports would happen here if needed, but the first one usually works if path is right)

# =========================================================
# CONFIG
# =========================================================
OUTPUT_PATH = BASE_DIR / "data" / "realtime" / "latest_sentiment_15m.csv"
CACHE_PATH = BASE_DIR / "data" / "realtime" / "processed_urls.txt"

REQUIRED_COLUMNS = [
    "timestamp", "sentiment_mean", "sentiment_std", "news_volume",
    "positive_ratio", "negative_ratio", "neutral_ratio",
    "strong_positive_ratio", "strong_negative_ratio",
    "max_confidence", "event_count", "negative_event_count",
    "positive_event_count", "high_impact_event_count", "high_impact_event",
    "relevance_weighted_sentiment", "event_signal_mean", "event_signal_sum",
    "gdelt_volume", "rss_volume", "gdelt_sentiment", "rss_sentiment",
    "sentiment_momentum", "news_volume_change", "sentiment_spike"
]

DECAY_FACTOR = 0.5  # Decay sentiment by 50% if no new news

def create_zero_row() -> pd.DataFrame:
    """Creates a fallback row with all sentiment metrics set to zero."""
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    data = {col: [0.0] for col in REQUIRED_COLUMNS}
    data["timestamp"] = [now]
    return normalize_layer2_bucket_frame(pd.DataFrame(data))

def get_decayed_row() -> pd.DataFrame:
    """
    Loads previous sentiment and decays it.
    Ensures the model doesn't just see a hard 'zero' jump.
    """
    if not OUTPUT_PATH.exists():
        return create_zero_row()
    
    try:
        prev_df = pd.read_csv(OUTPUT_PATH)
        if prev_df.empty:
            return create_zero_row()
            
        row = prev_df.iloc[-1].copy()
        
        # Apply decay to sentiment metrics (not volumes or timestamps)
        sentiment_cols = [
            "sentiment_mean", "sentiment_std", "max_confidence",
            "gdelt_sentiment", "rss_sentiment", "sentiment_spike",
            "relevance_weighted_sentiment", "sentiment_momentum",
            "positive_ratio", "negative_ratio"
        ]
        
        for col in sentiment_cols:
            if col in row:
                row[col] *= DECAY_FACTOR
        
        # Reset volumes
        for col in ["news_volume", "gdelt_volume", "rss_volume", "event_count", "news_volume_change"]:
            if col in row:
                row[col] = 0.0
                
        row["timestamp"] = pd.Timestamp.now(tz="UTC").floor("15min")

        return normalize_layer2_bucket_frame(pd.DataFrame([row]))
    except Exception as e:
        print(f"Decay error: {e}")
        return create_zero_row()

def load_cached_urls() -> set:
    if not CACHE_PATH.exists():
        return set()
    with open(CACHE_PATH, "r") as f:
        return set(line.strip() for line in f if line.strip())

def save_new_urls(new_urls: List[str]):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "a") as f:
        for url in new_urls:
            f.write(url + "\n")

def main():
    print("=" * 60)
    print("Real-time Sentiment Fetcher Starting")
    print("=" * 60)

    try:
        # Load cache
        seen_urls = load_cached_urls()
        print(f"Loaded {len(seen_urls)} cached URLs.")

        # 1. Fetch latest articles
        print("Fetching GDELT and RSS articles...")
        gdelt_articles = GdeltFetcher().fetch()
        rss_articles = RssFetcher().fetch()
        all_articles = gdelt_articles + rss_articles
        
        # Deduplicate
        fresh_articles = [a for a in all_articles if a.get("url") not in seen_urls]
        print(f"Found {len(all_articles)} total articles. {len(fresh_articles)} are new.")
        
        if not fresh_articles:
            print("No new articles found currently. Applying sentiment decay fallback.")
            df_final = get_decayed_row()
        else:
            print(f"Processing {len(fresh_articles)} new articles with FinBERT...")
            
            # 2. Extract sentiment
            analyzer = FinBertSentimentAnalyzer()
            # Note: relevance_threshold=1.0 is handled inside process_articles
            df_processed = process_articles(fresh_articles, analyzer=analyzer, relevance_threshold=1.0)
            
            if df_processed.empty:
                print("No relevant articles after filtering. Applying sentiment decay fallback.")
                df_final = get_decayed_row()
            else:
                # Cache new URLs
                save_new_urls(df_processed["url"].tolist())
                
                # 3. Aggregate into 15m intervals
                print("Aggregating sentiment into 15-minute buckets...")
                df_aggregated = aggregate_sentiment_15m(df_processed)
                
                if df_aggregated.empty:
                    df_final = get_decayed_row()
                else:
                    # Take only the latest bucket
                    df_final = normalize_layer2_bucket_frame(df_aggregated.sort_values("timestamp").tail(1).copy())

        # 4. Final Format Check
        for col in REQUIRED_COLUMNS:
            if col not in df_final.columns:
                df_final[col] = 0.0
        
        df_final = normalize_layer2_bucket_frame(df_final)[REQUIRED_COLUMNS]

        # 5. Save Output
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(OUTPUT_PATH, index=False)
        
        print(f"Real-time sentiment saved to: {OUTPUT_PATH}")
        print(f"Timestamp: {df_final['timestamp'].iloc[0]}")
        print(f"Mean Sentiment: {df_final['sentiment_mean'].iloc[0]:.4f}")
        print(f"Volume: {df_final['news_volume'].iloc[0]}")
        
    except Exception as e:
        print(f"Error in sentiment fetcher: {e}")
        print("Falling back to safety decayed row.")
        df_final = get_decayed_row()
        df_final.to_csv(OUTPUT_PATH, index=False)

    print("=" * 60)

if __name__ == "__main__":
    main()
