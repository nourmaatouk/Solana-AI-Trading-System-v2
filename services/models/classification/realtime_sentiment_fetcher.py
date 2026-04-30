from __future__ import annotations

import sys
from pathlib import Path
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
except ImportError:
    # If standard imports fail due to path structure, try relative to models
    sys.path.append(str(PROJECT_ROOT / "nlp"))
    # (re-imports would happen here if needed, but the first one usually works if path is right)

# = :=======================================================
# CONFIG
# =========================================================
OUTPUT_PATH = BASE_DIR / "data" / "realtime" / "latest_sentiment_15m.csv"

REQUIRED_COLUMNS = [
    "timestamp", "sentiment_mean", "sentiment_std", "news_volume",
    "positive_ratio", "negative_ratio", "neutral_ratio",
    "max_confidence", "gdelt_volume", "rss_volume",
    "gdelt_sentiment", "rss_sentiment", "sentiment_spike"
]

def create_zero_row() -> pd.DataFrame:
    """Creates a fallback row with all sentiment metrics set to zero."""
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    data = {col: [0.0] for col in REQUIRED_COLUMNS}
    data["timestamp"] = [now]
    return pd.DataFrame(data)

def main():
    print("=" * 60)
    print("Real-time Sentiment Fetcher Starting")
    print("=" * 60)

    try:
        # 1. Fetch latest articles
        print("Fetching GDELT and RSS articles...")
        gdelt_articles = GdeltFetcher().fetch()
        rss_articles = RssFetcher().fetch()
        all_articles = gdelt_articles + rss_articles
        
        if not all_articles:
            print("No articles found currently. Using fallback zero row.")
            df_final = create_zero_row()
        else:
            print(f"Found {len(all_articles)} articles. Processing with FinBERT...")
            
            # 2. Extract sentiment
            analyzer = FinBertSentimentAnalyzer()
            df_processed = process_articles(all_articles, analyzer=analyzer)
            
            if df_processed.empty:
                print("No valid articles after cleaning. Using fallback zero row.")
                df_final = create_zero_row()
            else:
                # 3. Aggregate into 15m intervals
                print("Aggregating sentiment into 15-minute buckets...")
                df_aggregated = aggregate_sentiment_15m(df_processed)
                
                if df_aggregated.empty:
                    df_final = create_zero_row()
                else:
                    # Take only the latest bucket
                    df_final = df_aggregated.sort_values("timestamp").tail(1).copy()

        # 4. Final Format Check
        # Ensure all required columns exist (even if aggregated output was missing some)
        for col in REQUIRED_COLUMNS:
            if col not in df_final.columns:
                df_final[col] = 0.0
        
        # Select and order columns
        df_final = df_final[REQUIRED_COLUMNS]

        # 5. Save Output
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(OUTPUT_PATH, index=False)
        
        print(f"Real-time sentiment saved to: {OUTPUT_PATH}")
        print(f"Timestamp: {df_final['timestamp'].iloc[0]}")
        print(f"Mean Sentiment: {df_final['sentiment_mean'].iloc[0]:.4f}")
        print(f"Volume: {df_final['news_volume'].iloc[0]}")
        
    except Exception as e:
        print(f"Error in sentiment fetcher: {e}")
        print("Falling back to safety zero row.")
        df_final = create_zero_row()
        df_final.to_csv(OUTPUT_PATH, index=False)

    print("=" * 60)

if __name__ == "__main__":
    main()
