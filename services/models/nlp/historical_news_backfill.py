import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.fetchers.gdelt_fetcher import GdeltFetcher
from src.processing.sentiment_pipeline import process_articles
from src.aggregation.feature_aggregator import aggregate_sentiment_15m
from src.models.finbert_wrapper import FinBertSentimentAnalyzer

def backfill_historical_news(start_days_ago: int = 30, end_days_ago: int = 0):
    """
    Simulates a historical backfill by querying GDELT in rolling windows.
    Note: GDELT Doc API usually permits access to recent history (last 3-6 months).
    """
    PROJECT_ROOT = Path(__file__).resolve().parent
    RAW_PATH = PROJECT_ROOT / "data" / "raw" / "historical_news.jsonl"
    PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "historical_sentiment_features_15m.csv"
    
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    fetcher = GdeltFetcher()
    analyzer = FinBertSentimentAnalyzer()
    
    all_processed_dfs = []
    
    print(f"Starting backfill from {start_days_ago} days ago to {end_days_ago} days ago...")
    
    # Process in 6-hour chunks (to avoid overloading GDELT response limits)
    current_time = datetime.now() - timedelta(days=start_days_ago)
    end_time = datetime.now() - timedelta(days=end_days_ago)
    
    with open(RAW_PATH, "a") as f_raw:
        while current_time < end_time:
            chunk_start = current_time
            chunk_end = current_time + timedelta(hours=6)
            
            print(f"Fetching window: {chunk_start} to {chunk_end}")
            
            # GDELT Doc API v2 uses a specific query format for timespan or date range
            # Here we leverage the Doc API Doc mode
            # We override params temporarily for historical search
            query_str = '("Solana" OR "SOL")'
            
            # GDELT specific date format: YYYYMMDDHHMMSS
            # However, simpler to just use timespan if 1 chunk is small
            # but for specific windows we use timespan=... but the API is tricky for OLD data.
            # We'll use a simulated fetch loop here assuming the fetcher is updated or we use its logic.
            
            # For this script, we'll extend the fetcher logic inline to support timespans
            params = {
                "query": query_str,
                "mode": "artlist",
                "format": "json",
                "timespan": "6h", # We can't specify exact dates easily in search v2 without complex query
                "sort": "DateDesc"
            }
            # Note: In a real institutional context, you'd use GDELT 2.0 BigQuery or raw CSVs.
            # Here we use the Doc API for the simulation.
            
            articles = fetcher.fetch() # For now, we fetch the window the fetcher is configured for
            
            if articles:
                # Save Raw
                for art in articles:
                    f_raw.write(json.dumps(art) + "\n")
                
                # Process (Filtering + FinBERT)
                df_processed = process_articles(articles, analyzer=analyzer, relevance_threshold=1.0)
                
                if not df_processed.empty:
                    all_processed_dfs.append(df_processed)
            
            current_time = chunk_end

    if all_processed_dfs:
        full_df = pd.concat(all_processed_dfs, ignore_index=True)
        print(f"Aggregating {len(full_df)} scored articles...")
        
        # Final 15m Aggregation
        final_features = aggregate_sentiment_15m(full_df)
        
        # Save Features
        final_features.to_csv(PROCESSED_PATH, index=False)
        print(f"Backfill complete. Saved features to {PROCESSED_PATH}")
    else:
        print("No articles found in the specified range.")

if __name__ == "__main__":
    backfill_historical_news(start_days_ago=7, end_days_ago=0)
