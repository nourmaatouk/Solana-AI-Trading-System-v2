import sys
from typing import List, Dict

from src.config import get_settings
from src.fetchers.gdelt_fetcher import GdeltFetcher
from src.fetchers.rss_fetcher import RssFetcher
from src.processing.sentiment_pipeline import process_articles
from src.models.finbert_wrapper import FinBertSentimentAnalyzer
from src.aggregation.feature_aggregator import aggregate_sentiment_15m
from src.storage.csv_writer import save_sentiment_features, save_processed_articles


def deduplicate_articles(articles: List[Dict]) -> List[Dict]:
    """Remove duplicates by URL or title."""
    seen_urls = set()
    seen_titles = set()
    deduped = []
    
    for article in articles:
        url = article.get("url")
        title = article.get("title")
        
        if url and url in seen_urls:
            continue
            
        if title and title in seen_titles:
            continue
            
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)
            
        deduped.append(article)
        
    return deduped


def main():
    print("Starting NLP Sentiment Pipeline...")
    
    try:
        # Load settings
        settings = get_settings()
        
        # 1. Fetch articles
        print("Fetching GDELT...")
        gdelt_articles = GdeltFetcher().fetch()
        print(f"-> GDELT: {len(gdelt_articles)} articles.")
        
        print("Fetching RSS...")
        rss_articles = RssFetcher().fetch()
        print(f"-> RSS: {len(rss_articles)} articles.")
        
        # 2. Merge and deduplicate
        all_articles = gdelt_articles + rss_articles
        raw_count = len(all_articles)
        
        unique_articles = deduplicate_articles(all_articles)
        unique_count = len(unique_articles)
        print(f"-> Deduplicated: {unique_count} unique articles.")
        
        if unique_count == 0:
            print("No new articles. Exiting.")
            sys.exit(0)
            
        # 3. Process sentiment
        print("Running FinBERT sentiment analysis...")
        df_processed = process_articles(unique_articles, analyzer=FinBertSentimentAnalyzer())
        
        if df_processed.empty:
            print("No valid articles after text cleaning. Exiting.")
            sys.exit(0)
            
        processed_count = len(df_processed)
        print(f"-> Processed: {processed_count} articles.")

        # Save granular processed articles for validation
        print("Saving article-level data for validation...")
        save_processed_articles(df_processed)

        # 4. Aggregate intervals
        print("Aggregating to 15-minute intervals...")
        df_aggregated = aggregate_sentiment_15m(df_processed)
        
        if df_aggregated.empty:
            print("Empty aggregation. Exiting.")
            sys.exit(0)
            
        bucket_count = len(df_aggregated)
        print(f"-> Generated {bucket_count} buckets.")
        
        # 5. Save output
        print("Saving features to CSV...")
        output_path = save_sentiment_features(df_aggregated)
        
        # Print summary
        print("\n=== RUN SUMMARY ===")
        print(f"Raw Articles:       {raw_count}")
        print(f"Duplicates Removed: {raw_count - unique_count}")
        print(f"Processed by ML:    {processed_count}")
        print(f"15m Buckets:        {bucket_count}")
        print(f"Output File:        {output_path}")
        print("===================\n")

    except Exception as e:
        print(f"\nERROR: Pipeline crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
