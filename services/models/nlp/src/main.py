import sys

from src.config import get_settings
from src.fetchers.gdelt_fetcher import GdeltFetcher
from src.fetchers.rss_fetcher import RssFetcher
from src.fetchers.reddit_fetcher import RedditFetcher
from src.processing.sentiment_pipeline import process_articles
from src.models.finbert_wrapper import FinBertSentimentAnalyzer
from src.aggregation.feature_aggregator import aggregate_sentiment_15m
from src.storage.csv_writer import save_sentiment_features, save_processed_articles
from src.storage.redis_cache import RedisCache
from src.storage.timescale_writer import TimescaleWriter
from src.utils.deduplicator import deduplicate_articles
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _apply_sentiment_decay(df_aggregated, cache: RedisCache):
    """
    If this run produced no aggregated rows, fall back to the last cached bucket
    so downstream consumers always receive a non-empty sentiment row.
    Decays the sentiment score by 50% to signal staleness.
    """
    if not df_aggregated.empty:
        return df_aggregated

    last = cache.get_latest()
    if last is None:
        return df_aggregated

    import pandas as pd
    decayed = dict(last)
    for col in ["sentiment_mean", "relevance_weighted_sentiment", "event_signal_sum"]:
        if col in decayed and decayed[col] is not None:
            try:
                decayed[col] = float(decayed[col]) * 0.5
            except (TypeError, ValueError):
                pass
    decayed["news_volume"] = 0
    logger.info("No fresh articles — emitting decayed sentiment row from cache.")
    return pd.DataFrame([decayed])


def main():
    logger.info("Starting NLP Sentiment Pipeline...")

    settings = get_settings()
    cache = RedisCache()
    ts_writer = TimescaleWriter()

    try:
        # 1. Fetch articles
        logger.info("Fetching GDELT...")
        gdelt_articles = GdeltFetcher().fetch()
        logger.info(f"GDELT: {len(gdelt_articles)} articles.")

        logger.info("Fetching RSS...")
        rss_articles = RssFetcher().fetch()
        logger.info(f"RSS: {len(rss_articles)} articles.")

        reddit_articles = []
        if settings.REDDIT_FETCH_ENABLED:
            logger.info("Fetching Reddit...")
            reddit_articles = RedditFetcher().fetch()
            logger.info(f"Reddit: {len(reddit_articles)} posts.")

        # 2. Merge and deduplicate
        all_articles = gdelt_articles + rss_articles + reddit_articles
        raw_count = len(all_articles)

        unique_articles = deduplicate_articles(all_articles)
        unique_count = len(unique_articles)
        logger.info(f"Deduplicated: {unique_count} unique articles ({raw_count - unique_count} removed).")

        if unique_count == 0:
            logger.warning("No new articles fetched.")
            import pandas as pd
            df_aggregated = _apply_sentiment_decay(pd.DataFrame(), cache)
            if not df_aggregated.empty:
                save_sentiment_features(df_aggregated)
                cache.cache_dataframe(df_aggregated)
            sys.exit(0)

        # 3. Process sentiment
        logger.info("Running FinBERT sentiment analysis...")
        df_processed = process_articles(unique_articles, analyzer=FinBertSentimentAnalyzer())

        if df_processed.empty:
            logger.warning("No valid articles after text cleaning.")
            import pandas as pd
            df_aggregated = _apply_sentiment_decay(pd.DataFrame(), cache)
            if not df_aggregated.empty:
                save_sentiment_features(df_aggregated)
                cache.cache_dataframe(df_aggregated)
            sys.exit(0)

        processed_count = len(df_processed)
        logger.info(f"Processed: {processed_count} articles.")

        logger.info("Saving article-level data for validation...")
        save_processed_articles(df_processed)

        # 4. Aggregate to 15-minute buckets
        logger.info("Aggregating to 15-minute intervals...")
        df_aggregated = aggregate_sentiment_15m(df_processed)
        df_aggregated = _apply_sentiment_decay(df_aggregated, cache)

        if df_aggregated.empty:
            logger.error("Empty aggregation even after decay fallback. Exiting.")
            sys.exit(0)

        bucket_count = len(df_aggregated)
        logger.info(f"Generated {bucket_count} 15m buckets.")

        # 5. Persist outputs
        logger.info("Saving features to CSV...")
        output_path = save_sentiment_features(df_aggregated)

        logger.info("Caching features to Redis...")
        cached_rows = cache.cache_dataframe(df_aggregated)

        logger.info("Writing features to TimescaleDB...")
        written_rows = ts_writer.write(df_aggregated)

        # Summary
        logger.info("=== RUN SUMMARY ===")
        logger.info(f"Raw Articles:       {raw_count}")
        logger.info(f"Duplicates Removed: {raw_count - unique_count}")
        logger.info(f"Processed by ML:    {processed_count}")
        logger.info(f"15m Buckets:        {bucket_count}")
        logger.info(f"CSV Output:         {output_path}")
        logger.info(f"Redis Cached:       {cached_rows} rows")
        logger.info(f"TimescaleDB Rows:   {written_rows}")
        logger.info("===================")

    except Exception as e:
        logger.error(f"Pipeline crashed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        ts_writer.close()


if __name__ == "__main__":
    main()
