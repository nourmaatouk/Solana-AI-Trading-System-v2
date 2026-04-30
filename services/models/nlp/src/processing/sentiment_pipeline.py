import pandas as pd
from typing import List, Dict, Optional
from src.processing.text_cleaner import prepare_article_text
from src.models.finbert_wrapper import FinBertSentimentAnalyzer


def process_articles(articles: List[Dict], analyzer: Optional[FinBertSentimentAnalyzer] = None) -> pd.DataFrame:
    """
    Processes raw articles fetched from external feeds.
    Drops short/empty payloads and maps structural inputs dynamically into DataFrames.
    """
    if not articles:
        return pd.DataFrame()

    if analyzer is None:
        analyzer = FinBertSentimentAnalyzer()

    processed_data = []

    for article in articles:
        cleaned_text = prepare_article_text(article)
        
        # Skip articles where cleaned text dropped below the minimum boundaries (e.g < 20 tokens)
        if not cleaned_text:
            continue

        sentiment_results = analyzer.get_sentiment(cleaned_text)

        processed_data.append({
            "timestamp": article.get("published_at"),
            "source": article.get("source"),
            "feed_source": article.get("feed_source"),
            "title": article.get("title"),
            "url": article.get("url"),
            "cleaned_text": cleaned_text,
            "negative_prob": sentiment_results.get("negative_prob"),
            "neutral_prob": sentiment_results.get("neutral_prob"),
            "positive_prob": sentiment_results.get("positive_prob"),
            "sentiment_score": sentiment_results.get("sentiment_score"),
            "confidence": sentiment_results.get("confidence"),
            "sentiment_label": sentiment_results.get("sentiment_label")
        })

    df = pd.DataFrame(processed_data)
    
    # Cast timestamp implicitly to UTC datetime bounds for downstream aggregations
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        
    return df
