import pandas as pd
from typing import List, Dict, Optional
from src.processing.text_cleaner import prepare_article_text, calculate_relevance_score, classify_event_type, get_event_weight
from src.models.finbert_wrapper import FinBertSentimentAnalyzer


def process_articles(
    articles: List[Dict], 
    analyzer: Optional[FinBertSentimentAnalyzer] = None,
    relevance_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Processes raw articles fetched from external feeds.
    Drops short/empty payloads and filters by relevance score.
    """
    if not articles:
        return pd.DataFrame()

    if analyzer is None:
        analyzer = FinBertSentimentAnalyzer()

    processed_data = []
    raw_count = len(articles)
    dropped_count = 0
    filtered_count = 0

    for article in articles:
        cleaned_text = prepare_article_text(article)
        
        # 1. Skip articles where cleaned text dropped below min boundaries
        if not cleaned_text:
            dropped_count += 1
            continue
            
        # 2. Relevance Filtering
        rel_score = calculate_relevance_score(cleaned_text)
        if rel_score < relevance_threshold:
            filtered_count += 1
            continue

        # 3. Sentiment Analysis
        sentiment_results = analyzer.get_sentiment(cleaned_text)
        score = sentiment_results.get("sentiment_score", 0.0)
        
        # 4. Event Logic
        etype = classify_event_type(cleaned_text)
        eweight = get_event_weight(etype)

        processed_data.append({
            "timestamp": article.get("published_at"),
            "source": article.get("source"),
            "feed_source": article.get("feed_source"),
            "title": article.get("title"),
            "url": article.get("url"),
            "cleaned_text": cleaned_text,
            "relevance_score": rel_score,
            "event_type": etype,
            "event_weight": eweight,
            "event_signal": score * eweight,
            "high_impact_event": 1 if eweight >= 2.0 else 0,
            "negative_prob": sentiment_results.get("negative_prob"),
            "neutral_prob": sentiment_results.get("neutral_prob"),
            "positive_prob": sentiment_results.get("positive_prob"),
            "sentiment_score": score,
            "confidence": sentiment_results.get("confidence"),
            "sentiment_label": sentiment_results.get("sentiment_label")
        })

    kept_count = len(processed_data)
    
    print("\n--- Relevance Filtering Stats ---")
    print(f"Raw Articles Count:      {raw_count}")
    print(f"Dropped (Short/Empty):   {dropped_count}")
    print(f"Filtered (Relevance):    {filtered_count}")
    print(f"Kept Articles:           {kept_count}")
    if raw_count > 0:
        print(f"Percentage Kept:         {(kept_count / raw_count) * 100:.2f}%")
    print("---------------------------------\n")

    df = pd.DataFrame(processed_data)
    
    # Cast timestamp implicitly to UTC datetime bounds for downstream aggregations
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        
    return df
