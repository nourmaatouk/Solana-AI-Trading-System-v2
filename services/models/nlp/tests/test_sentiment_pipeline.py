import pytest
import pandas as pd
from typing import Dict, Optional
from src.processing.sentiment_pipeline import process_articles


class MockAnalyzer:
    """Mock analyzer bypassing downloading Heavy FinBERT huggingface models."""
    def get_sentiment(self, text: Optional[str]) -> Dict:
        return {
            "negative_prob": 0.1,
            "neutral_prob": 0.2,
            "positive_prob": 0.7,
            "sentiment_score": 0.6,
            "confidence": 0.7,
            "sentiment_label": "positive"
        }


def test_process_articles_skips_empty():
    """Ensure articles failing text limits are safely skipped resulting in an empty dataframe."""
    analyzer = MockAnalyzer()
    
    articles = [
        {
            "source": "rss",
            "published_at": "2026-04-29T10:00:00Z",
            "title": "Too Short.",
            "description": "",
            "text": ""
        }
    ]
    
    df = process_articles(articles, analyzer=analyzer)
    assert df.empty, "Dataframe should be entirely empty due to 20-word limit exclusions"


def test_process_articles_computes_dataframe_schema():
    """Validates full integration mapping logic pushing raw strings down into DataFrames"""
    analyzer = MockAnalyzer()

    articles = [
        {
            "source": "gdelt",
            "title": (
                "A completely valid long text that surpasses the token requirement easily so it remains inside the loop perfectly. "
                "This extra sentence ensures we comfortably exceed the 20-token minimum and avoid accidental trimming by the cleaner. "
            ) * 3,
            "url": "http://example.com/long",
            "published_at": "2026-04-29T10:00:00Z",
            "feed_source": "gdelt_project"
        }
    ]

    # relevance_threshold=1.0 — this test validates output schema, not filtering behaviour
    df = process_articles(articles, analyzer=analyzer, relevance_threshold=1.0)
    
    assert not df.empty
    assert len(df) == 1
    
    # Check all explicit dataframe columns specified map correctly
    expected_columns = [
        "timestamp", "source", "feed_source", "title", "url", "cleaned_text",
        "negative_prob", "neutral_prob", "positive_prob", "sentiment_score",
        "confidence", "sentiment_label"
    ]
    for col in expected_columns:
        assert col in df.columns
        
    # Check mapped overrides format properly
    row = df.iloc[0]
    assert row["source"] == "gdelt"
    # Timestamp is parsed to UTC datetime in the pipeline
    import pandas as pd
    expected_ts = pd.to_datetime("2026-04-29T10:00:00Z", errors="coerce", utc=True)
    assert row["timestamp"] == expected_ts
    assert "surpasses the token requirement" in row["cleaned_text"]
    assert row["sentiment_label"] == "positive"
    assert row["sentiment_score"] == 0.6
