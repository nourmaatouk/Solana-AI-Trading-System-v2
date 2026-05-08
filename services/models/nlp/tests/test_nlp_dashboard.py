from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "22_nlp_sentiment_dashboard.py"
SPEC = importlib.util.spec_from_file_location("nlp_sentiment_dashboard", MODULE_PATH)
dashboard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(dashboard)


def test_build_html_report_renders_sentiment_and_articles():
    sentiment_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-07T10:00:00Z",
                "sentiment_mean": -0.25,
                "news_volume": 2,
                "positive_ratio": 0.25,
                "negative_ratio": 0.75,
                "event_signal_sum": -1.2,
                "sentiment_spike": 0.4,
                "high_impact_event_count": 1,
            }
        ]
    )
    articles_df = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-07T09:58:00Z",
                "title": "Solana outage pressure hits market",
                "event_type": "outage/network",
                "sentiment_score": -0.8,
                "event_signal": -2.4,
                "sentiment_label": "negative",
            }
        ]
    )
    realtime_df = pd.DataFrame()

    result = dashboard.build_html_report(sentiment_df, articles_df, realtime_df, row_limit=20)

    assert "NLP Sentiment Dashboard" in result
    assert "Solana outage pressure hits market" in result
    assert "outage/network" in result
    assert "-0.2500" in result


def test_build_html_report_handles_missing_data():
    result = dashboard.build_html_report(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), row_limit=20)

    assert "No sentiment bucket rows found" in result
    assert "No processed article rows found" in result
    assert "Latest source: missing" in result
