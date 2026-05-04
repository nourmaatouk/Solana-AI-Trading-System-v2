import pandas as pd

from src.aggregation.feature_aggregator import aggregate_sentiment_15m
from src.processing.sentiment_pipeline import process_articles


class MockAnalyzer:
	def get_sentiment(self, text):
		return {
			"negative_prob": 0.05,
			"neutral_prob": 0.10,
			"positive_prob": 0.85,
			"sentiment_score": 0.72,
			"confidence": 0.93,
			"sentiment_label": "positive",
		}


def test_nlp_contract_process_to_aggregate_pipeline():
	articles = [
		{
			"published_at": "2026-04-30T10:05:00Z",
			"source": "gdelt",
			"feed_source": "gdelt_project",
			"title": "Solana partners with a major infrastructure provider",
			"description": "The announcement mentions a broader ecosystem integration.",
			"text": "This article is long enough to pass the cleaner and contains the expected signal terms for the contract.",
			"url": "http://example.com/article-1",
		},
		{
			"published_at": "2026-04-30T10:12:00Z",
			"source": "rss",
			"feed_source": "rss_feed",
			"title": "Network outage affects validators",
			"description": "Operators report a brief service interruption.",
			"text": "The outage article is also long enough to survive cleaning and should remain in the pipeline.",
			"url": "http://example.com/article-2",
		},
	]

	processed = process_articles(articles, analyzer=MockAnalyzer(), relevance_threshold=1.0)

	assert not processed.empty
	for column in [
		"timestamp",
		"source",
		"sentiment_score",
		"sentiment_label",
		"event_type",
		"event_signal",
		"high_impact_event",
		"relevance_score",
	]:
		assert column in processed.columns

	agg_df = aggregate_sentiment_15m(processed)

	assert not agg_df.empty
	assert "timestamp" in agg_df.columns
	assert "event_count" in agg_df.columns
	assert agg_df["event_count"].sum() >= 1
	assert agg_df["high_impact_event"].max() in [0, 1]
