import unittest
import pandas as pd
from src.processing.text_cleaner import classify_event_type
from src.aggregation.feature_aggregator import aggregate_sentiment_to_15m

class TestEventDetection(unittest.TestCase):
    
    def test_event_classification(self):
        """Test that text is correctly mapped to event categories."""
        self.assertEqual(classify_event_type("Solana network outage fixed"), "outage/network")
        self.assertEqual(classify_event_type("A major hack on Raydium"), "hack/security")
        self.assertEqual(classify_event_type("SEC approves Solana ETF"), "ETF/regulation")
        self.assertEqual(classify_event_type("Solana partners with Google"), "partnership")
        self.assertEqual(classify_event_type("Price surge after pump"), "market_move")
        self.assertEqual(classify_event_type("Whale moves 1M SOL"), "whale")
        self.assertEqual(classify_event_type("Normal day in crypto"), "general")

    def test_event_aggregation(self):
        """Test that event features are correctly aggregated into 15m buckets."""
        df = pd.DataFrame([
            {
                "bucket": "2026-04-30 10:00",
                "event_type": "outage/network",
                "sentiment_score": -0.8,
                "sentiment_label": "negative",
                "confidence": 0.9,
                "source": "gdelt",
                "is_positive": 0, "is_negative": 1, "is_neutral": 0
            },
            {
                "bucket": "2026-04-30 10:00",
                "event_type": "partnership",
                "sentiment_score": 0.9,
                "sentiment_label": "positive",
                "confidence": 0.95,
                "source": "rss",
                "is_positive": 1, "is_negative": 0, "is_neutral": 0
            }
        ])
        
        agg_df = aggregate_sentiment_to_15m(df)
        
        # event_count should be 2
        self.assertEqual(agg_df.iloc[0]["event_count"], 2)
        # negative_event_count (outage) should be 1
        self.assertEqual(agg_df.iloc[0]["negative_event_count"], 1)
        # positive_event_count (partnership) should be 1
        self.assertEqual(agg_df.iloc[0]["positive_event_count"], 1)
        # high_impact_event (outage is critical) should be 1
        self.assertEqual(agg_df.iloc[0]["high_impact_event"], 1)

if __name__ == "__main__":
    unittest.main()
