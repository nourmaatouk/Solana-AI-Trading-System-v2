import unittest
import pandas as pd
from src.processing.text_cleaner import calculate_relevance_score
from src.processing.sentiment_pipeline import process_articles

class TestRelevanceFilter(unittest.TestCase):
    
    def test_relevance_scoring(self):
        """Test that different keywords produce expected scores."""
        # Solana and SOL (1.5 each)
        score_sol = calculate_relevance_score("Solana is a great network")
        self.assertEqual(score_sol, 1.5)
        
        # Other keywords (1.0 each)
        score_etf = calculate_relevance_score("The Bitcoin ETF was approved")
        self.assertEqual(score_etf, 1.0)
        
        # Mixed
        score_mixed = calculate_relevance_score("Solana outage reported after hack")
        # solana(1.5) + outage(1.0) + hack(1.0) = 3.5
        self.assertEqual(score_mixed, 3.5)
        
        # Irrelevant
        score_none = calculate_relevance_score("The weather is nice today in Paris")
        self.assertEqual(score_none, 0.0)

    def test_pipeline_filtering(self):
        """Test that the pipeline correctly drops irrelevant articles."""
        articles = [
            {
                "published_at": "2026-04-30T10:00:00Z",
                "title": "Solana hits new all time high",
                "text": "SOL is performing well",
                "source": "test"
            },
            {
                "published_at": "2026-04-30T10:05:00Z",
                "title": "Unrelated news about a movie",
                "text": "The actors were great",
                "source": "test"
            }
        ]
        
        # Mock analyzer to avoid loading FinBERT in tests
        class MockAnalyzer:
            def get_sentiment(self, text):
                return {"sentiment_score": 0.5, "sentiment_label": "positive", "confidence": 0.9}
        
        # With threshold 1.0, only the Solana article should pass
        df = process_articles(articles, analyzer=MockAnalyzer(), relevance_threshold=1.0)
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["title"], "Solana hits new all time high")
        self.assertIn("relevance_score", df.columns)
        self.assertGreaterEqual(df.iloc[0]["relevance_score"], 1.0)

if __name__ == "__main__":
    unittest.main()
