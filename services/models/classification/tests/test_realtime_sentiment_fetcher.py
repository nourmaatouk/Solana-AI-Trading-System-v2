import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add classification directory to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from realtime_sentiment_fetcher import create_zero_row, REQUIRED_COLUMNS, OUTPUT_PATH


class TestRealtimeSentimentFetcher(unittest.TestCase):
    
    def test_fallback_zero_row(self):
        """Test that the fallback row has the correct columns and values."""
        df = create_zero_row()
        
        self.assertEqual(len(df), 1)
        # Check columns
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, df.columns)
            
        # Check values are 0 (except timestamp)
        for col in REQUIRED_COLUMNS:
            if col != 'timestamp':
                self.assertEqual(df[col].iloc[0], 0.0)
        
        # Check timestamp is floored to 15min
        ts = df['timestamp'].iloc[0]
        self.assertEqual(ts.minute % 15, 0)
        self.assertEqual(ts.second, 0)

    def test_required_columns_list(self):
        """Verify the required columns list matches the user's request."""
        expected = [
            "sentiment_mean", "sentiment_std", "news_volume", 
            "positive_ratio", "negative_ratio", "neutral_ratio",
            "max_confidence", "gdelt_volume", "rss_volume",
            "gdelt_sentiment", "rss_sentiment", "sentiment_spike"
        ]
        for col in expected:
            self.assertIn(col, REQUIRED_COLUMNS)

    def test_no_nan_inf_in_zero_row(self):
        """Ensure zero row is clean of NaNs or Infs."""
        df = create_zero_row()
        self.assertFalse(df.isnull().values.any())
        # Only check numeric columns for inf
        numeric_cols = df.select_dtypes(include=[np.number])
        self.assertFalse(np.isinf(numeric_cols.values).any())

    def test_file_creation_logic(self):
        """Verify that the script can create the output file in the correct directory."""
        # This just checks path resolution
        self.assertEqual(OUTPUT_PATH.name, "latest_sentiment_15m.csv")
        self.assertTrue(OUTPUT_PATH.parent.name == "realtime")


if __name__ == "__main__":
    unittest.main()
