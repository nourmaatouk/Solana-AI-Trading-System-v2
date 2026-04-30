import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add classification directory to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from realtime_merge_features import merge_realtime_data


class TestRealtimeMergeFeatures(unittest.TestCase):
    def setUp(self):
        # Mock tech features (10 rows)
        self.tech_data = pd.DataFrame({
            'timestamp': pd.to_datetime([1714470000000 + i*900000 for i in range(10)], unit='ms', utc=True),
            'close': np.random.uniform(100, 110, 10),
            'rsi_14': np.random.uniform(30, 70, 10)
        })
        
        # Mock sentiment features (1 row matching the last tech row)
        self.sent_data = pd.DataFrame({
            'timestamp': [self.tech_data['timestamp'].iloc[-1]],
            'sentiment_mean': [0.5],
            'news_volume': [10]
        })

    def test_all_market_rows_kept(self):
        """Test that the merge keeps all technical feature rows (Left Join)."""
        df_merged = merge_realtime_data(self.tech_data, self.sent_data)
        self.assertEqual(len(df_merged), len(self.tech_data))

    def test_sentiment_columns_added(self):
        """Test that columns from sentiment data are present in output."""
        df_merged = merge_realtime_data(self.tech_data, self.sent_data)
        self.assertIn('sentiment_mean', df_merged.columns)
        self.assertIn('news_volume', df_merged.columns)

    def test_missing_sentiment_is_zero(self):
        """Test that rows without corresponding sentiment are filled with 0.0."""
        df_merged = merge_realtime_data(self.tech_data, self.sent_data)
        # Only the last row matched, others should be 0.0
        for i in range(len(df_merged) - 1):
            self.assertEqual(df_merged['sentiment_mean'].iloc[i], 0.0)
            self.assertEqual(df_merged['news_volume'].iloc[i], 0.0)

    def test_no_invalid_values(self):
        """Test for NaNs or Infs in merged output."""
        df_merged = merge_realtime_data(self.tech_data, self.sent_data)
        # Timestamp is datetime, others should be numeric
        numeric_cols = df_merged.select_dtypes(include=[np.number])
        self.assertFalse(numeric_cols.isnull().values.any(), "Missing values found")
        self.assertFalse(np.isinf(numeric_cols.values).any(), "Infinite values found")


if __name__ == "__main__":
    unittest.main()
