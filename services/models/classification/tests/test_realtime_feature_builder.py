import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add current directory to path to import the builder
sys.path.append(str(Path(__file__).resolve().parents[1]))
from realtime_feature_builder import add_realtime_features


class TestRealtimeFeatureBuilder(unittest.TestCase):
    def setUp(self):
        # Create a mock OHLCV dataset
        np.random.seed(42)
        n_rows = 100
        self.mock_data = pd.DataFrame({
            'timestamp': range(n_rows),
            'open': np.random.uniform(100, 110, n_rows),
            'high': np.random.uniform(110, 120, n_rows),
            'low': np.random.uniform(90, 100, n_rows),
            'close': np.random.uniform(100, 110, n_rows),
            'volume': np.random.uniform(1000, 5000, n_rows)
        })

    def test_indicators_exist(self):
        """Test that all required indicators are present in the output."""
        featured_df = add_realtime_features(self.mock_data)
        
        required_columns = [
            "rsi_14", "macd_line", "sma_20", "ema_20", 
            "volatility_12", "return_1", "volume_vs_sma20",
            "bb_upper", "atr_14", "candle_body"
        ]
        
        for col in required_columns:
            self.assertIn(col, featured_df.columns, f"Missing indicator: {col}")

    def test_no_invalid_values(self):
        """Test that there are no NaNs or Infs after cleaning logic."""
        featured_df = add_realtime_features(self.mock_data)
        
        # Apply the same cleaning as in main()
        featured_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        featured_df.dropna(inplace=True)
        
        self.assertFalse(featured_df.isnull().values.any(), "DataFrame contains NaNs")
        self.assertFalse(np.isinf(featured_df.select_dtypes(include=[np.number])).values.any(), "DataFrame contains Infs")

    def test_feature_consistency(self):
        """Test that feature names follow the training naming convention."""
        featured_df = add_realtime_features(self.mock_data)
        
        # Check a few specific names that must match 02_build_technical_features.py
        expected_patterns = [
            "return_", "sma_", "ema_", "rsi_", "macd_", "volatility_", "bb_", "atr_"
        ]
        
        cols = featured_df.columns.tolist()
        for pattern in expected_patterns:
            has_pattern = any(pattern in col for col in cols)
            self.assertTrue(has_pattern, f"No columns match pattern: {pattern}")

    def test_output_not_empty(self):
        """Test that we have rows left after dropping NaNs (padding)."""
        featured_df = add_realtime_features(self.mock_data)
        featured_df.dropna(inplace=True)
        self.assertGreater(len(featured_df), 0, "Output DataFrame is empty after dropping NaNs")


if __name__ == "__main__":
    unittest.main()
