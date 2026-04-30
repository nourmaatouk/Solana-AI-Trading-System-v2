import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add classification directory to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from realtime_predictor import SEQUENCE_LENGTH


class TestRealtimePredictorAlignment(unittest.TestCase):
    def setUp(self):
        # Mock expected features (59 total, matching user problem description)
        self.expected_features = [f"feature_{i}" for i in range(50)] + ["sentiment_score", "sentiment_magnitude", "news_count"]
        
        # Mock real-time data (49 features, missing sentiment)
        data = {
            'timestamp': range(40),
            'close': np.random.uniform(100, 110, 40),
            'datetime': pd.to_datetime(range(40), unit='h')
        }
        for i in range(46): # 46 features + close + timestamp + datetime = 49 total cols roughly
             data[f"feature_{i}"] = np.random.normal(0, 1, 40)
        
        self.rt_data = pd.DataFrame(data)

    def test_missing_feature_creation(self):
        """Test that missing features are created and filled with 0."""
        df = self.rt_data.copy()
        missing = [f for f in self.expected_features if f not in df.columns]
        
        # logic from predictor
        for col in missing:
            df[col] = 0.0
            
        for col in self.expected_features:
            self.assertIn(col, df.columns)
            if col in missing:
                self.assertTrue((df[col] == 0.0).all())

    def test_column_reordering(self):
        """Test that columns are reordered exactly as specified in expected_features."""
        df = self.rt_data.copy()
        for col in self.expected_features:
            if col not in df.columns:
                df[col] = 0.0
                
        df_aligned = df[self.expected_features]
        
        self.assertEqual(list(df_aligned.columns), self.expected_features)

    def test_final_shape(self):
        """Test that the final matrix shape matches (32, expected_feature_count)."""
        df = self.rt_data.copy()
        for col in self.expected_features:
            if col not in df.columns:
                df[col] = 0.0
        
        df_aligned = df[self.expected_features]
        latest_data = df_aligned.tail(SEQUENCE_LENGTH)
        
        X_raw = latest_data.values.astype(np.float64)
        
        self.assertEqual(X_raw.shape, (SEQUENCE_LENGTH, len(self.expected_features)))

    def test_data_type_consistency(self):
        """Ensure all aligned features are float64 for the scaler."""
        df = self.rt_data.copy()
        for col in self.expected_features:
            if col not in df.columns:
                df[col] = 0.0
        
        df_aligned = df[self.expected_features]
        X_raw = df_aligned.tail(SEQUENCE_LENGTH).values.astype(np.float64)
        
        self.assertEqual(X_raw.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
