import os
import tempfile
import pandas as pd
from src.processing.merge_sentiment_with_market import merge_sentiment_features


def test_merge_sentiment_with_market():
    """Verify left join keeps all market rows and fills missing sentiment values with 0."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        market_file = os.path.join(tmp_dir, "market.csv")
        sentiment_file = os.path.join(tmp_dir, "sentiment.csv")
        out_file = os.path.join(tmp_dir, "merged.csv")
        
        # Fake market data
        market_df = pd.DataFrame([
            {"timestamp": "2026-04-29T10:00:00Z", "close": 100},
            {"timestamp": "2026-04-29T10:15:00Z", "close": 105},
            {"timestamp": "2026-04-29T10:30:00Z", "close": 102},
        ])
        market_df.to_csv(market_file, index=False)
        
        # Fake sentiment data
        sentiment_df = pd.DataFrame([
            {"timestamp": "2026-04-29T10:15:00Z", "sentiment_mean": 0.5, "news_volume": 42},
        ])
        sentiment_df.to_csv(sentiment_file, index=False)
        
        # Run merge
        result_df = merge_sentiment_features(
            market_path=market_file,
            sentiment_path=sentiment_file,
            output_path=out_file
        )
        
        # Verify length 
        assert len(result_df) == 3
        
        # Verify filled matching data
        row_1015 = result_df.iloc[1]
        assert row_1015["close"] == 105
        assert row_1015["sentiment_mean"] == 0.5
        assert row_1015["news_volume"] == 42
        
        # Verify missing data becomes 0
        row_1000 = result_df.iloc[0]
        assert row_1000["close"] == 100
        assert row_1000["sentiment_mean"] == 0.0
        assert row_1000["news_volume"] == 0.0
        
        # Verify file creation
        assert os.path.exists(out_file)
