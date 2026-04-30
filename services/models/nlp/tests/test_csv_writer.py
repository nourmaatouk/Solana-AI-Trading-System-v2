import pytest
import os
import pandas as pd
import tempfile
from src.storage.csv_writer import save_sentiment_features


def test_save_empty_df_does_not_crash():
    """Ensure bypassing saves safely and gracefully terminates when pushed an empty bound dataset."""
    df = pd.DataFrame()
    
    output_path = save_sentiment_features(df, output_path="dummy.csv")
    
    assert output_path is None
    assert not os.path.exists("dummy.csv")


def test_save_valid_df_creates_file_and_reloads():
    """Validates structural CSV allocations logic writes cleanly and correctly tracks schema."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = os.path.join(tmp_dir, "test_features.csv")
        
        # Construct explicit aggregated DataFrame payload layout mock
        df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime("2026-04-29T10:00:00Z"),
                "sentiment_mean": 0.5,
                "news_volume": 42
            }
        ])
        
        # Execute save trigger explicitly mapping mock testing route 
        result_path = save_sentiment_features(df, output_path=test_file)
        
        # Assess mapped path strings equate
        assert result_path == test_file
        assert os.path.exists(test_file)
        
        # Verify reloading integrity validates successfully seamlessly via standard Pandas 
        loaded_df = pd.read_csv(test_file)
        
        assert not loaded_df.empty
        assert len(loaded_df) == 1
        
        # Check standard columns persist back structurally natively 
        assert "timestamp" in loaded_df.columns
        assert "sentiment_mean" in loaded_df.columns
        assert "news_volume" in loaded_df.columns
        
        # Verify matching logic preserves values properly identically
        assert loaded_df.iloc[0]["news_volume"] == 42
