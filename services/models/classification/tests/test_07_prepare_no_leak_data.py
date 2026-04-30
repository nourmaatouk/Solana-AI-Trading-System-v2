import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import importlib

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

mod = importlib.import_module("07_prepare_no_leak_data")
process_data = mod.process_data


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def test_prepare_no_leak_data(tmp_dir):
    input_csv = tmp_dir / "input.csv"
    data_dir = tmp_dir / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Needs at least 32+ rows for sequence building. Let's make 60 rows.
    n_rows = 60

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="15min"),
            "open": np.random.randn(n_rows),
            "close": np.random.randn(n_rows),
            "sentiment_score": np.random.randn(n_rows),  # Sentiment feature
            "positive_ratio": np.abs(np.random.randn(n_rows)),  # Sentiment feature
            "target_direction": np.random.randint(0, 2, size=n_rows),  # Target
            "future_return": np.random.randn(n_rows),  # Leakage
            "t+1": np.random.randn(n_rows),  # Leakage
        }
    )
    
    # Inject NaN and inf values to test cleanup logic
    df.loc[10, "open"] = np.nan
    df.loc[20, "close"] = np.inf
    df.loc[30, "sentiment_score"] = -np.inf
    
    df.to_csv(input_csv, index=False)

    res = process_data(input_csv, data_dir)

    # 1. verify sentiment columns are included
    # 2. verify leakage columns are excluded
    feat_cols = res["feature_columns"]
    assert "sentiment_score" in feat_cols
    assert "positive_ratio" in feat_cols

    assert "future_return" not in feat_cols
    assert "t+1" not in feat_cols
    assert "target_direction" not in feat_cols
    assert "timestamp" not in feat_cols

    # 3. verify output shapes are valid
    # train bounds: n=60, 3 removed -> 57. TRAIN=0.7 (39). seq_len=32. train seqs = 39 - 32 = 7
    X_train = res["X_train"]
    y_train = res["y_train"]
    assert len(X_train) == 7
    assert len(y_train) == 7
    assert X_train.shape[1] == 32  # seq len
    assert X_train.shape[2] == len(feat_cols)

    # 4. no NaN values in saved arrays
    assert not np.isnan(X_train).any()
    assert not np.isinf(X_train).any()

    # Verify files saved
    assert (data_dir / "X_train_no_leak.npy").exists()
    assert (data_dir / "y_train_no_leak.npy").exists()
