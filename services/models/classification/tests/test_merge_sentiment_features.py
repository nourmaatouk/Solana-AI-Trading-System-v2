import importlib
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

merge_mod = importlib.import_module("06_merge_sentiment_features")
merge_features = merge_mod.merge_features


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def test_merge_sentiment_features(tmp_dir):
    tech_csv = tmp_dir / "tech.csv"
    sent_csv = tmp_dir / "sent.csv"
    out_csv = tmp_dir / "out.csv"

    # Fake technical dataset with leakage columns
    df_tech = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:15:00", "2026-01-01 00:30:00"],
            "price": [100.0, 101.0, 102.0],
            "future_return": [0.01, 0.01, 0.01],  # Leakage column
            "target": [1, 0, 1],  # Leakage column
            "target_up": [1, 0, 1],  # Leakage column
            "target_direction": [1, 0, 1],  # Leakage column
            "future_return_h": [0.02, 0.02, 0.02],  # Leakage column
        }
    )
    df_tech.to_csv(tech_csv, index=False)

    # Fake sentiment dataset with different naming and leakage column
    df_sent = pd.DataFrame(
        {
            "datetime": ["2026-01-01 00:00:00", "2026-01-01 00:30:00"],
            "sentiment_score": [0.8, -0.5],
            "t+1": [1, 2],  # Leakage column in sentiment
            "Future": [0, 1],  # Leakage column (case mixed)
            "lead": [5, 6],  # Leakage column
        }
    )
    df_sent.to_csv(sent_csv, index=False)

    # Run merge function
    res_df = merge_features(tech_csv, sent_csv, out_csv)

    # verify output file is created
    assert out_csv.exists()

    # verify all technical rows are kept
    assert len(res_df) == 3

    # verify sentiment columns are added
    assert "sentiment_score" in res_df.columns

    # verify missing sentiment values become 0
    # The middle row (00:15:00) doesn't have sentiment
    assert res_df.loc[1, "sentiment_score"] == 0.0

    # verify leakage columns are removed but target_direction is kept
    assert "future_return" not in res_df.columns
    assert "target" not in res_df.columns
    assert "target_up" not in res_df.columns
    assert "target_direction" in res_df.columns
    assert "future_return_h" not in res_df.columns
    assert "t+1" not in res_df.columns
    assert "future" not in res_df.columns
    assert "Future" not in res_df.columns
    assert "lead" not in res_df.columns

    # verify timestamps are UTC datetime and kept as timestamp col
    assert "timestamp" in res_df.columns
    assert str(res_df["timestamp"].dtype) == "datetime64[ns, UTC]"
