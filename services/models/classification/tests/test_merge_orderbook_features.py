"""
test_merge_orderbook_features.py

Tests for 06b_merge_orderbook_features.py:
  1. All market rows are kept after merge
  2. Orderbook columns are added
  3. Missing orderbook values become 0
  4. No NaN/inf values remain
"""

import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

merge_mod = importlib.import_module("06b_merge_orderbook_features")
merge_orderbook = merge_mod.merge_orderbook
resample_orderbook_15m = merge_mod.resample_orderbook_15m


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _make_market_csv(path: Path) -> pd.DataFrame:
    """Create a small market+sentiment CSV with 4 rows at 15-min intervals."""
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-04-30 15:00:00+00:00",
                "2026-04-30 15:15:00+00:00",
                "2026-04-30 15:30:00+00:00",
                "2026-04-30 15:45:00+00:00",
            ],
            "close": [83.0, 83.5, 84.0, 83.8],
            "volume": [1000, 1100, 1200, 1050],
            "rsi_14": [55.0, 60.0, 65.0, 58.0],
            "sentiment_score": [0.1, 0.0, -0.3, 0.2],
            "target_direction": [1, 0, 1, 0],
            "future_return_h": [0.01, -0.01, 0.02, -0.005],  # leakage column
        }
    )
    df.to_csv(path, index=False)
    return df


def _make_orderbook_csv(path: Path) -> pd.DataFrame:
    """Create orderbook data with multiple sub-minute rows.
    Covers two 15-min buckets (15:00 and 15:15) but NOT 15:30 or 15:45.
    """
    df = pd.DataFrame(
        {
            "timestamp": [
                # Bucket 15:00
                "2026-04-30 15:00:00",
                "2026-04-30 15:01:00",
                "2026-04-30 15:05:00",
                "2026-04-30 15:10:00",
                # Bucket 15:15
                "2026-04-30 15:15:00",
                "2026-04-30 15:20:00",
            ],
            "imbalance_mean": [0.1, 0.2, 0.3, 0.4, -0.1, -0.2],
            "imbalance_std": [0.4, 0.5, 0.45, 0.48, 0.5, 0.42],
            "spread_mean": [0.1, 0.2, 0.15, 0.12, 0.3, 0.25],
            "mid_price": [83.0, 83.1, 83.2, 83.15, 83.5, 83.6],
            "bid_volume_mean": [5000, 4800, 5100, 4900, 4700, 4600],
            "ask_volume_mean": [4500, 4600, 4400, 4700, 5000, 4800],
            "pressure_indicator": [0, 0, 1, 0, 0, 1],
        }
    )
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------
# Test 1: All market rows are kept
# ---------------------------------------------------------
def test_all_market_rows_kept(tmp_dir):
    market_csv = tmp_dir / "market.csv"
    ob_csv = tmp_dir / "ob.csv"
    out_csv = tmp_dir / "out.csv"

    _make_market_csv(market_csv)
    _make_orderbook_csv(ob_csv)

    result = merge_orderbook(market_csv, ob_csv, out_csv)

    assert len(result) == 4, f"Expected 4 rows, got {len(result)}"
    assert out_csv.exists()


# ---------------------------------------------------------
# Test 2: Orderbook columns are added
# ---------------------------------------------------------
def test_orderbook_columns_added(tmp_dir):
    market_csv = tmp_dir / "market.csv"
    ob_csv = tmp_dir / "ob.csv"
    out_csv = tmp_dir / "out.csv"

    _make_market_csv(market_csv)
    _make_orderbook_csv(ob_csv)

    result = merge_orderbook(market_csv, ob_csv, out_csv)

    expected_ob_cols = [
        "imbalance_mean",
        "imbalance_std",
        "spread_mean",
        "mid_price",
        "bid_volume_mean",
        "ask_volume_mean",
        "pressure_indicator",
    ]
    for col in expected_ob_cols:
        assert col in result.columns, f"Missing orderbook column: {col}"


# ---------------------------------------------------------
# Test 3: Missing orderbook values become 0
# ---------------------------------------------------------
def test_missing_orderbook_filled_zero(tmp_dir):
    market_csv = tmp_dir / "market.csv"
    ob_csv = tmp_dir / "ob.csv"
    out_csv = tmp_dir / "out.csv"

    _make_market_csv(market_csv)
    _make_orderbook_csv(ob_csv)

    result = merge_orderbook(market_csv, ob_csv, out_csv)

    # Rows at 15:30 and 15:45 have no orderbook data → should be 0
    ob_cols = [
        "imbalance_mean",
        "imbalance_std",
        "spread_mean",
        "mid_price",
        "bid_volume_mean",
        "ask_volume_mean",
        "pressure_indicator",
    ]
    ts = pd.to_datetime(result["timestamp"], utc=True)
    mask_no_ob = ts.isin(
        pd.to_datetime(
            ["2026-04-30 15:30:00+00:00", "2026-04-30 15:45:00+00:00"], utc=True,
        )
    )
    for col in ob_cols:
        vals = result.loc[mask_no_ob, col].values
        assert (vals == 0).all(), (
            f"Column {col} should be 0 for missing orderbook rows, got {vals}"
        )


# ---------------------------------------------------------
# Test 4: No NaN or inf values remain
# ---------------------------------------------------------
def test_no_nan_inf(tmp_dir):
    market_csv = tmp_dir / "market.csv"
    ob_csv = tmp_dir / "ob.csv"
    out_csv = tmp_dir / "out.csv"

    df_market = _make_market_csv(market_csv)
    _make_orderbook_csv(ob_csv)

    result = merge_orderbook(market_csv, ob_csv, out_csv)

    # Exclude timestamp column (non-numeric)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    assert not result[numeric_cols].isna().any().any(), "NaN values found"
    assert not np.isinf(result[numeric_cols].values).any(), "Inf values found"


# ---------------------------------------------------------
# Test 5: Leakage columns are removed
# ---------------------------------------------------------
def test_leakage_columns_removed(tmp_dir):
    market_csv = tmp_dir / "market.csv"
    ob_csv = tmp_dir / "ob.csv"
    out_csv = tmp_dir / "out.csv"

    _make_market_csv(market_csv)
    _make_orderbook_csv(ob_csv)

    result = merge_orderbook(market_csv, ob_csv, out_csv)

    assert "future_return_h" not in result.columns
    assert "target_direction" in result.columns  # Must be preserved


# ---------------------------------------------------------
# Test 6: Resample orderbook aggregation is correct
# ---------------------------------------------------------
def test_resample_orderbook_aggregation():
    df_ob = pd.DataFrame(
        {
            "timestamp": [
                "2026-04-30 15:00:00",
                "2026-04-30 15:01:00",
                "2026-04-30 15:14:00",
            ],
            "imbalance_mean": [0.1, 0.2, 0.3],
            "spread_mean": [1.0, 2.0, 3.0],
        }
    )
    resampled = resample_orderbook_15m(df_ob)

    # All 3 rows fall into the 15:00 bucket
    assert len(resampled) == 1
    assert float(resampled.iloc[0]["imbalance_mean"]) == pytest.approx(0.2, abs=1e-9)
    assert float(resampled.iloc[0]["spread_mean"]) == pytest.approx(2.0, abs=1e-9)
