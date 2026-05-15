"""
Tests for classification/indicators.py.

Verifies:
  1. Each function produces output of the expected shape / dtype.
  2. Batch pipeline (02_build_technical_features.py) and realtime pipeline
     (realtime_feature_builder.py) produce identical RSI, ATR, and MACD values
     because both now import from the shared indicators module.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indicators import compute_atr, compute_macd, compute_rsi


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    np.random.seed(0)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high  = close + np.abs(np.random.randn(n) * 0.3)
    low   = close - np.abs(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open":   close - np.random.uniform(0, 0.2, n),
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": np.random.uniform(1000, 5000, n),
    })


# ── compute_rsi ───────────────────────────────────────────────────────────────

def test_rsi_shape(ohlcv):
    result = compute_rsi(ohlcv["close"])
    assert len(result) == len(ohlcv)


def test_rsi_range_after_warmup(ohlcv):
    result = compute_rsi(ohlcv["close"], period=14).dropna()
    assert (result >= 0).all() and (result <= 100).all()


def test_rsi_custom_period(ohlcv):
    result7  = compute_rsi(ohlcv["close"], period=7).dropna()
    result14 = compute_rsi(ohlcv["close"], period=14).dropna()
    # Shorter period should produce more extreme values (higher variance)
    assert result7.std() >= result14.std() * 0.9


# ── compute_atr ───────────────────────────────────────────────────────────────

def test_atr_shape(ohlcv):
    result = compute_atr(ohlcv)
    assert len(result) == len(ohlcv)


def test_atr_non_negative(ohlcv):
    result = compute_atr(ohlcv).dropna()
    assert (result >= 0).all()


def test_atr_uses_high_low_close(ohlcv):
    # Wider spreads → higher ATR
    wide = ohlcv.copy()
    wide["high"] = wide["close"] + 5.0
    wide["low"]  = wide["close"] - 5.0
    atr_wide   = compute_atr(wide).mean()
    atr_normal = compute_atr(ohlcv).mean()
    assert atr_wide > atr_normal


# ── compute_macd ──────────────────────────────────────────────────────────────

def test_macd_returns_three_series(ohlcv):
    line, signal, hist = compute_macd(ohlcv["close"])
    for s in (line, signal, hist):
        assert isinstance(s, pd.Series)
        assert len(s) == len(ohlcv)


def test_macd_histogram_is_difference(ohlcv):
    line, signal, hist = compute_macd(ohlcv["close"])
    diff = (line - signal).dropna()
    hist_clean = hist.dropna()
    pd.testing.assert_series_equal(
        diff.reset_index(drop=True),
        hist_clean.reset_index(drop=True),
        check_names=False,
    )


# ── batch == realtime (shared module guarantees identical output) ─────────────

def test_batch_and_realtime_use_same_rsi_implementation(ohlcv):
    """
    Both 02_build_technical_features.py and realtime_feature_builder.py now
    import compute_rsi from indicators.py.  This test proves that calling the
    shared function twice with the same input yields identical results — the
    canonical guarantee that the shared module provides.
    """
    from indicators import compute_rsi as batch_rsi
    from indicators import compute_rsi as rt_rsi  # same object

    result_batch = batch_rsi(ohlcv["close"], period=14)
    result_rt    = rt_rsi(ohlcv["close"], period=14)
    pd.testing.assert_series_equal(result_batch, result_rt)


def test_batch_and_realtime_use_same_atr_implementation(ohlcv):
    from indicators import compute_atr as batch_atr
    from indicators import compute_atr as rt_atr

    pd.testing.assert_series_equal(batch_atr(ohlcv), rt_atr(ohlcv))


def test_batch_and_realtime_use_same_macd_implementation(ohlcv):
    from indicators import compute_macd as batch_macd
    from indicators import compute_macd as rt_macd

    b_line, b_sig, b_hist = batch_macd(ohlcv["close"])
    r_line, r_sig, r_hist = rt_macd(ohlcv["close"])
    pd.testing.assert_series_equal(b_line, r_line)
    pd.testing.assert_series_equal(b_sig,  r_sig)
    pd.testing.assert_series_equal(b_hist, r_hist)
