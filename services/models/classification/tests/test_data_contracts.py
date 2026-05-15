"""
Data quality contract tests.

These tests catch silent data drift early — before a bad CSV reaches the
model and produces garbage signals.  They validate the shape, content, and
freshness of every file that the realtime pipeline reads or writes.

Run with:
  python -m pytest services/models/classification/tests/test_data_contracts.py -v
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE / "data" / "processed"
REALTIME_DIR = BASE / "data" / "realtime"
FEATURE_COLS_PATH = PROCESSED_DIR / "feature_columns.json"

EXPECTED_FEATURE_COUNT = 47
EXPECTED_ORDERBOOK_COLS = [
    "imbalance_mean", "imbalance_std", "spread_mean", "mid_price",
    "bid_volume_mean", "ask_volume_mean", "pressure_indicator",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load_csv(path: Path) -> pd.DataFrame:
    assert path.exists(), f"File missing: {path}"
    df = pd.read_csv(path)
    assert not df.empty, f"Empty DataFrame: {path}"
    return df


def _assert_no_nan_inf(df: pd.DataFrame, label: str) -> None:
    numeric = df.select_dtypes(include=[np.number])
    nan_cols = numeric.columns[numeric.isnull().any()].tolist()
    inf_cols = numeric.columns[np.isinf(numeric).any()].tolist()
    assert not nan_cols, f"{label}: NaN in columns {nan_cols}"
    assert not inf_cols, f"{label}: Inf in columns {inf_cols}"


def _assert_monotonic_timestamps(df: pd.DataFrame, col: str, label: str) -> None:
    ts = pd.to_datetime(df[col], utc=True, errors="coerce")
    assert ts.notna().all(), f"{label}: non-parseable timestamps in {col}"
    assert ts.is_monotonic_increasing, f"{label}: timestamps in {col} are not monotonically increasing"


def _load_feature_columns_or_skip() -> list[str]:
    if not FEATURE_COLS_PATH.exists():
        pytest.skip(
            "feature_columns.json is a generated training artifact; "
            "run 07_prepare_no_leak_data.py before enforcing this contract."
        )
    return json.loads(FEATURE_COLS_PATH.read_text())


# ── feature_columns.json ──────────────────────────────────────────────────────
def test_feature_columns_exists():
    if not FEATURE_COLS_PATH.exists():
        pytest.skip("feature_columns.json is not present in this checkout — run 07_prepare_no_leak_data.py")


def test_feature_columns_count():
    cols = _load_feature_columns_or_skip()
    assert len(cols) == EXPECTED_FEATURE_COUNT, (
        f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(cols)}. "
        "Did a feature column get added/removed? Re-run 07 and 08."
    )


def test_feature_columns_no_duplicates():
    cols = _load_feature_columns_or_skip()
    assert len(cols) == len(set(cols)), "feature_columns.json contains duplicate column names"


def test_feature_columns_no_leakage():
    leakage_substrings = ["future_return", "target", "lead", "future", "t+1"]
    cols = _load_feature_columns_or_skip()
    bad = [c for c in cols if any(sub in c for sub in leakage_substrings)]
    assert not bad, f"Leakage columns detected: {bad}"


# ── realtime feature CSV ──────────────────────────────────────────────────────
@pytest.mark.skipif(
    not (REALTIME_DIR / "latest_sol_15m_features.csv").exists(),
    reason="realtime features CSV not present (pipeline not run yet)",
)
def test_realtime_features_shape():
    """
    latest_sol_15m_features.csv is an intermediate file produced by
    realtime_feature_builder.py *before* the orderbook merge step.
    It should contain all trained columns except the 7 orderbook columns
    (those are added by realtime_merge_features.py).
    The merged file latest_features_with_sentiment.csv must contain all 47.
    """
    df = _load_csv(REALTIME_DIR / "latest_sol_15m_features.csv")
    all_cols = _load_feature_columns_or_skip()
    # Pre-merge file: check non-orderbook columns only
    pre_merge_cols = [c for c in all_cols if c not in EXPECTED_ORDERBOOK_COLS]
    missing = [c for c in pre_merge_cols if c not in df.columns]
    assert not missing, (
        f"Realtime features (pre-merge) missing expected non-orderbook columns: {missing}"
    )


@pytest.mark.skipif(
    not (REALTIME_DIR / "latest_features_with_sentiment.csv").exists(),
    reason="merged feature CSV not present (pipeline not run yet)",
)
def test_merged_features_no_nan_inf():
    df = _load_csv(REALTIME_DIR / "latest_features_with_sentiment.csv")
    _assert_no_nan_inf(df, "latest_features_with_sentiment.csv")


@pytest.mark.skipif(
    not (REALTIME_DIR / "latest_features_with_sentiment.csv").exists(),
    reason="merged feature CSV not present (pipeline not run yet)",
)
def test_merged_features_has_all_expected_feature_cols():
    df = _load_csv(REALTIME_DIR / "latest_features_with_sentiment.csv")
    cols = _load_feature_columns_or_skip()
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Merged feature CSV is missing trained columns: {missing}"


@pytest.mark.skipif(
    not (REALTIME_DIR / "latest_features_with_sentiment.csv").exists(),
    reason="merged feature CSV not present (pipeline not run yet)",
)
def test_merged_features_has_orderbook_cols():
    df = _load_csv(REALTIME_DIR / "latest_features_with_sentiment.csv")
    missing = [c for c in EXPECTED_ORDERBOOK_COLS if c not in df.columns]
    assert not missing, f"Merged features missing orderbook columns: {missing}"


# ── realtime signal JSON ──────────────────────────────────────────────────────
@pytest.mark.skipif(
    not (REALTIME_DIR / "last_signal.json").exists(),
    reason="last_signal.json not present (pipeline not run yet)",
)
def test_signal_json_schema():
    data = json.loads((REALTIME_DIR / "last_signal.json").read_text())
    assert data.get("status") == "ok", f"Signal status is not 'ok': {data.get('status')}"
    assert "signal" in data, "last_signal.json missing 'signal' key"
    assert data["signal"] in ("UP", "DOWN", "NEUTRAL"), f"Unexpected signal value: {data['signal']}"
    assert "probability_up" in data
    p = float(data["probability_up"])
    assert 0.0 <= p <= 1.0, f"probability_up out of [0, 1]: {p}"


@pytest.mark.skipif(
    not (REALTIME_DIR / "last_fused_decision.json").exists(),
    reason="last_fused_decision.json not present (pipeline not run yet)",
)
def test_fused_decision_json_schema():
    data = json.loads((REALTIME_DIR / "last_fused_decision.json").read_text())
    assert data.get("status") == "ok", f"Fused decision status not 'ok': {data.get('status')}"
    assert data.get("action") in ("BUY", "SELL", "HOLD"), f"Unexpected action: {data.get('action')}"
    score = data.get("fused_score")
    assert score is not None, "fused_decision missing fused_score"
    assert -1.0 <= float(score) <= 1.0, f"fused_score out of [-1, 1]: {score}"
    assert data.get("risk_level") in ("LOW", "MEDIUM", "HIGH"), f"Bad risk_level: {data.get('risk_level')}"


# ── signal freshness ───────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not (REALTIME_DIR / "last_fused_decision.json").exists(),
    reason="last_fused_decision.json not present",
)
def test_fused_decision_not_ancient():
    """Warn if the fused decision is more than 24 hours old."""
    data = json.loads((REALTIME_DIR / "last_fused_decision.json").read_text())
    raw = data.get("generated_at_utc") or data.get("signal_timestamp")
    if raw is None:
        pytest.skip("No timestamp in fused decision — cannot check freshness")
    ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)
    age = datetime.now(timezone.utc) - ts
    assert age < timedelta(hours=24), (
        f"Fused decision is {age} old — pipeline may not be running. "
        "Start the 15-min cron or run the pipeline manually."
    )


# ── orderbook_status.json ─────────────────────────────────────────────────────
@pytest.mark.skipif(
    not (REALTIME_DIR / "orderbook_status.json").exists(),
    reason="orderbook_status.json not present (orderbook collector not running)",
)
def test_orderbook_status_schema():
    data = json.loads((REALTIME_DIR / "orderbook_status.json").read_text())
    assert "status" in data, "orderbook_status.json missing 'status'"
    assert "connected" in data, "orderbook_status.json missing 'connected'"


# ── processed NPY splits ──────────────────────────────────────────────────────
@pytest.mark.skipif(
    not (PROCESSED_DIR / "X_train_h4.npy").exists(),
    reason="training splits not present (run 07_prepare_no_leak_data.py first)",
)
def test_training_splits_shape_consistent():
    X_train = np.load(PROCESSED_DIR / "X_train_h4.npy")
    y_train = np.load(PROCESSED_DIR / "y_train_h4.npy")
    X_val = np.load(PROCESSED_DIR / "X_val_h4.npy")
    y_val = np.load(PROCESSED_DIR / "y_val_h4.npy")
    X_test = np.load(PROCESSED_DIR / "X_test_h4.npy")
    y_test = np.load(PROCESSED_DIR / "y_test_h4.npy")

    assert X_train.shape[1] == EXPECTED_FEATURE_COUNT, (
        f"X_train has {X_train.shape[1]} cols, expected {EXPECTED_FEATURE_COUNT}"
    )
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train row mismatch"
    assert X_val.shape[0] == y_val.shape[0], "X_val and y_val row mismatch"
    assert X_test.shape[0] == y_test.shape[0], "X_test and y_test row mismatch"
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature count differs across splits"


@pytest.mark.skipif(
    not (PROCESSED_DIR / "X_train_h4.npy").exists(),
    reason="training splits not present",
)
def test_training_splits_no_nan_inf():
    for split in ("train", "val", "test"):
        X = np.load(PROCESSED_DIR / f"X_{split}_h4.npy")
        assert not np.isnan(X).any(), f"NaN in X_{split}_h4.npy"
        assert not np.isinf(X).any(), f"Inf in X_{split}_h4.npy"


@pytest.mark.skipif(
    not (PROCESSED_DIR / "y_train_h4.npy").exists(),
    reason="training splits not present",
)
def test_labels_are_binary():
    for split in ("train", "val", "test"):
        y = np.load(PROCESSED_DIR / f"y_{split}_h4.npy")
        unique = set(y.tolist())
        assert unique <= {0, 1}, f"y_{split}_h4 contains non-binary values: {unique}"
