import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))
from realtime_predictor import (  # noqa: E402
    align_latest_features,
    find_missing_features,
    latest_orderbook_unavailable,
    score_to_signal,
    validate_artifact_contract,
    write_signal_json,
)


class FakeScaler:
    def __init__(self, n_features: int):
        self.n_features_in_ = n_features


class FakeModel:
    def __init__(self, n_features: int):
        self.n_features_in_ = n_features


def test_align_latest_features_uses_single_latest_row_in_expected_order():
    expected_features = ["feature_b", "feature_a", "feature_c"]
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0],
            "feature_b": [10.0, 20.0],
            "feature_c": [100.0, 200.0],
            "unused_extra": [999.0, 999.0],
        }
    )

    X_raw, missing = align_latest_features(df, expected_features)

    assert missing == []
    assert X_raw.shape == (1, len(expected_features))
    assert X_raw.dtype == np.float64
    assert X_raw.tolist() == [[20.0, 2.0, 200.0]]


def test_missing_features_are_reported_without_zero_padding():
    expected_features = ["feature_a", "feature_b", "orderbook_imbalance"]
    df = pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]})

    X_raw, missing = align_latest_features(df, expected_features)

    assert X_raw.shape == (0, len(expected_features))
    assert missing == ["orderbook_imbalance"]


def test_find_missing_features_keeps_registry_order():
    df = pd.DataFrame({"b": [1.0]})

    assert find_missing_features(df, ["a", "b", "c"]) == ["a", "c"]


def test_invalid_latest_feature_values_raise():
    df = pd.DataFrame({"feature_a": [1.0, np.inf], "feature_b": [2.0, 3.0]})

    with pytest.raises(ValueError, match="feature_a"):
        align_latest_features(df, ["feature_a", "feature_b"])


def test_artifact_contract_accepts_matching_feature_counts():
    expected_features = ["a", "b", "c"]

    validate_artifact_contract(expected_features, FakeScaler(3), FakeModel(3))


def test_artifact_contract_rejects_scaler_or_model_mismatch():
    expected_features = ["a", "b", "c"]

    with pytest.raises(ValueError, match="scaler expects 4 features"):
        validate_artifact_contract(expected_features, FakeScaler(4), FakeModel(3))

    with pytest.raises(ValueError, match="model expects 4 features"):
        validate_artifact_contract(expected_features, FakeScaler(3), FakeModel(4))


def test_score_to_signal_is_neutral_first():
    assert score_to_signal(0.61, threshold=0.60) == ("UP", "CONFIDENT_UP")
    assert score_to_signal(0.39, threshold=0.60) == ("DOWN", "CONFIDENT_DOWN")
    assert score_to_signal(0.50, threshold=0.60) == ("NEUTRAL", "LOW_CONFIDENCE")


def test_orderbook_availability_flag_blocks_stale_live_orderbook():
    expected_features = ["feature_a", "imbalance_mean"]
    df = pd.DataFrame(
        {
            "feature_a": [1.0],
            "imbalance_mean": [0.0],
            "orderbook_data_available": [0],
        }
    )

    assert latest_orderbook_unavailable(df, expected_features)


def test_orderbook_availability_flag_is_ignored_when_model_has_no_orderbook():
    expected_features = ["feature_a"]
    df = pd.DataFrame({"feature_a": [1.0], "orderbook_data_available": [0]})

    assert not latest_orderbook_unavailable(df, expected_features)


def test_write_signal_json_creates_machine_readable_signal(tmp_path):
    output_path = tmp_path / "last_signal.json"

    payload = write_signal_json(
        signal="DOWN",
        reason="CONFIDENT_DOWN",
        status="ok",
        probability_up=0.3,
        probability_down=0.7,
        latest_timestamp=pd.Timestamp("2026-05-07T10:15:00Z"),
        latest_utc="2026-05-07 10:15:00+00:00",
        latest_close=np.float64(89.08),
        details={"source": "test"},
        output_path=output_path,
    )

    assert output_path.exists()
    assert payload["signal"] == "DOWN"
    assert payload["reason"] == "CONFIDENT_DOWN"
    assert payload["latest_close"] == 89.08
    assert payload["details"] == {"source": "test"}
