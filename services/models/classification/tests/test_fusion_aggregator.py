"""
Tests for realtime_fusion_aggregator.py.

Covers:
  - Signal adapters parse their JSON/CSV inputs correctly.
  - FusionAggregator is called and produces a valid action.
  - main() writes last_fused_decision.json with the expected schema.
  - Graceful degradation when one or more upstream files are missing.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_CLASSIFICATION = Path(__file__).resolve().parents[1]
_MODELS_DIR     = _CLASSIFICATION.parent
_FUSION_SRC     = _MODELS_DIR / "fusion" / "src"

sys.path.insert(0, str(_CLASSIFICATION))
sys.path.insert(0, str(_FUSION_SRC))

import realtime_fusion_aggregator as rfa


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_last_signal(tmp_path: Path, **overrides) -> Path:
    data = {
        "status": "ok",
        "signal": "BUY",
        "reason": "HIGH_CONFIDENCE",
        "probability_up": 0.72,
        "probability_down": 0.28,
        "latest_timestamp": "2026-01-01T12:00:00+00:00",
        "latest_close": 100.0,
        "horizon": 4,
        "confidence_threshold": 0.69,
    }
    data.update(overrides)
    p = tmp_path / "last_signal.json"
    p.write_text(json.dumps(data))
    return p


def _make_spoofing_score(tmp_path: Path, **overrides) -> Path:
    data = {
        "status": "ok",
        "generated_at_utc": "2026-01-01T12:00:00+00:00",
        "n_events": 10,
        "branching_ratio": 0.12,
        "is_suspicious": False,
        "spoofing_probability": 0.05,
        "message": "normal activity",
    }
    data.update(overrides)
    p = tmp_path / "spoofing_score.json"
    p.write_text(json.dumps(data))
    return p


def _make_sentiment_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame([{
        "timestamp": "2026-01-01T12:00:00+00:00",
        "sentiment_mean": 0.3,
        "relevance_weighted_sentiment": 0.25,
        "high_impact_event": 0,
        "max_confidence": 0.7,
        "news_volume": 5,
        "positive_ratio": 0.6,
        "negative_ratio": 0.1,
        "neutral_ratio": 0.3,
    }])
    p = tmp_path / "latest_features_with_sentiment.csv"
    df.to_csv(p, index=False)
    return p


# ── _load_classification_signal ───────────────────────────────────────────────

class TestLoadClassificationSignal:
    def test_parses_buy_signal(self, tmp_path, monkeypatch):
        _make_last_signal(tmp_path, signal="BUY", probability_up=0.72)
        monkeypatch.setattr(rfa, "SIGNAL_PATH", tmp_path / "last_signal.json")
        sig = rfa._load_classification_signal()
        assert sig is not None
        assert sig.signal == "BUY"
        assert abs(sig.prediction - 0.72) < 1e-6
        assert sig.confidence >= 0.5

    def test_parses_neutral_as_hold(self, tmp_path, monkeypatch):
        _make_last_signal(tmp_path, signal="NEUTRAL", probability_up=0.51)
        monkeypatch.setattr(rfa, "SIGNAL_PATH", tmp_path / "last_signal.json")
        sig = rfa._load_classification_signal()
        assert sig is not None
        assert sig.signal == "HOLD"

    def test_returns_none_on_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rfa, "SIGNAL_PATH", tmp_path / "nonexistent.json")
        assert rfa._load_classification_signal() is None

    def test_returns_none_when_status_not_ok(self, tmp_path, monkeypatch):
        _make_last_signal(tmp_path, status="error")
        monkeypatch.setattr(rfa, "SIGNAL_PATH", tmp_path / "last_signal.json")
        assert rfa._load_classification_signal() is None


# ── _load_spoofing_signal ─────────────────────────────────────────────────────

class TestLoadSpoofingSignal:
    def test_parses_low_confidence(self, tmp_path, monkeypatch):
        _make_spoofing_score(tmp_path, spoofing_probability=0.05, is_suspicious=False)
        monkeypatch.setattr(rfa, "SPOOFING_PATH", tmp_path / "spoofing_score.json")
        sig = rfa._load_spoofing_signal()
        assert sig is not None
        assert sig.severity == "low"
        assert abs(sig.confidence - 0.05) < 1e-6

    def test_parses_high_confidence_as_high_severity(self, tmp_path, monkeypatch):
        _make_spoofing_score(tmp_path, spoofing_probability=0.75, is_suspicious=True)
        monkeypatch.setattr(rfa, "SPOOFING_PATH", tmp_path / "spoofing_score.json")
        sig = rfa._load_spoofing_signal()
        assert sig is not None
        assert sig.severity == "high"

    def test_returns_none_on_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rfa, "SPOOFING_PATH", tmp_path / "nonexistent.json")
        assert rfa._load_spoofing_signal() is None


# ── _load_sentiment_signal ────────────────────────────────────────────────────

class TestLoadSentimentSignal:
    def test_parses_positive_sentiment(self, tmp_path, monkeypatch):
        _make_sentiment_csv(tmp_path)
        monkeypatch.setattr(rfa, "SENTIMENT_FEAT_PATH", tmp_path / "latest_features_with_sentiment.csv")
        sig = rfa._load_sentiment_signal()
        assert sig is not None
        assert sig.sentiment_label == "POSITIVE"
        assert sig.sentiment_score > 0

    def test_negative_sentiment_label(self, tmp_path, monkeypatch):
        df = pd.DataFrame([{
            "timestamp": "2026-01-01T12:00:00+00:00",
            "relevance_weighted_sentiment": -0.5,
            "high_impact_event": 1,
            "max_confidence": 0.9,
            "news_volume": 3,
        }])
        p = tmp_path / "latest_features_with_sentiment.csv"
        df.to_csv(p, index=False)
        monkeypatch.setattr(rfa, "SENTIMENT_FEAT_PATH", p)
        sig = rfa._load_sentiment_signal()
        assert sig is not None
        assert sig.sentiment_label == "NEGATIVE"
        assert sig.high_impact_event is True

    def test_returns_none_on_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rfa, "SENTIMENT_FEAT_PATH", tmp_path / "nonexistent.csv")
        assert rfa._load_sentiment_signal() is None


# ── main() integration ────────────────────────────────────────────────────────

class TestFusionAggregatorMain:
    def _patch_paths(self, tmp_path, monkeypatch):
        _make_last_signal(tmp_path)
        _make_spoofing_score(tmp_path)
        _make_sentiment_csv(tmp_path)
        out = tmp_path / "last_fused_decision.json"
        monkeypatch.setattr(rfa, "SIGNAL_PATH",         tmp_path / "last_signal.json")
        monkeypatch.setattr(rfa, "SPOOFING_PATH",        tmp_path / "spoofing_score.json")
        monkeypatch.setattr(rfa, "SENTIMENT_FEAT_PATH",  tmp_path / "latest_features_with_sentiment.csv")
        monkeypatch.setattr(rfa, "OUTPUT_PATH",          out)
        return out

    def test_writes_output_file(self, tmp_path, monkeypatch):
        out = self._patch_paths(tmp_path, monkeypatch)
        rc  = rfa.main()
        assert rc == 0
        assert out.exists()

    def test_output_schema(self, tmp_path, monkeypatch):
        out = self._patch_paths(tmp_path, monkeypatch)
        rfa.main()
        decision = json.loads(out.read_text())
        required = {"status", "generated_at_utc", "action", "fused_score", "risk_level"}
        assert required <= decision.keys()
        assert decision["action"] in ("BUY", "SELL", "HOLD")
        assert -1.0 <= decision["fused_score"] <= 1.0
        assert decision["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_graceful_when_spoofing_missing(self, tmp_path, monkeypatch):
        _make_last_signal(tmp_path)
        _make_sentiment_csv(tmp_path)
        out = tmp_path / "last_fused_decision.json"
        monkeypatch.setattr(rfa, "SIGNAL_PATH",         tmp_path / "last_signal.json")
        monkeypatch.setattr(rfa, "SPOOFING_PATH",        tmp_path / "nonexistent.json")
        monkeypatch.setattr(rfa, "SENTIMENT_FEAT_PATH",  tmp_path / "latest_features_with_sentiment.csv")
        monkeypatch.setattr(rfa, "OUTPUT_PATH",          out)
        rc = rfa.main()
        assert rc == 0
        decision = json.loads(out.read_text())
        assert decision["action"] in ("BUY", "SELL", "HOLD")

    def test_returns_1_when_all_missing(self, tmp_path, monkeypatch):
        out = tmp_path / "last_fused_decision.json"
        monkeypatch.setattr(rfa, "SIGNAL_PATH",         tmp_path / "a.json")
        monkeypatch.setattr(rfa, "SPOOFING_PATH",        tmp_path / "b.json")
        monkeypatch.setattr(rfa, "SENTIMENT_FEAT_PATH",  tmp_path / "c.csv")
        monkeypatch.setattr(rfa, "OUTPUT_PATH",          out)
        rc = rfa.main()
        assert rc == 1
        # Even on failure a fallback JSON is written
        assert out.exists()
        decision = json.loads(out.read_text())
        assert decision["status"] == "no_signals"
