"""Tests for the API server (no live data required)."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch

# FastAPI test client
try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("fastapi not installed", allow_module_level=True)

from apps.api.main import app, REALTIME_DIR, PREFERENCES_PATH

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "endpoints" in data
    assert "/signal" in data["endpoints"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_signal_returns_404_when_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr("apps.api.main.REALTIME_DIR", tmp_path)
    import apps.api.main as m
    monkeypatch.setattr(m, "_PATHS", {
        k: tmp_path / v.name for k, v in m._PATHS.items()
    })
    r = client.get("/signal")
    assert r.status_code == 404


def test_signal_returns_fused_decision(tmp_path, monkeypatch):
    import apps.api.main as m
    fused = {
        "status": "ok",
        "action": "HOLD",
        "fused_score": -0.24,
        "risk_level": "MEDIUM",
        "generated_at_utc": "2026-05-13T16:01:42+00:00",
    }
    fused_path = tmp_path / "last_fused_decision.json"
    fused_path.write_text(json.dumps(fused))
    monkeypatch.setattr(m, "_PATHS", {**m._PATHS, "fused_decision": fused_path, "signal": tmp_path / "x.json"})
    r = client.get("/signal")
    assert r.status_code == 200
    data = r.json()
    assert data["action"] == "HOLD"
    assert data["_source"] == "fused_decision"
    assert data["_age_minutes"] is not None


def test_status_returns_404_when_no_data(tmp_path, monkeypatch):
    import apps.api.main as m
    monkeypatch.setattr(m, "_PATHS", {**m._PATHS, "pipeline_run": tmp_path / "missing.json"})
    r = client.get("/status")
    assert r.status_code == 404


def test_preferences_default():
    r = client.get("/preferences")
    assert r.status_code == 200
    data = r.json()
    assert "asset" in data
    assert "risk_level" in data
    assert "trading_mode" in data


def test_preferences_update(tmp_path, monkeypatch):
    import apps.api.main as m
    pref_path = tmp_path / "preferences.json"
    monkeypatch.setattr(m, "PREFERENCES_PATH", pref_path)
    r = client.post("/preferences", json={"risk_level": "low", "trading_mode": "paper"})
    assert r.status_code == 200
    data = r.json()
    assert data["risk_level"] == "low"
    assert data["trading_mode"] == "paper"
    assert pref_path.exists()


def test_preferences_ignores_unknown_keys(tmp_path, monkeypatch):
    import apps.api.main as m
    monkeypatch.setattr(m, "PREFERENCES_PATH", tmp_path / "p.json")
    r = client.post("/preferences", json={"unknown_key": "bad", "risk_level": "high"})
    assert r.status_code == 200
    assert "unknown_key" not in r.json()


def test_system_endpoint():
    r = client.get("/system")
    assert r.status_code == 200
    assert "timestamp_utc" in r.json()
