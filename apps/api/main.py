"""
Solana AI Trading System — API server

Minimal read-only FastAPI backend that exposes the realtime pipeline outputs
as HTTP/JSON endpoints.  No auth, no writes — this is an internal monitoring
API intended to be consumed by the dashboard or CLI tools.

Run with:
  uvicorn apps.api.main:app --reload --port 8000

Or from the repo root:
  python -m uvicorn apps.api.main:app --port 8000
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── Path configuration ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
REALTIME_DIR = REPO_ROOT / "services" / "models" / "classification" / "data" / "realtime"
DASHBOARD_DIST = REPO_ROOT / "apps" / "dashboard" / "dist"

_PATHS = {
    "signal":         REALTIME_DIR / "last_signal.json",
    "fused_decision": REALTIME_DIR / "last_fused_decision.json",
    "pipeline_run":   REALTIME_DIR / "last_pipeline_run.json",
    "paper_trading":  REALTIME_DIR / "paper_trading_log_fusion.json",
    "orderbook":      REALTIME_DIR / "orderbook_status.json",
    "sentiment":      REALTIME_DIR / "sentiment_status.json",
    "scheduler":      REALTIME_DIR / "scheduler_status.json",
}
HISTORY_PATH = REALTIME_DIR / "paper_trading_history.json"
MODEL_DIR    = REPO_ROOT / "services" / "models" / "models"
REPORTS_DIR  = REPO_ROOT / "services" / "models" / "classification" / "reports"

PREFERENCES_PATH = REALTIME_DIR / "preferences.json"

_DEFAULT_PREFERENCES: dict[str, Any] = {
    "asset": "SOL/USDT",
    "risk_level": "medium",
    "trading_mode": "paper",
    "confidence_threshold": 0.69,
    "max_position_size": 1.0,
}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Solana AI Trading System API",
    description="Read-only monitoring API for the realtime pipeline.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Data not found: {path.name}. Run the pipeline first.")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {path.name}: {exc}")


def _age_minutes(payload: dict, key: str = "generated_at_utc") -> float | None:
    raw = payload.get(key) or payload.get("last_run_utc") or payload.get("started_at_utc")
    if raw is None:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)
        return round((datetime.now(timezone.utc) - ts).total_seconds() / 60.0, 2)
    except ValueError:
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "Solana AI Trading System API",
        "version": "0.1.0",
        "endpoints": ["/signal", "/status", "/paper-trading", "/preferences", "/health"],
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Quick liveness check — does not read any data files."""
    return {"status": "ok", "timestamp_utc": datetime.now(timezone.utc).isoformat()}


@app.get("/signal", tags=["trading"])
def get_signal() -> dict[str, Any]:
    """
    Latest fused trading signal from the realtime pipeline.

    Returns the unified multi-model decision (XGBoost + FinBERT + TEN-GNN)
    from last_fused_decision.json.  Falls back to the raw XGBoost signal
    if the fused decision is unavailable.
    """
    fused_path = _PATHS["fused_decision"]
    xgb_path = _PATHS["signal"]

    if fused_path.exists():
        payload = _read_json(fused_path)
        payload["_source"] = "fused_decision"
        payload["_age_minutes"] = _age_minutes(payload)
        return payload

    if xgb_path.exists():
        payload = _read_json(xgb_path)
        payload["_source"] = "xgboost_only"
        payload["_age_minutes"] = _age_minutes(payload)
        return payload

    raise HTTPException(
        status_code=404,
        detail="No signal available. Run: python services/models/classification/run_realtime_pipeline.py",
    )


@app.get("/signal/breakdown", tags=["trading"])
def get_signal_breakdown() -> dict[str, Any]:
    """
    Structured breakdown of the latest fused signal for UI display.

    Returns XGBoost probabilities with gate info, FinBERT sentiment,
    TEN-GNN spoofing confidence, and signal freshness.
    """
    fused_path = _PATHS["fused_decision"]
    sentiment_path = _PATHS["sentiment"]

    if not fused_path.exists():
        raise HTTPException(status_code=404, detail="No fused decision found. Run the pipeline first.")

    fused = _read_json(fused_path)
    sentiment = _read_json(sentiment_path) if sentiment_path.exists() else {}

    age_min = _age_minutes(fused)
    prob_up = float(fused.get("classification_probability_up", 0))
    prob_down = float(fused.get("classification_probability_down", 0))
    confidence_gate = 0.69
    passes_gate = prob_up >= confidence_gate or prob_down >= confidence_gate

    spoofing_conf = float(fused.get("spoofing_confidence", 0))

    return {
        "action": fused.get("action") or fused.get("signal", "HOLD"),
        "fused_score": fused.get("fused_score"),
        "age_minutes": age_min,
        "is_stale": age_min > 20 if age_min is not None else None,
        "xgboost": {
            "probability_up": prob_up,
            "probability_down": prob_down,
            "confidence_gate": confidence_gate,
            "passes_gate": passes_gate,
        },
        "sentiment": {
            "label": sentiment.get("label") or fused.get("risk_level", "NEUTRAL"),
            "score": sentiment.get("score") or sentiment.get("sentiment_score"),
            "reasoning": fused.get("reasoning", {}).get("sentiment"),
        },
        "spoofing": {
            "confidence": spoofing_conf,
            "is_suspicious": spoofing_conf >= 0.5,
            "reasoning": fused.get("reasoning", {}).get("spoofing"),
        },
        "reason": fused.get("reason", ""),
    }


@app.get("/status", tags=["system"])
def get_status() -> dict[str, Any]:
    """
    Last pipeline run summary: steps, timing, overall status.

    Returns the contents of last_pipeline_run.json.
    """
    payload = _read_json(_PATHS["pipeline_run"])
    payload["_age_minutes"] = _age_minutes(payload, key="finished_at_utc")
    return payload


@app.get("/paper-trading", tags=["trading"])
def get_paper_trading() -> dict[str, Any]:
    """
    Latest paper trading metrics: equity, win rate, Sharpe, open position.

    Returns the most recent paper_trading_log_fusion.json entry.  Falls back
    to the XGBoost-only log if the fusion log is absent.
    """
    fusion_log = _PATHS["paper_trading"]
    xgb_log = REALTIME_DIR / "paper_trading_log.json"

    if fusion_log.exists():
        payload = _read_json(fusion_log)
        payload["_source"] = "fusion"
        return payload

    if xgb_log.exists():
        payload = _read_json(xgb_log)
        payload["_source"] = "xgboost_only"
        return payload

    raise HTTPException(
        status_code=404,
        detail="No paper trading log found. Run the pipeline with --paper-trade.",
    )


@app.get("/preferences", tags=["config"])
def get_preferences() -> dict[str, Any]:
    """Current user preferences (asset, risk level, trading mode)."""
    if not PREFERENCES_PATH.exists():
        return _DEFAULT_PREFERENCES.copy()
    return _read_json(PREFERENCES_PATH)


@app.post("/preferences", tags=["config"])
def update_preferences(body: dict[str, Any]) -> dict[str, Any]:
    """
    Update user preferences.

    Accepted keys: asset, risk_level, trading_mode, confidence_threshold,
    max_position_size.  Unknown keys are ignored.
    """
    allowed = set(_DEFAULT_PREFERENCES.keys())
    current = get_preferences()
    for key, value in body.items():
        if key in allowed:
            current[key] = value
    PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFERENCES_PATH.write_text(json.dumps(current, indent=2))
    return current


@app.get("/system", tags=["system"])
def get_system() -> dict[str, Any]:
    """
    Aggregated system health: orderbook collector, sentiment daemon, last signal age.
    """
    result: dict[str, Any] = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}

    for key, path in _PATHS.items():
        if path.exists():
            try:
                data = json.loads(path.read_text())
                result[key] = {
                    "status": data.get("status", "unknown"),
                    "age_minutes": _age_minutes(data),
                }
            except Exception:
                result[key] = {"status": "unreadable"}
        else:
            result[key] = {"status": "missing"}

    return result


@app.get("/history", tags=["trading"])
def get_history(last: int = 100) -> list[dict[str, Any]]:
    """
    Rolling paper trading history (last N entries, default 100).

    Each entry: run_utc, signal, action, fused_score, equity,
    total_return, n_trades, win_rate, sharpe, latest_close.
    """
    if not HISTORY_PATH.exists():
        return []
    try:
        entries = json.loads(HISTORY_PATH.read_text())
        return entries[-last:] if isinstance(entries, list) else []
    except Exception:
        return []


@app.get("/scheduler", tags=["system"])
def get_scheduler() -> dict[str, Any]:
    """Scheduler status: last run, next run, interval, pipeline result."""
    path = _PATHS["scheduler"]
    if not path.exists():
        return {"status": "not_started", "detail": "Run realtime_scheduler.py to start the scheduler."}
    payload = _read_json(path)
    payload["_age_minutes"] = _age_minutes(payload)
    return payload


@app.get("/backtest", tags=["trading"])
def get_backtest() -> dict[str, Any]:
    """Full backtest results: Sharpe, win rate, drawdown, fold summaries, model summary."""
    backtest_path = MODEL_DIR / "backtest_results.json"
    summary_path  = REPORTS_DIR / "model_performance_summary.json"
    result: dict[str, Any] = {}
    if backtest_path.exists():
        try:
            result["backtest"] = json.loads(backtest_path.read_text())
        except Exception:
            result["backtest"] = None
    if summary_path.exists():
        try:
            result["model_summary"] = json.loads(summary_path.read_text())
        except Exception:
            result["model_summary"] = None
    if not result:
        raise HTTPException(status_code=404, detail="No backtest data found. Run 20_walk_forward_validation.py first.")
    return result


@app.get("/walkforward", tags=["trading"])
def get_walkforward() -> dict[str, Any]:
    """Walk-forward validation results: per-fold metrics, aggregate statistics."""
    path = MODEL_DIR / "walk_forward_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="walk_forward_results.json not found. Run 20_walk_forward_validation.py first.")
    return _read_json(path)


@app.post("/scheduler/run", tags=["system"])
def trigger_pipeline_run() -> dict[str, Any]:
    """
    Trigger a single pipeline run immediately (background subprocess).

    Returns quickly — the pipeline runs asynchronously.
    Poll GET /signal or GET /status to see the result.
    """
    scheduler_script = (
        REPO_ROOT / "services" / "models" / "classification" / "realtime_scheduler.py"
    )
    if not scheduler_script.exists():
        raise HTTPException(status_code=500, detail="Scheduler script not found.")

    subprocess.Popen(
        [sys.executable, str(scheduler_script), "--once"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "status": "triggered",
        "message": "Pipeline run started in background. "
                   "Poll GET /signal in ~20 seconds for the result.",
        "triggered_at_utc": datetime.now(timezone.utc).isoformat(),
    }


# ── Dashboard (static files) — must be registered after all API routes ────────
# Served at /app/  (built with Vite base: '/app/')
# If the dist folder does not exist, the dashboard routes are simply skipped.
if DASHBOARD_DIST.exists():
    app.mount(
        "/app/assets",
        StaticFiles(directory=DASHBOARD_DIST / "assets"),
        name="dashboard-assets",
    )

    @app.get("/app", include_in_schema=False)
    @app.get("/app/", include_in_schema=False)
    def dashboard_index() -> FileResponse:
        return FileResponse(DASHBOARD_DIST / "index.html")
