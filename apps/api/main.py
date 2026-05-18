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
import logging as _logging
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_log = _logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Path configuration ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
REALTIME_DIR = REPO_ROOT / "services" / "models" / "classification" / "data" / "realtime"

_PIPELINE_LOG = REALTIME_DIR / "pipeline_run.log"
_RETRAIN_LOG  = REALTIME_DIR / "retrain_run.log"

_CLASSIFICATION_DIR = REPO_ROOT / "services" / "models" / "classification"
_FUSION_SRC_DIR     = REPO_ROOT / "services" / "models" / "fusion" / "src"
for _p in (_CLASSIFICATION_DIR, _FUSION_SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from config import CONFIDENCE_THRESHOLD, SIGNAL_STALE_MINUTES, SPOOFING_MAX_AGE_MIN, SYMBOL  # noqa: F401
except ImportError:
    _log.warning("classification config import failed — using hardcoded fallback values")
    CONFIDENCE_THRESHOLD  = 0.69
    SIGNAL_STALE_MINUTES  = 20
    SPOOFING_MAX_AGE_MIN  = 30
    SYMBOL                = "SOL/USDT"
try:
    from fusion_aggregator import SPOOFING_SEVERITY_HIGH  # noqa: F401
except ImportError:
    _log.warning("fusion_aggregator import failed — using hardcoded SPOOFING_SEVERITY_HIGH=0.6")
    SPOOFING_SEVERITY_HIGH = 0.6
DASHBOARD_DIST = REPO_ROOT / "apps" / "trading-dashboard" / "frontend" / "dist"

_PATHS = {
    "signal":         REALTIME_DIR / "last_signal.json",
    "fused_decision": REALTIME_DIR / "last_fused_decision.json",
    "rl_signal":      REALTIME_DIR / "last_rl_signal.json",
    "pipeline_run":   REALTIME_DIR / "last_pipeline_run.json",
    "paper_trading":  REALTIME_DIR / "paper_trading_log_fusion.json",
    "orderbook":      REALTIME_DIR / "orderbook_status.json",
    "sentiment":      REALTIME_DIR / "sentiment_status.json",
    "scheduler":      REALTIME_DIR / "scheduler_status.json",
}
HISTORY_PATH = REALTIME_DIR / "paper_trading_history.json"
MODEL_DIR    = REPO_ROOT / "services" / "models" / "models"
REPORTS_DIR  = REPO_ROOT / "services" / "models" / "classification" / "reports"

PREFERENCES_PATH    = REALTIME_DIR / "preferences.json"
SCHEDULER_STOP_FLAG = REALTIME_DIR / "scheduler_stop.flag"
ACCURACY_SUMM_PATH  = REALTIME_DIR / "live_accuracy_summary.json"
ACCURACY_LOG_PATH   = REALTIME_DIR / "live_accuracy_log.json"
DRIFT_STATUS_PATH   = REALTIME_DIR / "drift_status.json"
RETRAIN_LOG_PATH    = MODEL_DIR / "retrain_log.json"

_DEFAULT_PREFERENCES: dict[str, Any] = {
    "asset": SYMBOL,
    "risk_level": "medium",
    "trading_mode": "paper",
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "max_position_size": 1.0,
}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Solana AI Trading System API",
    description="Read-only monitoring API for the realtime pipeline.",
    version="0.1.0",
)

_cors_raw = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:5174,http://localhost:3001",
)
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Optional API-key gate ─────────────────────────────────────────────────────
# Set API_KEY in the environment to require X-API-Key on every request except
# /health and / (so uptime monitors and browsers can still reach those).
_API_KEY = os.environ.get("API_KEY", "").strip()
_AUTH_EXEMPT = {"/", "/health"}

if _API_KEY:
    @app.middleware("http")
    async def _require_api_key(request: Request, call_next):
        if request.url.path not in _AUTH_EXEMPT:
            provided = request.headers.get("X-API-Key", "")
            if provided != _API_KEY:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)
    _log.info("API key authentication enabled")


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


def _tcp_ok(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _parse_host_port(url: str, default_port: int) -> tuple[str, int]:
    try:
        p = urlparse(url)
        return (p.hostname or "localhost"), (p.port or default_port)
    except Exception:
        return "localhost", default_port


# ── Rate-limit state ──────────────────────────────────────────────────────────
_SCHEDULER_RATE_LIMIT_S = int(os.environ.get("SCHEDULER_RATE_LIMIT_S", "60"))
_last_scheduler_trigger: datetime | None = None


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


@app.get("/deep-health", tags=["meta"])
def deep_health() -> dict[str, Any]:
    """
    Full dependency health check.

    Verifies Redis, PostgreSQL, pipeline signal freshness, and scheduler
    status.  Returns HTTP 200 even when checks fail so monitoring tools
    can always parse the JSON body — inspect the `healthy` field instead.
    """
    now = datetime.now(timezone.utc)
    checks: dict[str, Any] = {}

    # Redis
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    r_host, r_port = _parse_host_port(redis_url, 6379)
    checks["redis"] = {"ok": _tcp_ok(r_host, r_port), "host": r_host, "port": r_port}

    # PostgreSQL
    pg_dsn = os.environ.get("TIMESCALE_DSN", "")
    if pg_dsn:
        pg_host, pg_port = _parse_host_port(pg_dsn, 5432)
        checks["postgres"] = {"ok": _tcp_ok(pg_host, pg_port), "host": pg_host, "port": pg_port}
    else:
        checks["postgres"] = {"ok": None, "note": "TIMESCALE_DSN not configured"}

    # Pipeline freshness — last fused signal
    sig_path = _PATHS.get("fused_decision")
    if sig_path and sig_path.exists():
        try:
            sig = json.loads(sig_path.read_text())
            age = _age_minutes(sig)
            stale = age is not None and age > SIGNAL_STALE_MINUTES
            checks["pipeline"] = {"ok": not stale, "signal_age_minutes": age, "stale_threshold_minutes": SIGNAL_STALE_MINUTES}
        except Exception:
            checks["pipeline"] = {"ok": False, "note": "signal file unreadable"}
    else:
        checks["pipeline"] = {"ok": False, "note": "no signal file — run the pipeline first"}

    # Scheduler last run
    sched_path = _PATHS.get("scheduler")
    if sched_path and sched_path.exists():
        try:
            sched = json.loads(sched_path.read_text())
            age = _age_minutes(sched)
            checks["scheduler"] = {"ok": True, "last_run_age_minutes": age}
        except Exception:
            checks["scheduler"] = {"ok": False, "note": "scheduler status unreadable"}
    else:
        checks["scheduler"] = {"ok": False, "note": "no scheduler status yet"}

    # RL signal freshness (optional — ok=None when model not trained yet)
    rl_path = _PATHS.get("rl_signal")
    if rl_path and rl_path.exists():
        try:
            rl = json.loads(rl_path.read_text())
            age = _age_minutes(rl)
            stale = age is not None and age > SIGNAL_STALE_MINUTES
            checks["rl_signal"] = {"ok": not stale, "signal_age_minutes": age, "signal": rl.get("signal")}
        except Exception:
            checks["rl_signal"] = {"ok": False, "note": "rl signal file unreadable"}
    else:
        checks["rl_signal"] = {"ok": None, "note": "no rl signal — run train_ppo.py first"}

    overall = all(v.get("ok") is not False for v in checks.values())
    return {
        "healthy": overall,
        "timestamp_utc": now.isoformat(),
        "checks": checks,
    }


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
    confidence_gate = CONFIDENCE_THRESHOLD
    passes_gate = prob_up >= confidence_gate or prob_down >= confidence_gate

    spoofing_conf = float(fused.get("spoofing_confidence", 0))

    # Extract real sentiment from the fused reasoning string
    # (sentiment_status.json is a daemon health file, not a sentiment result)
    import re as _re
    raw_sent = (fused.get("reasoning") or {}).get("sentiment", "")
    upper_sent = raw_sent.upper()
    sent_label = (
        "POSITIVE" if "POSITIVE" in upper_sent else
        "NEGATIVE" if "NEGATIVE" in upper_sent else
        "BULLISH"  if "BULLISH"  in upper_sent else
        "BEARISH"  if "BEARISH"  in upper_sent else
        "NEUTRAL"
    )
    _score_m = _re.search(r'\(([-\d.]+)\)', raw_sent)
    sent_score = float(_score_m.group(1)) if _score_m else None

    return {
        "action": fused.get("action") or fused.get("signal", "HOLD"),
        "fused_score": fused.get("fused_score"),
        "age_minutes": age_min,
        "is_stale": age_min > SIGNAL_STALE_MINUTES if age_min is not None else None,
        "xgboost": {
            "probability_up": prob_up,
            "probability_down": prob_down,
            "confidence_gate": confidence_gate,
            "passes_gate": passes_gate,
        },
        "sentiment": {
            "label": sent_label,
            "score": sent_score,
            "reasoning": raw_sent,
        },
        "spoofing": {
            "confidence": spoofing_conf,
            "is_suspicious": spoofing_conf >= SPOOFING_SEVERITY_HIGH,
            "reasoning": fused.get("reasoning", {}).get("spoofing"),
        },
        "sentiment_relevance": float(fused.get("sentiment_relevance", 0)),
        "reason": fused.get("reason", ""),
    }


@app.get("/rl-signal", tags=["trading"])
def get_rl_signal() -> dict[str, Any]:
    """Latest PPO reinforcement learning signal."""
    rl_path = _PATHS["rl_signal"]
    if not rl_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No RL signal yet. Train the model first: python services/models/rl/train_ppo.py",
        )
    payload = _read_json(rl_path)
    payload["_age_minutes"] = _age_minutes(payload)
    return payload


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

    return {"status": "no_data", "message": "No paper trading log yet. Run the pipeline first."}


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


@app.post("/scheduler/stop", tags=["system"])
def stop_scheduler() -> dict[str, Any]:
    """
    Pause the pipeline scheduler.

    Writes a stop-flag file that the scheduler daemon checks at the start of
    each sleep cycle.  The current run (if any) completes normally; subsequent
    runs are skipped until the flag is cleared by POST /scheduler/run.
    """
    SCHEDULER_STOP_FLAG.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULER_STOP_FLAG.write_text(datetime.now(timezone.utc).isoformat())
    return {
        "stopped": True,
        "message": "Stop flag set. Scheduler will pause after the current run completes.",
        "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/scheduler/run", tags=["system"])
def trigger_pipeline_run() -> dict[str, Any]:
    """
    Trigger a single pipeline run immediately (background subprocess).

    Returns quickly — the pipeline runs asynchronously.
    Poll GET /signal or GET /status to see the result.
    Rate-limited to one call per 60 seconds.
    """
    global _last_scheduler_trigger
    now = datetime.now(timezone.utc)
    if _last_scheduler_trigger is not None:
        elapsed = (now - _last_scheduler_trigger).total_seconds()
        if elapsed < _SCHEDULER_RATE_LIMIT_S:
            retry_in = int(_SCHEDULER_RATE_LIMIT_S - elapsed)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limited. Pipeline was triggered {int(elapsed)}s ago. Retry in {retry_in}s.",
            )
    _last_scheduler_trigger = now

    # Clear the stop flag so the daemon resumes if it was paused
    if SCHEDULER_STOP_FLAG.exists():
        SCHEDULER_STOP_FLAG.unlink()

    scheduler_script = (
        REPO_ROOT / "services" / "models" / "classification" / "realtime_scheduler.py"
    )
    if not scheduler_script.exists():
        raise HTTPException(status_code=500, detail="Scheduler script not found.")

    REALTIME_DIR.mkdir(parents=True, exist_ok=True)
    _log_fh = open(_PIPELINE_LOG, "w")
    proc = subprocess.Popen(
        [sys.executable, str(scheduler_script), "--once"],
        cwd=str(REPO_ROOT),
        stdout=_log_fh,
        stderr=subprocess.STDOUT,
    )
    _log_fh.close()
    return {
        "status": "triggered",
        "pid": proc.pid,
        "log": str(_PIPELINE_LOG),
        "message": "Pipeline run started in background. "
                   "Poll GET /signal in ~20 seconds for the result.",
        "triggered_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/accuracy", tags=["trading"])
def get_accuracy() -> dict[str, Any]:
    """
    Live signal accuracy statistics derived from paper_trading_history.

    Returns overall accuracy, rolling last-30 and last-60 window accuracy,
    and per-direction (BUY / SELL) breakdowns.
    """
    if not ACCURACY_SUMM_PATH.exists():
        return {
            "status": "no_data",
            "message": "No accuracy data yet. The tracker runs after each pipeline cycle.",
            "n_signals": 0,
        }
    try:
        data = json.loads(ACCURACY_SUMM_PATH.read_text())
        data["status"] = "ok"
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/accuracy/log", tags=["trading"])
def get_accuracy_log(last: int = 100) -> list[dict[str, Any]]:
    """Individual settled signal outcomes (entry price vs outcome price)."""
    if not ACCURACY_LOG_PATH.exists():
        return []
    try:
        entries = json.loads(ACCURACY_LOG_PATH.read_text())
        return entries[-last:] if isinstance(entries, list) else []
    except Exception:
        return []


@app.get("/drift", tags=["system"])
def get_drift() -> dict[str, Any]:
    """
    Model drift status: compares rolling live accuracy against the walk-forward
    baseline. Status is one of: OK / WARNING / DRIFT_DETECTED / INSUFFICIENT_DATA / NO_DATA.
    """
    if not DRIFT_STATUS_PATH.exists():
        return {
            "status": "NO_DATA",
            "detail": "Drift detector has not run yet.",
            "generated_at_utc": None,
        }
    try:
        return json.loads(DRIFT_STATUS_PATH.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/retrain/status", tags=["system"])
def get_retrain_status() -> dict[str, Any]:
    """Last auto-retrain run: timing, AUC before/after, whether model was swapped."""
    if not RETRAIN_LOG_PATH.exists():
        return {
            "status": "never_run",
            "message": "Auto-retrain has not run yet. Scheduled Sundays at 02:00 UTC.",
        }
    try:
        data = json.loads(RETRAIN_LOG_PATH.read_text())
        data["status"] = "ok"
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/retrain/trigger", tags=["system"])
def trigger_retrain() -> dict[str, Any]:
    """
    Manually trigger a model retraining run (background subprocess).
    Use sparingly — each run takes ~5–10 minutes.
    """
    retrain_script = (
        REPO_ROOT / "services" / "models" / "classification" / "auto_retrain.py"
    )
    if not retrain_script.exists():
        raise HTTPException(status_code=500, detail="auto_retrain.py not found.")

    REALTIME_DIR.mkdir(parents=True, exist_ok=True)
    _log_fh = open(_RETRAIN_LOG, "w")
    proc = subprocess.Popen(
        [sys.executable, str(retrain_script)],
        cwd=str(REPO_ROOT),
        stdout=_log_fh,
        stderr=subprocess.STDOUT,
    )
    _log_fh.close()
    return {
        "status": "triggered",
        "pid": proc.pid,
        "log": str(_RETRAIN_LOG),
        "message": "Retraining started in background. Poll GET /retrain/status for result.",
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
