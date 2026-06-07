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

import csv as _csv
import json
import logging as _logging
import os
import re as _re
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

# Import config from the canonical source. If these fail at startup something
# is genuinely broken — fail loudly rather than silently masking the issue with
# hardcoded fallbacks that will drift from config.py.
from config import CONFIDENCE_THRESHOLD, SIGNAL_STALE_MINUTES, SYMBOL
from fusion_aggregator import SPOOFING_SEVERITY_HIGH

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
NEWS_ARTICLES_PATH = REPO_ROOT / "services" / "models" / "nlp" / "data" / "processed" / "processed_articles.csv"
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


# ── Rate-limit state ─────────────────────────────────────────────────────────
# File-backed so the limit works correctly under multi-worker uvicorn
# (each worker would otherwise have its own independent in-memory timestamp).
_SCHEDULER_RATE_LIMIT_S = int(os.environ.get("SCHEDULER_RATE_LIMIT_S", "60"))
_RATE_LIMIT_PATH = REALTIME_DIR / "scheduler_rate_limit.json"


def _rl_get() -> datetime | None:
    """Return the timestamp of the last scheduler trigger (cross-worker safe)."""
    if not _RATE_LIMIT_PATH.exists():
        return None
    try:
        raw = json.loads(_RATE_LIMIT_PATH.read_text()).get("triggered_at_utc")
        return (
            datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)
            if raw else None
        )
    except Exception:
        return None


def _rl_set(now: datetime) -> None:
    try:
        REALTIME_DIR.mkdir(parents=True, exist_ok=True)
        _RATE_LIMIT_PATH.write_text(json.dumps({"triggered_at_utc": now.isoformat()}))
    except Exception:
        pass  # persistence failure is non-fatal; the run proceeds


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

    # Tri-state ok semantics: True = healthy, False = unhealthy, None = not
    # applicable (e.g. Timescale not configured, RL model not trained). Only
    # explicit False degrades the overall verdict.
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

    # event_type / high_impact: prefer the structured fields written by the
    # fusion aggregator; fall back to parsing the reasoning string for older JSON.
    _event_m = _re.search(r'event=(\S+)', raw_sent)
    sent_event_type = fused.get("event_type") or (_event_m.group(1) if _event_m else None)
    sent_high_impact = bool(fused.get("high_impact_event", False))

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
            "event_type": sent_event_type,
            "high_impact": sent_high_impact,
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


@app.get("/news", tags=["sentiment"])
def get_news(limit: int = 20) -> dict[str, Any]:
    """
    Recent news articles with per-article FinBERT sentiment.

    Source: the NLP pipeline's processed_articles.csv (GDELT + RSS feeds).
    Read-only. Returns an empty list (not an error) when the file is absent or
    has no rows, so the dashboard can render a clean empty state.
    """
    limit = max(1, min(limit, 200))
    if not NEWS_ARTICLES_PATH.exists():
        return {"articles": [], "count": 0, "source_file_present": False}

    try:
        with open(NEWS_ARTICLES_PATH, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read news data: {exc}")

    def _num(v: Any) -> float | None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _flag(v: Any) -> bool:
        return str(v).strip().lower() in ("1", "1.0", "true", "yes")

    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    articles = [
        {
            "timestamp":         r.get("timestamp") or None,
            # CSV column `source` holds the feed tag (gdelt/rss); `feed_source`
            # holds the publisher name. Expose them with intuitive field names.
            "feed_source":       (r.get("source") or "").upper() or None,   # GDELT | RSS
            "source":            r.get("feed_source") or None,              # publisher
            "title":             r.get("title") or None,
            "url":               r.get("url") or None,
            "sentiment_label":   (r.get("sentiment_label") or "").upper() or None,
            "sentiment_score":   _num(r.get("sentiment_score")),
            "confidence":        _num(r.get("confidence")),
            "relevance_score":   _num(r.get("relevance_score")),
            "event_type":        r.get("event_type") or None,
            "high_impact_event": _flag(r.get("high_impact_event")),
        }
        for r in rows[:limit]
    ]
    return {"articles": articles, "count": len(articles), "source_file_present": True}


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
    now = datetime.now(timezone.utc)
    last = _rl_get()
    if last is not None:
        elapsed = (now - last).total_seconds()
        if elapsed < _SCHEDULER_RATE_LIMIT_S:
            retry_in = int(_SCHEDULER_RATE_LIMIT_S - elapsed)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limited. Pipeline was triggered {int(elapsed)}s ago. Retry in {retry_in}s.",
            )
    _rl_set(now)

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
    try:
        proc = subprocess.Popen(
            [sys.executable, str(scheduler_script), "--once"],
            cwd=str(REPO_ROOT),
            stdout=_log_fh,
            stderr=subprocess.STDOUT,
        )
    finally:
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
    try:
        proc = subprocess.Popen(
            [sys.executable, str(retrain_script)],
            cwd=str(REPO_ROOT),
            stdout=_log_fh,
            stderr=subprocess.STDOUT,
        )
    finally:
        _log_fh.close()
    return {
        "status": "triggered",
        "pid": proc.pid,
        "log": str(_RETRAIN_LOG),
        "message": "Retraining started in background. Poll GET /retrain/status for result.",
        "triggered_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/correlation", tags=["trading"])
def get_correlation(candles: int = 60) -> dict[str, Any]:
    """
    BTC/SOL correlation data for the Correlation page.

    Returns:
    - btc_features: the 5 co-features currently fed to the XGBoost classifier
    - btc_candles: last `candles` rows of BTC 15m OHLCV
    - sol_candles: last `candles` rows of SOL 15m OHLCV
    - rolling_correlation: Pearson r between BTC and SOL close returns (last `candles` rows)
    """
    import csv as _csv_mod

    btc_path = REALTIME_DIR / "latest_btc_15m.csv"
    sol_path = REALTIME_DIR / "latest_sol_15m.csv"
    feat_path = REALTIME_DIR / "latest_features_with_sentiment.csv"

    def _read_candles(path: Path, n: int) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            with path.open() as f:
                rows = list(_csv_mod.DictReader(f))
            out = []
            for row in rows[-n:]:
                try:
                    out.append({
                        "timestamp": int(row["timestamp"]),
                        "open":   float(row["open"]),
                        "high":   float(row["high"]),
                        "low":    float(row["low"]),
                        "close":  float(row["close"]),
                        "volume": float(row["volume"]),
                    })
                except (ValueError, KeyError):
                    continue
            return out
        except Exception:
            return []

    btc_candles = _read_candles(btc_path, candles)
    sol_candles = _read_candles(sol_path, candles)

    # BTC co-features from the latest feature row
    btc_features: dict[str, Any] = {}
    if feat_path.exists():
        try:
            with feat_path.open() as f:
                rows = list(_csv_mod.DictReader(f))
            if rows:
                last = rows[-1]
                for col in ("btc_return_4", "btc_return_16", "btc_vol_12", "btc_vs_sma20", "btc_rsi_14"):
                    try:
                        btc_features[col] = float(last[col]) if last.get(col) not in (None, "", "nan") else None
                    except (ValueError, KeyError):
                        btc_features[col] = None
                btc_features["feature_timestamp"] = last.get("timestamp")
        except Exception:
            pass

    # Rolling Pearson correlation between BTC and SOL close returns
    rolling_correlation: float | None = None
    try:
        if len(btc_candles) >= 5 and len(sol_candles) >= 5:
            # align on timestamp
            btc_map = {c["timestamp"]: c["close"] for c in btc_candles}
            sol_map = {c["timestamp"]: c["close"] for c in sol_candles}
            common = sorted(set(btc_map) & set(sol_map))
            if len(common) >= 5:
                btc_c = [btc_map[t] for t in common]
                sol_c = [sol_map[t] for t in common]
                # returns
                btc_r = [btc_c[i] / btc_c[i - 1] - 1 for i in range(1, len(btc_c))]
                sol_r = [sol_c[i] / sol_c[i - 1] - 1 for i in range(1, len(sol_c))]
                n = len(btc_r)
                mean_b = sum(btc_r) / n
                mean_s = sum(sol_r) / n
                cov = sum((btc_r[i] - mean_b) * (sol_r[i] - mean_s) for i in range(n)) / n
                std_b = (sum((x - mean_b) ** 2 for x in btc_r) / n) ** 0.5
                std_s = (sum((x - mean_s) ** 2 for x in sol_r) / n) ** 0.5
                if std_b > 0 and std_s > 0:
                    rolling_correlation = round(cov / (std_b * std_s), 4)
    except Exception:
        pass

    return {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "btc_features": btc_features,
        "btc_candles": btc_candles,
        "sol_candles": sol_candles,
        "rolling_correlation": rolling_correlation,
        "candles_requested": candles,
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

    @app.get("/app/{path:path}", include_in_schema=False)
    def dashboard_spa(path: str) -> FileResponse:  # noqa: ARG001
        # SPA catch-all: every deep link (e.g. /app/dashboard, /app/settings)
        # serves index.html so the React router handles the path client-side.
        return FileResponse(DASHBOARD_DIST / "index.html")
