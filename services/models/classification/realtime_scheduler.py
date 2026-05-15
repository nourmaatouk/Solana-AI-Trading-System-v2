#!/usr/bin/env python3
"""
Realtime Pipeline Scheduler

Runs the full pipeline on a fixed interval (default: every 15 minutes).
Also triggers the sentiment daemon when its cache is older than
SENTIMENT_MAX_AGE_MINUTES (default: 30).

Usage:
  python realtime_scheduler.py                    # daemon, 15-min interval
  python realtime_scheduler.py --interval 300     # 5-min interval
  python realtime_scheduler.py --once             # single run then exit (cron mode)

Writes scheduler_status.json next to the other realtime JSON files so the
API's /system endpoint can report its health.
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent
REALTIME_DIR = BASE_DIR / "data" / "realtime"
STATUS_PATH  = REALTIME_DIR / "scheduler_status.json"
SENTIMENT_STATUS_PATH = REALTIME_DIR / "sentiment_status.json"

DEFAULT_INTERVAL_SECONDS  = 900   # 15 minutes
DEFAULT_PIPELINE_TIMEOUT  = 120   # seconds per pipeline run
SENTIMENT_MAX_AGE_MINUTES = 30

_shutdown = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_minutes(path: Path) -> float | None:
    try:
        data = json.loads(path.read_text())
        raw = (data.get("last_run_utc") or data.get("generated_at_utc")
               or data.get("started_at_utc"))
        if not raw:
            return None
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds() / 60
    except Exception:
        return None


def _write_status(payload: dict) -> None:
    REALTIME_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2))


# ── Sub-tasks ─────────────────────────────────────────────────────────────────

def _maybe_refresh_sentiment(timeout: int = 180) -> dict:
    """Run the sentiment daemon only when its cache is stale."""
    age = _age_minutes(SENTIMENT_STATUS_PATH) if SENTIMENT_STATUS_PATH.exists() else None
    if age is not None and age < SENTIMENT_MAX_AGE_MINUTES:
        return {"triggered": False, "reason": f"fresh ({age:.1f} min old)"}

    label = f"{age:.1f} min old" if age is not None else "missing"
    print(f"  [scheduler] Sentiment cache {label} — refreshing…")
    try:
        proc = subprocess.run(
            [sys.executable,
             str(BASE_DIR / "realtime_sentiment_daemon.py"),
             "--once"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = proc.returncode == 0
        return {"triggered": True, "status": "ok" if ok else "error",
                "returncode": proc.returncode}
    except subprocess.TimeoutExpired:
        return {"triggered": True, "status": "timeout", "returncode": 124}
    except Exception as exc:
        return {"triggered": True, "status": "error", "error": str(exc)}


def _run_pipeline(timeout: int = DEFAULT_PIPELINE_TIMEOUT) -> dict:
    """Run the realtime pipeline once."""
    started = _now()
    print(f"  [scheduler] Running pipeline (timeout={timeout}s)…")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(BASE_DIR / "run_realtime_pipeline.py"),
                "--use-sentiment-daemon",
                "--fusion",
                "--fusion-aggregate",
                "--paper-trade",
                "--timeout", str(timeout),
            ],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
        )
        status = "ok" if proc.returncode == 0 else "degraded"
        if proc.stdout:
            for line in proc.stdout.strip().split("\n")[-5:]:
                print(f"    {line}")
        return {"status": status, "returncode": proc.returncode,
                "started_at_utc": started, "finished_at_utc": _now()}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "returncode": 124,
                "started_at_utc": started, "finished_at_utc": _now()}
    except Exception as exc:
        return {"status": "error", "returncode": -1, "error": str(exc),
                "started_at_utc": started, "finished_at_utc": _now()}


# ── Public API ────────────────────────────────────────────────────────────────

def run_once(pipeline_timeout: int = DEFAULT_PIPELINE_TIMEOUT) -> dict:
    """One full scheduler cycle: sentiment refresh (if needed) + pipeline."""
    sentiment = _maybe_refresh_sentiment()
    pipeline  = _run_pipeline(pipeline_timeout)
    return {"sentiment": sentiment, "pipeline": pipeline}


def run_daemon(interval: int, pipeline_timeout: int) -> None:
    """Main loop — runs until SIGTERM / SIGINT."""
    global _shutdown

    def _handle(sig, _frame):
        global _shutdown
        print(f"\n[scheduler] Signal {sig} received — finishing current run then stopping.")
        _shutdown = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    n_runs = 0
    print(f"[scheduler] Started. Interval: {interval}s ({interval / 60:.1f} min).")
    print(f"[scheduler] Status file: {STATUS_PATH}")

    while not _shutdown:
        n_runs += 1
        print(f"\n[scheduler] === Run #{n_runs} at {_now()} ===")

        _write_status({
            "status": "running",
            "last_run_utc": None,
            "next_run_utc": None,
            "interval_seconds": interval,
            "n_runs": n_runs,
            "last_pipeline_status": None,
        })

        result        = run_once(pipeline_timeout)
        last_run_utc  = _now()
        next_ts       = datetime.now(timezone.utc).timestamp() + interval
        next_run_utc  = datetime.fromtimestamp(next_ts, tz=timezone.utc).isoformat()

        _write_status({
            "status": "idle",
            "last_run_utc": last_run_utc,
            "next_run_utc": next_run_utc,
            "interval_seconds": interval,
            "n_runs": n_runs,
            "last_pipeline_status": result["pipeline"]["status"],
            "last_sentiment_triggered": result["sentiment"].get("triggered", False),
        })

        print(f"[scheduler] Run #{n_runs} done — "
              f"pipeline={result['pipeline']['status']}. "
              f"Next in {interval}s.")

        # Sleep in 5-second chunks so SIGTERM exits promptly
        slept = 0
        while slept < interval and not _shutdown:
            time.sleep(min(5, interval - slept))
            slept += 5

    _write_status({
        "status": "stopped",
        "last_run_utc": _now(),
        "interval_seconds": interval,
        "n_runs": n_runs,
    })
    print("[scheduler] Stopped cleanly.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Realtime pipeline scheduler")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit (cron mode)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS,
                        help=f"Interval between pipeline runs in seconds "
                             f"(default: {DEFAULT_INTERVAL_SECONDS})")
    parser.add_argument("--pipeline-timeout", type=int, default=DEFAULT_PIPELINE_TIMEOUT,
                        help=f"Max seconds for one pipeline run "
                             f"(default: {DEFAULT_PIPELINE_TIMEOUT})")
    args = parser.parse_args()

    REALTIME_DIR.mkdir(parents=True, exist_ok=True)

    if args.once:
        result = run_once(args.pipeline_timeout)
        _write_status({
            "status": "idle",
            "last_run_utc": _now(),
            "next_run_utc": None,
            "interval_seconds": args.interval,
            "n_runs": 1,
            "last_pipeline_status": result["pipeline"]["status"],
            "last_sentiment_triggered": result["sentiment"].get("triggered", False),
            "mode": "once",
        })
        print(f"\nPipeline: {result['pipeline']['status']}")
        return 0 if result["pipeline"]["status"] == "ok" else 1

    run_daemon(args.interval, args.pipeline_timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
