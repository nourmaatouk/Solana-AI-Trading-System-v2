"""
realtime_sentiment_daemon.py

Long-running sentiment refresh daemon.  Run this as a supervised process
(systemd, screen, or Docker) before starting the 15-minute pipeline cron.

Every --interval seconds (default 900 = 15 min) the daemon calls
realtime_sentiment_fetcher.py as a subprocess and writes:

  data/realtime/latest_sentiment_15m.csv   (the sentiment cache)
  data/realtime/sentiment_status.json      (health contract)

The pipeline's --use-sentiment-daemon flag reads sentiment_status.json to
decide whether to skip the inline sentiment_fetch step and use the cached CSV
instead.  This prevents GDELT/HuggingFace latency from blocking the 15-minute
pipeline run.

Usage:
  python realtime_sentiment_daemon.py                # 15-min refresh interval
  python realtime_sentiment_daemon.py --interval 300 # 5-min refresh interval
  python realtime_sentiment_daemon.py --once         # fetch once and exit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REALTIME_DIR = BASE_DIR / "data" / "realtime"
STATUS_PATH = REALTIME_DIR / "sentiment_status.json"
FETCHER_SCRIPT = BASE_DIR / "realtime_sentiment_fetcher.py"

DEFAULT_INTERVAL_SECONDS = 900  # 15 minutes
SUBPROCESS_TIMEOUT_SECONDS = 240  # generous — FinBERT + GDELT can be slow


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write_status(
    status: str,
    *,
    error: str | None = None,
    articles_fetched: int | None = None,
    last_run_utc: str | None = None,
    next_run_utc: str | None = None,
) -> None:
    REALTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "last_run_utc": last_run_utc or _now().isoformat(),
        "next_run_utc": next_run_utc,
        "articles_fetched": articles_fetched,
        "error": error,
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2))


def run_once(timeout: int = SUBPROCESS_TIMEOUT_SECONDS) -> dict:
    """Run realtime_sentiment_fetcher.py once and return a result dict."""
    started = _now()
    print(f"\n[{started.isoformat()}] Running sentiment fetcher …")
    try:
        completed = subprocess.run(
            [sys.executable, str(FETCHER_SCRIPT)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        finished = _now()
        elapsed = (finished - started).total_seconds()
        if completed.returncode == 0:
            print(f"  OK in {elapsed:.1f}s")
            if completed.stdout:
                print(completed.stdout.rstrip())
            return {"status": "ok", "elapsed": elapsed, "returncode": 0, "error": None}
        else:
            msg = f"fetcher exited {completed.returncode}"
            print(f"  FAILED: {msg}")
            if completed.stderr:
                print("  stderr:", completed.stderr[:400])
            return {"status": "failed", "elapsed": elapsed, "returncode": completed.returncode, "error": msg}
    except subprocess.TimeoutExpired:
        elapsed = (finished := _now()) - started
        msg = f"fetcher timed out after {timeout}s"
        print(f"  TIMEOUT: {msg}")
        return {"status": "timeout", "elapsed": timeout, "returncode": 124, "error": msg}
    except Exception as exc:
        msg = str(exc)
        print(f"  ERROR: {msg}")
        return {"status": "error", "elapsed": 0, "returncode": -1, "error": msg}


def run_daemon(interval: int) -> None:
    """Loop indefinitely, running the fetcher every `interval` seconds."""
    print("=" * 60)
    print("Sentiment Daemon  —  interval=%ds  (Ctrl-C to stop)" % interval)
    print("=" * 60)

    while True:
        started = _now()
        next_run = datetime.fromtimestamp(started.timestamp() + interval, tz=timezone.utc)

        _write_status(
            "running",
            last_run_utc=started.isoformat(),
            next_run_utc=next_run.isoformat(),
        )

        result = run_once()

        _write_status(
            result["status"],
            error=result["error"],
            last_run_utc=started.isoformat(),
            next_run_utc=next_run.isoformat(),
        )

        sleep_secs = max(0.0, (next_run - _now()).total_seconds())
        print(f"Next run in {sleep_secs:.0f}s  ({next_run.isoformat()})")
        time.sleep(sleep_secs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sentiment refresh daemon for the realtime pipeline.")
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS,
                   help="Refresh interval in seconds (default: 900 = 15 min).")
    p.add_argument("--once", action="store_true",
                   help="Run the fetcher exactly once then exit (useful for cron).")
    p.add_argument("--timeout", type=int, default=SUBPROCESS_TIMEOUT_SECONDS,
                   help="Per-run subprocess timeout in seconds.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.once:
        result = run_once(timeout=args.timeout)
        _write_status(result["status"], error=result["error"])
        raise SystemExit(0 if result["status"] == "ok" else 1)
    else:
        try:
            run_daemon(args.interval)
        except KeyboardInterrupt:
            print("\nDaemon stopped.")
            _write_status("stopped")
