from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]
REALTIME_DIR = BASE_DIR / "data" / "realtime"
LOG_DIR = REALTIME_DIR / "logs"
LAST_PIPELINE_RUN_PATH = REALTIME_DIR / "last_pipeline_run.json"
LAST_SIGNAL_PATH = REALTIME_DIR / "last_signal.json"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    script: Path
    critical: bool = True


def build_steps(args: argparse.Namespace) -> list[PipelineStep]:
    steps = []
    if not args.skip_market:
        steps.append(PipelineStep("market_fetch", BASE_DIR / "realtime_market_fetcher.py"))

    steps.append(PipelineStep("feature_builder", BASE_DIR / "realtime_feature_builder.py"))

    if not args.skip_sentiment:
        steps.append(PipelineStep("sentiment_fetch", BASE_DIR / "realtime_sentiment_fetcher.py"))

    steps.extend(
        [
            PipelineStep("feature_merge", BASE_DIR / "realtime_merge_features.py"),
            PipelineStep("predictor", BASE_DIR / "realtime_predictor.py"),
        ]
    )

    if not args.skip_dashboard:
        steps.append(PipelineStep("dashboard", BASE_DIR / "22_realtime_classifier_dashboard.py", critical=False))

    return steps


def append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(text)


def run_step(step: PipelineStep, timeout_seconds: int, dry_run: bool, log_path: Path) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    cmd = [sys.executable, str(step.script)]

    result: dict[str, Any] = {
        "name": step.name,
        "script": str(step.script),
        "critical": step.critical,
        "command": cmd,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "returncode": None,
        "status": "dry_run" if dry_run else "pending",
    }

    append_log(log_path, f"\n{'=' * 80}\n[{started_at.isoformat()}] {step.name}\n$ {' '.join(cmd)}\n")

    if dry_run:
        finished_at = datetime.now(timezone.utc)
        result["finished_at_utc"] = finished_at.isoformat()
        result["returncode"] = 0
        append_log(log_path, "[dry-run] skipped execution\n")
        return result

    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        finished_at = datetime.now(timezone.utc)
        result["finished_at_utc"] = finished_at.isoformat()
        result["returncode"] = completed.returncode
        result["status"] = "ok" if completed.returncode == 0 else "failed"
        append_log(log_path, completed.stdout)
        if completed.stderr:
            append_log(log_path, "\n[stderr]\n" + completed.stderr)
        return result
    except subprocess.TimeoutExpired as e:
        finished_at = datetime.now(timezone.utc)
        result["finished_at_utc"] = finished_at.isoformat()
        result["returncode"] = 124
        result["status"] = "timeout"
        append_log(log_path, f"\nTIMEOUT after {timeout_seconds}s\n")
        if e.stdout:
            append_log(log_path, str(e.stdout))
        if e.stderr:
            append_log(log_path, "\n[stderr]\n" + str(e.stderr))
        return result


def read_last_signal() -> dict[str, Any] | None:
    if not LAST_SIGNAL_PATH.exists():
        return None
    with open(LAST_SIGNAL_PATH, "r") as f:
        return json.load(f)


def write_pipeline_run(payload: dict[str, Any]) -> None:
    LAST_PIPELINE_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LAST_PIPELINE_RUN_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 15-minute realtime classification pipeline once.")
    parser.add_argument("--skip-market", action="store_true", help="Use the existing latest_sol_15m.csv instead of fetching Binance candles.")
    parser.add_argument("--skip-sentiment", action="store_true", help="Use the existing latest_sentiment_15m.csv instead of fetching news.")
    parser.add_argument("--skip-dashboard", action="store_true", help="Do not regenerate realtime_classifier_dashboard.html.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after failed critical steps. Mostly useful for debugging.")
    parser.add_argument("--dry-run", action="store_true", help="Print/write the planned steps without running them.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per step in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"realtime_pipeline_{run_id}.log"
    steps = build_steps(args)

    print("=" * 60)
    print("Realtime Classification Pipeline")
    print("=" * 60)
    print(f"Run ID       : {run_id}")
    print(f"Log path     : {log_path}")
    print(f"Last signal  : {LAST_SIGNAL_PATH}")
    print("Note         : orderbook WebSocket should run separately.")
    print()

    step_results = []
    overall_status = "ok"

    for step in steps:
        print(f"Running step: {step.name}")
        result = run_step(step, args.timeout, args.dry_run, log_path)
        step_results.append(result)

        status = result["status"]
        returncode = result["returncode"]
        print(f"  -> {status} (code={returncode})")

        if status != "ok" and status != "dry_run":
            if step.critical and not args.continue_on_error:
                overall_status = "failed"
                print(f"Stopping at failed critical step: {step.name}")
                break
            overall_status = "warning"

    last_signal = read_last_signal()
    payload = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status if not args.dry_run else "dry_run",
        "steps": step_results,
        "last_signal_path": str(LAST_SIGNAL_PATH),
        "last_signal": last_signal,
        "log_path": str(log_path),
    }
    write_pipeline_run(payload)

    print()
    print(f"Pipeline status: {payload['status']}")
    print(f"Run JSON       : {LAST_PIPELINE_RUN_PATH}")
    if last_signal:
        print(f"Latest signal  : {last_signal.get('signal')} ({last_signal.get('reason')})")
    else:
        print("Latest signal  : unavailable")

    return 0 if payload["status"] in {"ok", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
