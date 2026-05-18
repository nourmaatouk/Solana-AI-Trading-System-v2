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
LAST_PIPELINE_RUN_PATH  = REALTIME_DIR / "last_pipeline_run.json"
LAST_SIGNAL_PATH        = REALTIME_DIR / "last_signal.json"
LAST_FUSED_DECISION_PATH = REALTIME_DIR / "last_fused_decision.json"
ORDERBOOK_STATUS_PATH   = REALTIME_DIR / "orderbook_status.json"
SENTIMENT_STATUS_PATH   = REALTIME_DIR / "sentiment_status.json"
DEFAULT_ORDERBOOK_MAX_AGE_MINUTES = 30
DEFAULT_SENTIMENT_MAX_AGE_MINUTES = 30


@dataclass(frozen=True)
class PipelineStep:
    name: str
    script: Path
    critical: bool = True


def check_sentiment_daemon_fresh(max_age_minutes: int) -> tuple[bool, str]:
    """Return (is_fresh, reason).  Fresh means the daemon ran recently and succeeded."""
    if not SENTIMENT_STATUS_PATH.exists():
        return False, f"sentiment_status.json missing: {SENTIMENT_STATUS_PATH}"
    try:
        payload = json.loads(SENTIMENT_STATUS_PATH.read_text())
    except Exception as exc:
        return False, f"sentiment_status.json unreadable: {exc}"

    status = payload.get("status", "unknown")
    if status not in ("ok", "running"):
        return False, f"sentiment daemon status is {status!r}"

    last_run = _parse_utc(payload.get("last_run_utc"))
    if last_run is None:
        return False, "sentiment_status.json has no parseable last_run_utc"

    age_min = (datetime.now(timezone.utc) - last_run).total_seconds() / 60.0
    if age_min > max_age_minutes:
        return False, f"sentiment cache is stale ({age_min:.1f} min > {max_age_minutes})"

    return True, f"sentiment cache is fresh ({age_min:.1f} min old)"


def build_steps(args: argparse.Namespace) -> list[PipelineStep]:
    steps = []
    if not args.skip_market:
        steps.append(PipelineStep("market_fetch", BASE_DIR / "realtime_market_fetcher.py"))
        steps.append(PipelineStep("btc_fetch", BASE_DIR / "realtime_btc_fetcher.py"))

    steps.append(PipelineStep("feature_builder", BASE_DIR / "realtime_feature_builder.py"))
    steps.append(PipelineStep("hawkes_features", BASE_DIR / "realtime_hawkes_features.py", critical=False))

    if not args.skip_sentiment:
        if args.use_sentiment_daemon:
            # Daemon mode: validate cache freshness instead of running the fetcher live.
            # The step name "sentiment_cache_check" is handled specially in main().
            steps.append(PipelineStep("sentiment_cache_check", BASE_DIR / "realtime_sentiment_fetcher.py", critical=False))
        else:
            steps.append(PipelineStep("sentiment_fetch", BASE_DIR / "realtime_sentiment_fetcher.py"))

    steps.extend(
        [
            PipelineStep("feature_merge", BASE_DIR / "realtime_merge_features.py"),
            PipelineStep("predictor", BASE_DIR / "realtime_predictor.py"),
            PipelineStep("rl_predictor", BASE_DIR.parent / "rl" / "realtime_rl_predictor.py", critical=False),
        ]
    )

    if not args.skip_dashboard:
        steps.append(PipelineStep("dashboard", BASE_DIR / "22_realtime_classifier_dashboard.py", critical=False))

    if args.fusion:
        steps.append(PipelineStep("fusion", BASE_DIR / "realtime_fusion_predictor.py", critical=False))

    if args.fusion_aggregate or args.paper_trade:
        steps.append(PipelineStep("spoofing_detect", BASE_DIR / "realtime_spoofing_detector.py", critical=False))

    if args.fusion_aggregate:
        steps.append(PipelineStep("fusion_aggregate", BASE_DIR / "realtime_fusion_aggregator.py", critical=False))

    if args.paper_trade:
        steps.append(PipelineStep("paper_trader", BASE_DIR / "realtime_paper_trader.py", critical=False))

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


def _parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def validate_orderbook_status(max_age_minutes: int) -> tuple[bool, str, dict[str, Any]]:
    if not ORDERBOOK_STATUS_PATH.exists():
        return False, f"orderbook status file missing: {ORDERBOOK_STATUS_PATH}", {}

    try:
        with open(ORDERBOOK_STATUS_PATH, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"orderbook status file unreadable: {exc}", {}

    status = str(payload.get("status", "unknown"))
    connected = bool(payload.get("connected", False))
    latest_ts = _parse_utc(payload.get("latest_orderbook_timestamp"))
    status_ts = _parse_utc(payload.get("last_update_utc"))
    now = datetime.now(timezone.utc)

    details = {
        "status_path": str(ORDERBOOK_STATUS_PATH),
        "status": status,
        "connected": connected,
        "latest_orderbook_timestamp": payload.get("latest_orderbook_timestamp"),
        "last_update_utc": payload.get("last_update_utc"),
        "max_age_minutes": max_age_minutes,
        "last_error": payload.get("last_error"),
        "source": payload.get("source"),
    }

    if status != "running":
        return False, f"orderbook collector status is {status!r}; expected 'running'", details
    if not connected:
        return False, "orderbook collector is running but not connected", details
    if latest_ts is None:
        return False, "orderbook status has no parseable latest_orderbook_timestamp", details
    if status_ts is None:
        return False, "orderbook status has no parseable last_update_utc", details

    data_age = (now - latest_ts).total_seconds() / 60.0
    status_age = (now - status_ts).total_seconds() / 60.0
    details["orderbook_age_minutes"] = round(data_age, 2)
    details["status_age_minutes"] = round(status_age, 2)

    if data_age > max_age_minutes:
        return False, f"orderbook data is stale ({data_age:.1f} min > {max_age_minutes})", details
    if status_age > max_age_minutes:
        return False, f"orderbook status is stale ({status_age:.1f} min > {max_age_minutes})", details
    if payload.get("source") not in (None, "binance_partial_depth"):
        return False, f"unexpected orderbook source: {payload.get('source')}", details

    return True, "orderbook collector is running and fresh", details


def run_orderbook_preflight(max_age_minutes: int, dry_run: bool, log_path: Path) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "name": "orderbook_preflight",
        "script": None,
        "critical": True,
        "command": [],
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "returncode": None,
        "status": "dry_run" if dry_run else "pending",
    }
    append_log(log_path, f"\n{'=' * 80}\n[{started_at.isoformat()}] orderbook_preflight\n")

    if dry_run:
        finished_at = datetime.now(timezone.utc)
        result.update({
            "finished_at_utc": finished_at.isoformat(),
            "returncode": 0,
            "status": "dry_run",
            "reason": "dry-run skipped orderbook preflight",
        })
        append_log(log_path, "[dry-run] skipped orderbook preflight\n")
        return result

    ok, reason, details = validate_orderbook_status(max_age_minutes)
    finished_at = datetime.now(timezone.utc)
    result.update({
        "finished_at_utc": finished_at.isoformat(),
        "returncode": 0 if ok else 2,
        "status": "ok" if ok else "degraded",
        "reason": reason,
        "details": details,
    })
    append_log(log_path, json.dumps(result, indent=2) + "\n")
    return result


def read_last_signal() -> dict[str, Any] | None:
    if not LAST_SIGNAL_PATH.exists():
        return None
    with open(LAST_SIGNAL_PATH, "r") as f:
        return json.load(f)


def read_last_fused_decision() -> dict[str, Any] | None:
    if not LAST_FUSED_DECISION_PATH.exists():
        return None
    with open(LAST_FUSED_DECISION_PATH, "r") as f:
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
    parser.add_argument("--paper-trade", action="store_true", help="Run paper trader after predictor to simulate trades.")
    parser.add_argument("--fusion", action="store_true", help="Run DNN+XGBoost+sentiment fusion predictor (requires trained DNN artifacts).")
    parser.add_argument("--fusion-aggregate", action="store_true", dest="fusion_aggregate", help="Run lightweight multi-signal fusion aggregator (classification + spoofing + sentiment).")
    parser.add_argument("--orderbook-max-age-minutes", type=int, default=DEFAULT_ORDERBOOK_MAX_AGE_MINUTES,
                        help="Maximum allowed orderbook/status age before model steps are blocked.")
    parser.add_argument("--use-sentiment-daemon", action="store_true", dest="use_sentiment_daemon",
                        help="Read sentiment from the daemon cache (sentiment_status.json) instead of "
                             "running the live fetcher inline.  Start realtime_sentiment_daemon.py first.")
    parser.add_argument("--sentiment-max-age-minutes", type=int, default=DEFAULT_SENTIMENT_MAX_AGE_MINUTES,
                        dest="sentiment_max_age_minutes",
                        help="Maximum allowed sentiment cache age when --use-sentiment-daemon is set.")
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
    orderbook_preflight_done = False

    for step in steps:
        if step.name == "feature_merge" and not orderbook_preflight_done:
            orderbook_preflight_done = True
            print("Running step: orderbook_preflight")
            preflight = run_orderbook_preflight(args.orderbook_max_age_minutes, args.dry_run, log_path)
            step_results.append(preflight)
            print(f"  -> {preflight['status']} (code={preflight['returncode']})")
            if preflight["status"] not in {"ok", "dry_run"}:
                overall_status = "degraded"
                print(f"Stopping before model steps: {preflight.get('reason')}")
                break

        # Daemon-mode sentiment: validate cache instead of running the fetcher live.
        if step.name == "sentiment_cache_check":
            print("Running step: sentiment_cache_check")
            started_at_step = datetime.now(timezone.utc)
            if args.dry_run:
                cache_result: dict[str, Any] = {
                    "name": "sentiment_cache_check", "script": str(step.script),
                    "critical": False, "command": [], "started_at_utc": started_at_step.isoformat(),
                    "finished_at_utc": started_at_step.isoformat(), "returncode": 0, "status": "dry_run",
                }
            else:
                fresh, reason = check_sentiment_daemon_fresh(args.sentiment_max_age_minutes)
                cache_result = {
                    "name": "sentiment_cache_check", "script": str(step.script),
                    "critical": False, "command": [], "started_at_utc": started_at_step.isoformat(),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "returncode": 0 if fresh else 1,
                    "status": "ok" if fresh else "warning",
                    "reason": reason,
                }
                if not fresh:
                    print(f"  WARN: sentiment cache not fresh — {reason}. Continuing with existing CSV.")
                    overall_status = "warning"
            step_results.append(cache_result)
            print(f"  -> {cache_result['status']} (code={cache_result['returncode']})")
            continue

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

    last_signal        = read_last_signal()
    last_fused_decision = read_last_fused_decision()
    degraded_reason = next(
        (result.get("reason") for result in step_results if result.get("status") == "degraded"),
        None,
    )
    payload = {
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall_status if not args.dry_run else "dry_run",
        "degraded_reason": degraded_reason,
        "steps": step_results,
        "last_signal_path": str(LAST_SIGNAL_PATH),
        "last_signal": last_signal,
        "last_fused_decision": last_fused_decision,
        "log_path": str(log_path),
    }
    write_pipeline_run(payload)

    print()
    print(f"Pipeline status: {payload['status']}")
    if payload.get("degraded_reason"):
        print(f"Degraded reason: {payload['degraded_reason']}")
    print(f"Run JSON       : {LAST_PIPELINE_RUN_PATH}")
    if payload["status"] == "degraded":
        print("Latest signal  : not refreshed")
    elif last_signal:
        print(f"Latest signal  : {last_signal.get('signal')} ({last_signal.get('reason')})")
    else:
        print("Latest signal  : unavailable")
    if payload["status"] == "degraded":
        print("Fused decision : not refreshed")
    elif last_fused_decision:
        print(f"Fused decision : {last_fused_decision.get('action')} "
              f"(score={last_fused_decision.get('fused_score', 'n/a')}, "
              f"risk={last_fused_decision.get('risk_level', 'n/a')})")

    return 0 if payload["status"] in {"ok", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
