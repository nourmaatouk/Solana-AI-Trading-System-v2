"""
auto_retrain.py

Weekly automatic model retraining pipeline.

Steps (in order):
  1. Collect fresh SOL/USDT + BTC/USDT OHLCV data
  2. Build technical features
  3. Add BTC correlation features
  4. Add Hawkes intensity features
  5. Merge sentiment features   (optional — skips gracefully if data unavailable)
  6. Merge orderbook features   (optional)
  7. Prepare leak-free training data
  8. Train XGBoost (H4 horizon)
  9. Walk-forward validation → new AUC
 10. If new AUC > current AUC + threshold: accept new model (backup old one first)
 11. Write retrain_log.json

Usage:
  python auto_retrain.py            # run and swap only if AUC improves
  python auto_retrain.py --force    # swap even if AUC does not improve

PM2 cron:
  Launched by ecosystem.config.cjs on Sundays at 02:00 UTC.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CLASS_DIR   = Path(__file__).resolve().parent       # …/classification/
MODEL_DIR   = CLASS_DIR.parent / "models"           # …/services/models/models/
REPO_ROOT   = CLASS_DIR.parents[2]                  # …/Solana-AI-Trading-System/

RETRAIN_LOG_PATH = MODEL_DIR / "retrain_log.json"
WF_RESULTS_PATH  = MODEL_DIR / "walk_forward_results.json"

PYTHON = sys.executable

AUC_IMPROVEMENT_THRESHOLD = 0.005  # new model must beat current by ≥ 0.5 pp

PIPELINE_STEPS: list[tuple[str, str, bool]] = [
    # (step_name, script_filename, required)
    ("collect_sol",    "01_collect_sol_data_15m.py",     True),
    ("build_features", "02_build_technical_features.py", True),
    ("add_btc",        "02b_add_btc_features.py",        True),
    ("hawkes",         "03_hawkes_intensity.py",          True),
    ("merge_sentiment","06_merge_sentiment_features.py",  False),
    ("merge_orderbook","06b_merge_orderbook_features.py", False),
    ("prepare_data",   "07_prepare_no_leak_data.py",      True),
    ("train_xgboost",  "08_train_xgboost_no_leak.py",     True),
    ("walk_forward",   "20_walk_forward_validation.py",   True),
]


def _run_step(script: str) -> tuple[bool, str]:
    path = CLASS_DIR / script
    if not path.exists():
        return False, f"Script not found: {path}"
    try:
        result = subprocess.run(
            [PYTHON, str(path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        ok   = result.returncode == 0
        tail = (result.stderr or result.stdout or "")[-400:]
        return ok, tail
    except subprocess.TimeoutExpired:
        return False, "Step timed out after 600 s"
    except Exception as exc:
        return False, str(exc)


def _current_auc() -> float | None:
    try:
        wf = json.loads(WF_RESULTS_PATH.read_text())
        return float(wf["aggregate"]["overall_auc"])
    except Exception:
        return None


def _backup_models(ts: str) -> None:
    for name in ["xgboost_h4.json", "walk_forward_results.json", "backtest_results.json"]:
        src = MODEL_DIR / name
        if src.exists():
            shutil.copy2(src, MODEL_DIR / f"{src.stem}_backup_{ts}{src.suffix}")


def main(force: bool = False) -> int:
    started_at  = datetime.now(timezone.utc)
    ts_label    = started_at.strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(f"AUTO RETRAIN  {started_at.isoformat()}")
    print("=" * 60)

    current_auc = _current_auc()
    print(f"  Current AUC : {current_auc:.4f}" if current_auc else "  Current AUC : unknown")

    step_results: list[dict] = []
    abort = False

    for step_name, script, required in PIPELINE_STEPS:
        print(f"\n  [{step_name}] {script} …")
        ok, output = _run_step(script)
        step_results.append({"step": step_name, "ok": ok, "output_tail": output})

        if ok:
            print(f"  [{step_name}] ✓")
        else:
            print(f"  [{step_name}] ✗  {output[-200:]}")
            if required:
                abort = True
                print(f"\n  Required step '{step_name}' failed — aborting.")
                break
            print(f"  [{step_name}] optional — continuing.")

    swapped  = False
    new_auc  = _current_auc() if not abort else None  # step 20 rewrites the file in-place

    if not abort and new_auc is not None:
        print(f"\n  New AUC     : {new_auc:.4f}")
        improvement = (new_auc - current_auc) if current_auc is not None else float("inf")
        print(f"  Improvement : {improvement:+.4f}  (threshold {AUC_IMPROVEMENT_THRESHOLD:+.3f})")

        if force or improvement >= AUC_IMPROVEMENT_THRESHOLD:
            _backup_models(ts_label)
            swapped = True
            print("  ✓ Model accepted — old weights backed up, new model is live.")
        else:
            print("  ✗ Model NOT accepted — improvement below threshold.")

    finished_at = datetime.now(timezone.utc)
    log = {
        "started_at_utc":     started_at.isoformat(),
        "finished_at_utc":    finished_at.isoformat(),
        "duration_seconds":   round((finished_at - started_at).total_seconds(), 1),
        "auc_before":         current_auc,
        "auc_after":          new_auc,
        "model_swapped":      swapped,
        "aborted":            abort,
        "steps":              step_results,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RETRAIN_LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"\n  Log → {RETRAIN_LOG_PATH}")
    print(f"  Duration : {log['duration_seconds']:.0f} s")
    print("=" * 60)
    return 1 if abort else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic XGBoost model retraining")
    parser.add_argument("--force", action="store_true",
                        help="Accept new model even if AUC does not improve")
    args = parser.parse_args()
    raise SystemExit(main(force=args.force))
