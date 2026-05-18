"""
realtime_drift_detector.py

Compares rolling live accuracy (from live_accuracy_summary.json) against
the walk-forward baseline. Writes drift_status.json for the API and dashboard.

Thresholds (applied to the last-30-signal rolling window):
  drop <  5 pp below baseline  → OK
  drop >= 5 pp below baseline  → WARNING
  drop >= 10 pp below baseline → DRIFT_DETECTED
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import REALTIME_DIR

ACCURACY_SUMM_PATH = REALTIME_DIR / "live_accuracy_summary.json"
DRIFT_STATUS_PATH  = REALTIME_DIR / "drift_status.json"

MODEL_DIR       = Path(__file__).resolve().parent.parent / "models"
WF_RESULTS_PATH = MODEL_DIR / "walk_forward_results.json"

MIN_SIGNALS   = 10    # need at least this many settled signals before judging
WARNING_DROP  = 0.05  # 5 pp below baseline → WARNING
DRIFT_DROP    = 0.10  # 10 pp below baseline → DRIFT_DETECTED


def _baseline_accuracy() -> float:
    """Return gated combined_precision from walk-forward results as the baseline."""
    try:
        wf = json.loads(WF_RESULTS_PATH.read_text())
        return float(wf["aggregate"]["gated_production"]["combined_precision"])
    except Exception:
        return 0.74  # fallback: known backtest value


def main() -> int:
    print("=" * 60)
    print("DRIFT DETECTOR")
    print("=" * 60)

    baseline = _baseline_accuracy()
    now      = datetime.now(timezone.utc).isoformat()

    if not ACCURACY_SUMM_PATH.exists():
        status = {
            "status": "NO_DATA",
            "detail": f"No live accuracy data yet. Tracker needs ≥{MIN_SIGNALS} settled signals.",
            "baseline_accuracy": round(baseline, 4),
            "live_accuracy": None,
            "live_accuracy_all": None,
            "n_signals": 0,
            "generated_at_utc": now,
        }
        DRIFT_STATUS_PATH.write_text(json.dumps(status, indent=2))
        print("  No accuracy summary — drift status: NO_DATA")
        return 0

    try:
        summ = json.loads(ACCURACY_SUMM_PATH.read_text())
    except Exception as exc:
        print(f"  Failed to read accuracy summary: {exc}")
        return 1

    n        = summ.get("n_signals", 0)
    live_acc = summ.get("accuracy_30") or summ.get("accuracy")
    live_all = summ.get("accuracy")

    if n < MIN_SIGNALS or live_acc is None:
        status_str = "INSUFFICIENT_DATA"
        detail     = f"Only {n} settled signals (need ≥{MIN_SIGNALS})."
    else:
        drop = baseline - live_acc
        if drop >= DRIFT_DROP:
            status_str = "DRIFT_DETECTED"
            detail = (
                f"Live accuracy {live_acc * 100:.1f}% is {drop * 100:.1f} pp below "
                f"baseline {baseline * 100:.1f}%. Model retraining recommended."
            )
        elif drop >= WARNING_DROP:
            status_str = "WARNING"
            detail = (
                f"Live accuracy {live_acc * 100:.1f}% is {drop * 100:.1f} pp below "
                f"baseline {baseline * 100:.1f}%. Monitor closely."
            )
        else:
            status_str = "OK"
            detail = (
                f"Live accuracy {live_acc * 100:.1f}% is within "
                f"{abs(drop) * 100:.1f} pp of baseline {baseline * 100:.1f}%."
            )

    status = {
        "status":            status_str,
        "detail":            detail,
        "baseline_accuracy": round(baseline, 4),
        "live_accuracy":     live_acc,
        "live_accuracy_all": live_all,
        "n_signals":         n,
        "generated_at_utc":  now,
    }
    DRIFT_STATUS_PATH.write_text(json.dumps(status, indent=2))
    print(f"  Drift status : {status_str}")
    print(f"  {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
