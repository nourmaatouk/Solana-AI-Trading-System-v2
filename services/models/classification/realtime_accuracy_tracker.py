"""
realtime_accuracy_tracker.py

Evaluates the live accuracy of past BUY/SELL signals by comparing the entry
close price with the close price ACTIVE_HORIZON bars (1 hour) later, using
entries already stored in paper_trading_history.json.

Run after each paper trading step by the scheduler.

Writes:
  data/realtime/live_accuracy_log.json     — individual signal outcomes
  data/realtime/live_accuracy_summary.json — rolling accuracy stats (all / last-30 / last-60)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import ACTIVE_HORIZON, REALTIME_DIR

HISTORY_PATH       = REALTIME_DIR / "paper_trading_history.json"
ACCURACY_LOG_PATH  = REALTIME_DIR / "live_accuracy_log.json"
ACCURACY_SUMM_PATH = REALTIME_DIR / "live_accuracy_summary.json"

# Minimum price move to consider a signal correct (covers round-trip fees)
MIN_MOVE_PCT = 0.001  # 0.1 %


def _evaluate_signals(history: list[dict]) -> list[dict]:
    """
    For each BUY/SELL entry at index i, look up the close price ACTIVE_HORIZON
    bars later. Mark correct if price moved in the predicted direction by at
    least MIN_MOVE_PCT.

    Entries without a settled outcome (< ACTIVE_HORIZON bars after them) are
    skipped — they cannot be evaluated yet.
    """
    evaluated: list[dict] = []
    for i, entry in enumerate(history):
        action = str(entry.get("action", "HOLD")).upper()
        if action not in ("BUY", "SELL"):
            continue

        outcome_idx = i + ACTIVE_HORIZON
        if outcome_idx >= len(history):
            continue  # not enough future bars yet

        entry_price   = entry.get("latest_close")
        outcome_price = history[outcome_idx].get("latest_close")
        if entry_price is None or outcome_price is None:
            continue

        entry_price   = float(entry_price)
        outcome_price = float(outcome_price)
        move          = (outcome_price - entry_price) / entry_price

        correct = (move >= MIN_MOVE_PCT) if action == "BUY" else (move <= -MIN_MOVE_PCT)

        evaluated.append({
            "signal_time":   entry.get("run_utc"),
            "outcome_time":  history[outcome_idx].get("run_utc"),
            "action":        action,
            "entry_price":   round(entry_price, 6),
            "outcome_price": round(outcome_price, 6),
            "move_pct":      round(move * 100, 4),
            "correct":       correct,
            "horizon_bars":  ACTIVE_HORIZON,
        })

    return evaluated


def _compute_summary(log: list[dict]) -> dict:
    now = datetime.now(timezone.utc).isoformat()

    if not log:
        return {
            "n_signals": 0, "n_correct": 0,
            "accuracy": None, "accuracy_30": None, "accuracy_60": None,
            "buy_signals": 0, "sell_signals": 0,
            "buy_accuracy": None, "sell_accuracy": None,
            "generated_at_utc": now,
        }

    def acc(subset: list[dict]) -> float | None:
        if not subset:
            return None
        return round(sum(1 for e in subset if e["correct"]) / len(subset), 4)

    sorted_log = sorted(log, key=lambda e: e.get("signal_time") or "")
    buys  = [e for e in sorted_log if e["action"] == "BUY"]
    sells = [e for e in sorted_log if e["action"] == "SELL"]

    return {
        "n_signals":       len(sorted_log),
        "n_correct":       sum(1 for e in sorted_log if e["correct"]),
        "accuracy":        acc(sorted_log),
        "accuracy_30":     acc(sorted_log[-30:]),
        "accuracy_60":     acc(sorted_log[-60:]),
        "buy_signals":     len(buys),
        "sell_signals":    len(sells),
        "buy_accuracy":    acc(buys),
        "sell_accuracy":   acc(sells),
        "generated_at_utc": now,
    }


def main() -> int:
    print("=" * 60)
    print("ACCURACY TRACKER")
    print("=" * 60)

    if not HISTORY_PATH.exists():
        print("  No history file found — skipping.")
        return 0

    try:
        history = json.loads(HISTORY_PATH.read_text())
    except Exception as exc:
        print(f"  Failed to read history: {exc}")
        return 1

    log     = _evaluate_signals(history)
    summary = _compute_summary(log)

    ACCURACY_LOG_PATH.write_text(json.dumps(log, indent=2))
    ACCURACY_SUMM_PATH.write_text(json.dumps(summary, indent=2))

    n   = summary["n_signals"]
    acc = summary.get("accuracy")
    print(f"  Evaluated signals    : {n}")
    if acc is not None:
        print(f"  Overall accuracy     : {acc * 100:.1f}%")
    a30 = summary.get("accuracy_30")
    if a30 is not None:
        print(f"  Last-30  accuracy    : {a30 * 100:.1f}%")
    print(f"  BUY / SELL evaluated : {summary['buy_signals']} / {summary['sell_signals']}")
    print("  Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
