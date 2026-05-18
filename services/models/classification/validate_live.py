"""
validate_live.py — Daily live paper-trading validation snapshot.

Reads the live paper_trading_state_fusion.json and compares against
backtest reference targets.  Run once per day (or on demand) to check
whether live performance is tracking expectations.

Usage:
    python validate_live.py
    python validate_live.py --state path/to/state.json   # custom path
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
REALTIME_DIR = BASE_DIR / "data" / "realtime"
MODELS_DIR   = BASE_DIR.parent / "models"

DEFAULT_STATE   = REALTIME_DIR / "paper_trading_state_fusion.json"
BACKTEST_JSON   = MODELS_DIR   / "backtest_results.json"
WF_JSON         = MODELS_DIR   / "walk_forward_results.json"

# ── Reference targets (from backtest + walk-forward) ──────────────────────────
TARGET_WIN_RATE      = 0.55   # pass threshold (backtest: 74%)
TARGET_SHARPE        = 2.0    # pass threshold (backtest: 5.0)
TARGET_MAX_DRAWDOWN  = 0.20   # fail above this (backtest: 21.7%)
MIN_TRADES_SHARPE    = 10     # need ≥10 trades for reliable Sharpe
MIN_TRADES_WINRATE   = 5      # need ≥5 trades for meaningful win rate

BACKTEST_WIN_RATE    = 0.741
BACKTEST_SHARPE      = 4.998
BACKTEST_GATED_PREC  = 0.793


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ERROR: cannot read {path}: {e}")
        sys.exit(1)


def _pct(v: float, decimals: int = 2) -> str:
    return f"{v * 100:+.{decimals}f}%"


def _pass(ok: bool) -> str:
    return "PASS ✓" if ok else "FAIL ✗"


def _compute_sharpe(net_pnls: list[float]) -> float | None:
    if len(net_pnls) < MIN_TRADES_SHARPE:
        return None
    arr = np.array(net_pnls)
    std = arr.std()
    if std < 1e-12:
        return None
    # Annualise at 15-min bars, 252 trading days, 96 bars/day
    trades_per_year = len(arr) / (len(arr) / (252 * 96)) if len(arr) > 0 else 252 * 96
    # Simpler: use historical trade frequency assuming one trade per ~6h on average
    return float(arr.mean() / std * np.sqrt(252 * 96))


def _days_since(iso_ts: str) -> float | None:
    try:
        t = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - t).total_seconds() / 86_400
    except Exception:
        return None


# ── Main report ───────────────────────────────────────────────────────────────
def run(state_path: Path) -> None:
    state   = _load(state_path)
    trades  = state.get("trade_log", [])
    n       = len(trades)
    equity  = float(state.get("equity", 1.0))
    peak    = float(state.get("peak_equity", 1.0))
    n_wins  = int(state.get("n_wins", sum(t.get("won", 0) for t in trades)))
    open_pos = state.get("open_position")

    # Derived metrics
    win_rate    = n_wins / n if n > 0 else None
    total_ret   = equity - 1.0
    max_dd      = max(
        ((peak_v - v) / peak_v)
        for peak_v, v in zip(
            np.maximum.accumulate([1.0] + [t["equity_after"] for t in trades]),
            [1.0] + [t["equity_after"] for t in trades],
        )
    ) if n > 0 else 0.0

    net_pnls = [t.get("net_pnl", 0.0) for t in trades]
    sharpe   = _compute_sharpe(net_pnls)

    # Time context
    first_entry = trades[0]["entry_time"] if trades else None
    days_running = _days_since(first_entry) if first_entry else None
    trades_per_day = n / days_running if (days_running and days_running > 0) else None

    # ── Header ────────────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print()
    print("=" * 60)
    print(f"  LIVE PAPER TRADING  —  VALIDATION SNAPSHOT")
    print(f"  {now}")
    print("=" * 60)

    # ── Current state ─────────────────────────────────────────────────────────
    print()
    print("  CURRENT STATE")
    print(f"  {'Equity':<22} {equity:.6f}  ({_pct(total_ret)} from baseline 1.0)")
    print(f"  {'Peak equity':<22} {peak:.6f}")
    print(f"  {'Open position':<22} {'YES — ' + open_pos['direction'] + ' @ ' + str(open_pos.get('entry_price','?')) if open_pos else 'None'}")
    print(f"  {'Completed trades':<22} {n}")
    if days_running is not None:
        print(f"  {'Running since':<22} {days_running:.1f} days  (~{trades_per_day:.1f} trades/day)")

    # ── Metric scorecard ──────────────────────────────────────────────────────
    print()
    print("  SCORECARD  (Live → Target → Backtest reference)")
    print(f"  {'─'*56}")

    # Win rate
    wr_str = f"{win_rate*100:.1f}%" if win_rate is not None else f"N/A (need ≥{MIN_TRADES_WINRATE} trades, have {n})"
    wr_ok  = win_rate is not None and win_rate >= TARGET_WIN_RATE
    wr_status = _pass(wr_ok) if win_rate is not None else "PENDING"
    ref_wr = f"{BACKTEST_WIN_RATE*100:.1f}%"
    print(f"  {'Win rate':<20} {wr_str:<12}  target ≥{TARGET_WIN_RATE*100:.0f}%   ref {ref_wr:<8}  {wr_status}")

    # Sharpe
    sh_str = f"{sharpe:.2f}" if sharpe is not None else f"N/A (need ≥{MIN_TRADES_SHARPE} trades, have {n})"
    sh_ok  = sharpe is not None and sharpe >= TARGET_SHARPE
    sh_status = _pass(sh_ok) if sharpe is not None else "PENDING"
    ref_sh = f"{BACKTEST_SHARPE:.2f}"
    print(f"  {'Sharpe ratio':<20} {sh_str:<12}  target ≥{TARGET_SHARPE:.1f}       ref {ref_sh:<8}  {sh_status}")

    # Max drawdown
    dd_str = f"{max_dd*100:.2f}%"
    dd_ok  = max_dd <= TARGET_MAX_DRAWDOWN
    print(f"  {'Max drawdown':<20} {dd_str:<12}  target <{TARGET_MAX_DRAWDOWN*100:.0f}%      ref 21.7%     {_pass(dd_ok)}")

    # Equity trend
    eq_ok = equity >= 0.90
    print(f"  {'Equity':<20} {equity:.4f}{'':8}  target ≥0.90      ref —         {_pass(eq_ok)}")

    print(f"  {'─'*56}")

    # ── Trade breakdown ───────────────────────────────────────────────────────
    if trades:
        print()
        print("  TRADE BREAKDOWN")
        up_trades = [t for t in trades if t["direction"] == "UP"]
        dn_trades = [t for t in trades if t["direction"] == "DOWN"]
        print(f"  {'Direction':<12}  {'Count':>5}  {'Wins':>5}  {'Win%':>7}  {'Avg net P&L':>12}")
        for label, subset in [("UP (BUY)", up_trades), ("DOWN (SELL)", dn_trades)]:
            if not subset:
                continue
            w = sum(t.get("won", 0) for t in subset)
            avg = np.mean([t.get("net_pnl", 0) for t in subset]) * 100
            print(f"  {label:<12}  {len(subset):>5}  {w:>5}  {w/len(subset)*100:>6.1f}%  {avg:>+11.3f}%")

        print()
        print("  EXIT REASONS")
        exit_counts: dict[str, int] = {}
        for t in trades:
            r = t.get("exit_reason", "HORIZON")
            exit_counts[r] = exit_counts.get(r, 0) + 1
        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason:<16}  {count:>3} trades")

        print()
        print("  RECENT TRADES  (last 5, newest first)")
        print(f"  {'Direction':<8}  {'Entry':>8}  {'Exit':>8}  {'Gross':>8}  {'Net':>8}  {'Result'}")
        for t in reversed(trades[-5:]):
            won_str = "WIN " if t.get("won") else "LOSS"
            print(f"  {t['direction']:<8}  {t['entry_price']:>8.4f}  {t['exit_price']:>8.4f}"
                  f"  {t['gross_pnl']*100:>+7.3f}%  {t['net_pnl']*100:>+7.3f}%  {won_str}")

    # ── Projection ────────────────────────────────────────────────────────────
    if trades_per_day and trades_per_day > 0:
        days_to_10 = max(0, (MIN_TRADES_SHARPE - n) / trades_per_day)
        print()
        print("  PROJECTION")
        print(f"  At {trades_per_day:.1f} trades/day → Sharpe reliable in ~{days_to_10:.0f} more days"
              f" ({MIN_TRADES_SHARPE - n} trades needed)")
        days_to_validation = 14 - (days_running or 0)
        if days_to_validation > 0:
            print(f"  2-week validation window ends in ~{days_to_validation:.0f} days")

    # ── Overall verdict ───────────────────────────────────────────────────────
    print()
    checks_with_data = []
    if win_rate is not None:
        checks_with_data.append(wr_ok)
    checks_with_data.append(dd_ok)
    checks_with_data.append(eq_ok)
    if sharpe is not None:
        checks_with_data.append(sh_ok)

    if not checks_with_data:
        verdict = "PENDING  — not enough trades yet"
    elif all(checks_with_data):
        verdict = "ON TRACK — all checks passing"
    elif sum(checks_with_data) >= len(checks_with_data) / 2:
        verdict = "WATCH    — some checks failing, monitor closely"
    else:
        verdict = "ALERT    — majority of checks failing, consider pausing"

    print(f"  VERDICT: {verdict}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live paper trading validation snapshot")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE,
                        help="Path to paper_trading_state_fusion.json")
    args = parser.parse_args()
    run(args.state)
