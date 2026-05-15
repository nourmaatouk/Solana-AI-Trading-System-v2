"""
realtime_paper_trader.py

Paper trading simulator. Run after run_realtime_pipeline.py (or pass
--paper-trade to the pipeline runner).

Each run:
  1. Reads the latest fused decision from last_fused_decision.json.
     Falls back to legacy last_fusion_signal.json or XGBoost last_signal.json
     only when the fused-decision contract is unavailable.
  2. Checks any open position for early exits:
       STOP_LOSS   — gross P&L <= −STOP_LOSS_PCT (−1.5 %)
       TAKE_PROFIT — gross P&L >= +TAKE_PROFIT_PCT (+2.0 %)
       HORIZON     — exit deadline reached
  3. Opens a new position if the signal is UP/DOWN and no position is open.

Position sizing scales with model confidence:
  size = MIN_POSITION_SIZE + (1 − MIN_POSITION_SIZE) × confidence_factor
  where confidence_factor ∈ [0, 1] measures how far P_final exceeds the gate.
  At the gate boundary the bet is MIN_POSITION_SIZE (25 %) of equity;
  at maximum confidence (P=1 or P=0) the full equity is deployed.

Fee (0.1 %) is applied on the deployed fraction, so:
  equity_change = (gross − FEE_RT) × position_size

State is persisted in data/realtime/paper_trading_state*.json between runs.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from config import (
    ACTIVE_HORIZON,
    CONFIDENCE_THRESHOLD,
    MIN_POSITION_SIZE,
    REALTIME_DIR,
    STOP_LOSS_PCT,
    SYMBOL,
    TAKE_PROFIT_PCT,
)

# XGBoost-only signal (always present after the predictor step)
XGB_SIGNAL_PATH    = REALTIME_DIR / "last_signal.json"
SPOOFING_PATH      = REALTIME_DIR / "spoofing_score.json"

# Suppress new entries when orderbook shows self-exciting (spoofing) activity
SPOOFING_MAX_AGE_MIN = 30   # ignore spoofing score older than this
# Unified fused decision (preferred downstream contract for paper trading)
FUSED_DECISION_PATH = REALTIME_DIR / "last_fused_decision.json"

# Legacy fusion signal (present only when --fusion was passed to the pipeline)
FUSION_SIGNAL_PATH = REALTIME_DIR / "last_fusion_signal.json"

# Separate state/log per signal source so equity curves are comparable
XGB_STATE_PATH    = REALTIME_DIR / "paper_trading_state.json"
XGB_LOG_PATH      = REALTIME_DIR / "paper_trading_log.json"
FUSION_STATE_PATH   = REALTIME_DIR / "paper_trading_state_fusion.json"
FUSION_LOG_PATH     = REALTIME_DIR / "paper_trading_log_fusion.json"
HISTORY_PATH        = REALTIME_DIR / "paper_trading_history.json"
HISTORY_MAX_ENTRIES = 200

FEE_RT  = 0.001
HORIZON = ACTIVE_HORIZON

# FusedDecision uses fused_score in [-1, 1] and BUY/SELL thresholds +/-0.3.
# Mapping score -> pseudo P(up) = (score + 1) / 2 means BUY starts at 0.65.
FUSED_SCORE_ENTRY_THRESHOLD = 0.30
FUSED_PROB_GATE = (1.0 + FUSED_SCORE_ENTRY_THRESHOLD) / 2.0

# New entries are blocked when the market/signal timestamp is older than this.
# Exits are still evaluated so stale signals cannot trap already-open positions.
MAX_SIGNAL_AGE_MINUTES = int(os.getenv("PAPER_TRADER_MAX_SIGNAL_AGE_MINUTES", "90"))


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _parse_datetime(value) -> datetime | None:
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _classification_context() -> dict:
    """Return price/timestamp context from the latest XGBoost signal if present."""
    if not XGB_SIGNAL_PATH.exists():
        return {}
    try:
        raw = _read_json(XGB_SIGNAL_PATH)
    except Exception:
        return {}
    if raw.get("status") != "ok":
        return {}
    return {
        "asset": raw.get("asset", SYMBOL),
        "latest_timestamp": raw.get("latest_timestamp") or raw.get("latest_utc"),
        "latest_close": raw.get("latest_close"),
        "confidence_threshold": raw.get("confidence_threshold", CONFIDENCE_THRESHOLD),
    }


def _action_to_direction(action: str) -> str:
    action = str(action).upper()
    if action in ("BUY", "UP"):
        return "UP"
    if action in ("SELL", "DOWN"):
        return "DOWN"
    return "NEUTRAL"


def _direction_to_action(direction: str) -> str:
    direction = str(direction).upper()
    if direction == "UP":
        return "BUY"
    if direction == "DOWN":
        return "SELL"
    return "HOLD"


def _normalise_fused_decision(raw: dict) -> dict:
    """Adapt last_fused_decision.json into the paper-trader signal shape."""
    ctx = _classification_context()
    action = str(raw.get("action", "HOLD")).upper()
    score = float(raw.get("fused_score", 0.0))
    probability_up = raw.get("probability_up")
    if probability_up is None:
        probability_up = (score + 1.0) / 2.0

    reasoning = raw.get("reason") or raw.get("reasoning", {})
    if isinstance(reasoning, dict):
        reason = " | ".join(str(v) for v in reasoning.values()) or "FUSED_DECISION"
    else:
        reason = str(reasoning)

    return {
        "status": raw.get("status"),
        "asset": raw.get("asset", ctx.get("asset", SYMBOL)),
        "signal": _action_to_direction(action),
        "action": action,
        "reason": reason,
        "probability_up": float(probability_up),
        "score": score,
        "risk_level": raw.get("risk_level", "UNKNOWN"),
        "latest_timestamp": (
            raw.get("latest_timestamp")
            or raw.get("signal_timestamp")
            or ctx.get("latest_timestamp")
            or raw.get("generated_at_utc")
        ),
        "latest_close": raw.get("latest_close", ctx.get("latest_close")),
        "confidence_threshold": raw.get("confidence_threshold", FUSED_PROB_GATE),
        "source_contract": "last_fused_decision.json",
    }


def _normalise_probability_signal(raw: dict, *, source_contract: str) -> dict:
    """Adapt legacy probability signals into the same shape as fused decisions."""
    signal = str(raw.get("signal", "NEUTRAL")).upper()
    probability_up = float(raw.get("probability_up", 0.5))
    return {
        "status": raw.get("status"),
        "asset": raw.get("asset", SYMBOL),
        "signal": signal,
        "action": _direction_to_action(signal),
        "reason": raw.get("reason", source_contract),
        "probability_up": probability_up,
        "score": raw.get("fused_score"),
        "risk_level": raw.get("risk_level", "UNKNOWN"),
        "latest_timestamp": raw.get("latest_timestamp") or raw.get("latest_utc"),
        "latest_close": raw.get("latest_close"),
        "confidence_threshold": raw.get("confidence_threshold", CONFIDENCE_THRESHOLD),
        "source_contract": source_contract,
    }


def _signal_age_minutes(timestamp_raw, now_utc: datetime) -> float | None:
    ts = _parse_datetime(timestamp_raw)
    if ts is None:
        return None
    return (now_utc - ts).total_seconds() / 60.0


# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────────────────────────────────────
def _is_market_spoofed() -> bool:
    """Return True if the latest spoofing score flags suspicious activity.

    Returns False (allow trading) when the score file is missing, stale, or
    the Hawkes fit produced no result — fail-open so the gate only suppresses
    when evidence is present.
    """
    if not SPOOFING_PATH.exists():
        return False
    try:
        with open(SPOOFING_PATH) as f:
            score = json.load(f)
        if score.get("status") not in ("ok",):
            return False
        # Check freshness
        generated = score.get("generated_at_utc")
        if generated:
            gen_dt  = datetime.fromisoformat(generated)
            age_min = (datetime.now(timezone.utc) - gen_dt).total_seconds() / 60.0
            if age_min > SPOOFING_MAX_AGE_MIN:
                return False
        return bool(score.get("is_suspicious", False))
    except Exception:
        return False


def _position_size(prob_up: float, direction: str, gate: float) -> float:
    """Return fraction of equity to deploy, scaled by distance from the gate.

    At the gate boundary (prob_up == gate for UP, prob_up == 1-gate for DOWN)
    the size is MIN_POSITION_SIZE.  At maximum confidence (prob_up == 1 or 0)
    the full equity is deployed.
    """
    if direction == "UP":
        excess = prob_up - gate
    else:
        excess = (1.0 - gate) - prob_up

    max_excess = 1.0 - gate
    if max_excess <= 0:
        return 1.0

    factor = min(max(excess / max_excess, 0.0), 1.0)
    return MIN_POSITION_SIZE + (1.0 - MIN_POSITION_SIZE) * factor


# ─────────────────────────────────────────────────────────────────────────────
# EXIT LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def _exit_reason(pos: dict, current_close: float, now_utc: datetime) -> str | None:
    """Return 'STOP_LOSS', 'TAKE_PROFIT', 'HORIZON', or None (still holding)."""
    entry = pos["entry_price"]
    raw   = (current_close - entry) / entry
    gross = raw if pos["direction"] == "UP" else -raw

    if gross <= -STOP_LOSS_PCT:
        return "STOP_LOSS"
    if gross >= TAKE_PROFIT_PCT:
        return "TAKE_PROFIT"
    deadline = _parse_datetime(pos.get("exit_deadline"))
    if deadline is not None and now_utc >= deadline:
        return "HORIZON"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _empty_state() -> dict:
    return {
        "equity":                   1.0,
        "peak_equity":              1.0,
        "n_trades":                 0,
        "n_wins":                   0,
        "open_position":            None,
        "last_processed_timestamp": None,
        "trade_log":                [],
        "equity_curve":             [1.0],
    }


def load_state(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return _empty_state()


def save_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(state: dict) -> dict:
    n      = state["n_trades"]
    n_wins = state["n_wins"]
    eq     = state["equity"]
    curve  = np.array(state["equity_curve"])

    win_rate  = n_wins / n if n > 0 else 0.0
    total_ret = eq - 1.0

    peak   = 1.0
    max_dd = 0.0
    for v in curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    net_rets = [t["net_pnl"] for t in state["trade_log"]]
    if len(net_rets) >= 2:
        arr    = np.array(net_rets)
        sharpe = arr.mean() / (arr.std() + 1e-12) * np.sqrt(252 * 96)
    else:
        sharpe = 0.0

    exit_counts: dict[str, int] = {}
    for t in state["trade_log"]:
        reason = t.get("exit_reason", "HORIZON")
        exit_counts[reason] = exit_counts.get(reason, 0) + 1

    return {
        "n_trades":     n,
        "win_rate":     round(win_rate, 4),
        "total_return": round(total_ret, 6),
        "max_drawdown": round(max_dd, 6),
        "sharpe":       round(float(sharpe), 4),
        "equity":       round(eq, 6),
        "exit_counts":  exit_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_signal() -> tuple[dict, Path, Path, str]:
    """Return (sig_dict, state_path, log_path, source_label).

    Prefers the unified fused decision; falls back to legacy fusion and then
    XGBoost only for backward compatibility.
    """
    if FUSED_DECISION_PATH.exists():
        raw = _read_json(FUSED_DECISION_PATH)
        if raw.get("status") == "ok":
            return (
                _normalise_fused_decision(raw),
                FUSION_STATE_PATH,
                FUSION_LOG_PATH,
                "FUSED_DECISION",
            )

    if FUSION_SIGNAL_PATH.exists():
        raw = _read_json(FUSION_SIGNAL_PATH)
        if raw.get("status") == "ok":
            sig = _normalise_probability_signal(
                raw, source_contract="last_fusion_signal.json"
            )
            return sig, FUSION_STATE_PATH, FUSION_LOG_PATH, "LEGACY_FUSION"

    if not XGB_SIGNAL_PATH.exists():
        return {}, XGB_STATE_PATH, XGB_LOG_PATH, "XGB"

    raw = _read_json(XGB_SIGNAL_PATH)
    return (
        _normalise_probability_signal(raw, source_contract="last_signal.json"),
        XGB_STATE_PATH,
        XGB_LOG_PATH,
        "XGB",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 60)
    print("PAPER TRADER")
    print("=" * 60)

    # ── 1. Load signal ────────────────────────────────────────────────────────
    sig, state_path, log_path, source = _resolve_signal()

    if not sig:
        print("  No signal file found. Run the pipeline first.")
        return 1

    if sig.get("status") != "ok":
        print(f"  Signal status={sig.get('status')} — skipping.")
        return 0

    signal        = sig["signal"]
    action        = sig.get("action", _direction_to_action(signal))
    prob_up       = float(sig.get("probability_up", 0.5))
    score         = sig.get("score")
    risk_level    = sig.get("risk_level", "UNKNOWN")
    reason_text   = sig.get("reason", "")
    asset         = sig.get("asset", SYMBOL)
    sig_timestamp = sig.get("latest_timestamp")
    latest_close  = sig.get("latest_close")
    gate          = float(sig.get("confidence_threshold", CONFIDENCE_THRESHOLD))

    if latest_close is None:
        print("  Signal has no latest_close/price context — cannot paper trade safely.")
        return 1
    current_close = float(latest_close)

    now_utc        = datetime.now(timezone.utc)
    age_min        = _signal_age_minutes(sig_timestamp, now_utc)
    entry_block_reason = None
    if age_min is None:
        entry_block_reason = "signal timestamp is missing or unparseable"
    elif age_min > MAX_SIGNAL_AGE_MINUTES:
        entry_block_reason = (
            f"signal is stale: age={age_min:.1f} min "
            f"> max={MAX_SIGNAL_AGE_MINUTES} min"
        )

    print(f"  Source    : {source} ({sig.get('source_contract', 'unknown contract')})")
    print(f"  Asset     : {asset}")
    print(f"  Action    : {action}  (direction={signal}, P_up={prob_up:.4f})")
    if score is not None:
        print(f"  Score     : {float(score):+.4f}")
    print(f"  Risk      : {risk_level}")
    if reason_text:
        print(f"  Reason    : {reason_text}")
    print(f"  Candle    : {sig_timestamp}  close={current_close}")
    if age_min is not None:
        print(f"  Age       : {age_min:.1f} min  (max={MAX_SIGNAL_AGE_MINUTES})")

    # ── 2. Load state ─────────────────────────────────────────────────────────
    state = load_state(state_path)

    already_processed = state["last_processed_timestamp"] == sig_timestamp
    if already_processed:
        processed_reason = f"already processed candle {sig_timestamp}"
        entry_block_reason = (
            f"{entry_block_reason}; {processed_reason}"
            if entry_block_reason else processed_reason
        )

    pos             = state["open_position"]
    closed_this_run = False

    # ── 3. Check open position for SL / TP / Horizon exit ────────────────────
    if pos is not None:
        reason = _exit_reason(pos, current_close, now_utc)

        if reason is not None:
            entry = pos["entry_price"]
            raw   = (current_close - entry) / entry
            gross = raw if pos["direction"] == "UP" else -raw
            size  = pos.get("position_size", 1.0)
            net   = (gross - FEE_RT) * size

            state["equity"] = round(state["equity"] * (1 + net), 8)
            if state["equity"] > state["peak_equity"]:
                state["peak_equity"] = state["equity"]

            state["n_trades"] += 1
            won = int(gross > 0)
            state["n_wins"]  += won
            state["equity_curve"].append(state["equity"])

            trade_record = {
                "direction":     pos["direction"],
                "entry_price":   entry,
                "exit_price":    current_close,
                "entry_time":    pos["entry_time"],
                "exit_time":     now_utc.isoformat(),
                "gross_pnl":     round(gross, 6),
                "net_pnl":       round(net, 6),
                "position_size": round(size, 4),
                "prob":          pos["prob"],
                "won":           won,
                "exit_reason":   reason,
                "equity_after":  state["equity"],
            }
            state["trade_log"].append(trade_record)
            state["open_position"] = None
            closed_this_run = True

            icons = {"STOP_LOSS": "✗ SL", "TAKE_PROFIT": "✓ TP", "HORIZON": "⏱ HZ"}
            tag   = ("WIN " if won else "LOSS ") + icons.get(reason, reason)
            print(f"\n  CLOSED  [{tag}]")
            print(f"    Direction     : {pos['direction']}")
            print(f"    Entry → Exit  : {entry:.4f}  →  {current_close:.4f}")
            print(f"    Gross P&L     : {gross*100:+.3f}%   Net (on equity): {net*100:+.3f}%")
            print(f"    Position size : {size*100:.0f}% of equity")
            print(f"    Exit reason   : {reason}")
            print(f"    Equity        : {state['equity']:.6f}")

    # ── 4. Open new position if signal fires ──────────────────────────────────
    spoofed = _is_market_spoofed()
    if spoofed:
        print("\n  SPOOFING GATE — new entry suppressed (suspicious orderbook activity).")

    if (
        state["open_position"] is None
        and signal in ("UP", "DOWN")
        and not spoofed
        and entry_block_reason is None
    ):
        size          = _position_size(prob_up, signal, gate)
        entry_price   = current_close
        entry_time    = now_utc.isoformat()
        exit_deadline = (now_utc + timedelta(minutes=HORIZON * 15)).isoformat()

        state["open_position"] = {
            "direction":     signal,
            "entry_price":   entry_price,
            "entry_time":    entry_time,
            "exit_deadline": exit_deadline,
            "prob":          prob_up,
            "source":        source,
            "asset":         asset,
            "risk_level":    risk_level,
            "score":         score,
            "position_size": round(size, 4),
        }
        print(f"\n  OPENED position")
        print(f"    Direction     : {signal}")
        print(f"    Entry         : {entry_price:.4f}")
        print(f"    Exit by       : {exit_deadline}")
        print(f"    P(up)         : {prob_up:.4f}")
        print(f"    Position size : {size*100:.0f}% of equity  "
              f"(SL ={STOP_LOSS_PCT*100:.1f}%  TP={TAKE_PROFIT_PCT*100:.1f}%)")

    elif state["open_position"] is None and entry_block_reason is not None:
        print(f"\n  NEW ENTRY BLOCKED — {entry_block_reason}")

    elif state["open_position"] is not None and not closed_this_run:
        pos   = state["open_position"]
        raw   = (current_close - pos["entry_price"]) / pos["entry_price"]
        unreal = raw if pos["direction"] == "UP" else -raw
        size  = pos.get("position_size", 1.0)
        print(f"\n  HOLDING  {pos['direction']} @ {pos['entry_price']:.4f}  "
              f"(size={size*100:.0f}%)")
        print(f"    Exit deadline : {pos['exit_deadline']}")
        print(f"    Unrealised    : {unreal*100:+.3f}%  "
              f"| SL in {(-STOP_LOSS_PCT - unreal)*100:.2f}%  "
              f"TP in {(TAKE_PROFIT_PCT - unreal)*100:.2f}%")

    elif signal == "NEUTRAL":
        print("\n  NEUTRAL — no trade.")

    # ── 5. Save state ─────────────────────────────────────────────────────────
    state["last_processed_timestamp"] = sig_timestamp
    save_state(state, state_path)

    # ── 6. Print metrics ──────────────────────────────────────────────────────
    metrics = compute_metrics(state)
    print()
    _print_metrics(metrics, state)

    # ── 7. Save log ───────────────────────────────────────────────────────────
    log = {
        "status":              "ok",
        "last_run_utc":        now_utc.isoformat(),
        "signal_source":       source,
        "source_contract":     sig.get("source_contract"),
        "asset":               asset,
        "signal_timestamp":    sig_timestamp,
        "signal_age_minutes":  round(age_min, 4) if age_min is not None else None,
        "entry_block_reason":  entry_block_reason,
        "risk_level":          risk_level,
        "decision_score":      score,
        "reason":              reason_text,
        "metrics":             metrics,
        "open_position":       state["open_position"],
        "last_signal":         signal,
        "last_action":         action,
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # ── 8. Append to rolling history ──────────────────────────────────────────
    history_entry = {
        "run_utc":      now_utc.isoformat(),
        "signal":       signal,
        "action":       action,
        "fused_score":  score,
        "risk_level":   risk_level,
        "equity":       metrics.get("equity"),
        "total_return": metrics.get("total_return"),
        "n_trades":     metrics.get("n_trades"),
        "win_rate":     metrics.get("win_rate"),
        "sharpe":       metrics.get("sharpe"),
        "latest_close": sig.get("latest_close"),
    }
    try:
        history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else []
        history.append(history_entry)
        if len(history) > HISTORY_MAX_ENTRIES:
            history = history[-HISTORY_MAX_ENTRIES:]
        HISTORY_PATH.write_text(json.dumps(history, indent=2))
    except Exception:
        pass  # never crash the pipeline over history logging

    print(f"\n  Source: {source}  |  State: {state_path.name}  |  Log: {log_path.name}")
    print("=" * 60)
    return 0


def _print_metrics(metrics: dict, state: dict) -> None:
    n   = metrics["n_trades"]
    pos = state["open_position"]
    ec  = metrics.get("exit_counts", {})
    print("  ── Running metrics ──────────────────────────────────")
    print(f"  Trades       : {n}")
    print(f"  Win rate     : {metrics['win_rate']*100:.1f}%" if n > 0 else "  Win rate     : —")
    print(f"  Total return : {metrics['total_return']*100:+.2f}%")
    print(f"  Max drawdown : {metrics['max_drawdown']*100:.2f}%")
    print(f"  Sharpe       : {metrics['sharpe']:.2f}" if n >= 2 else "  Sharpe       : — (need ≥2 trades)")
    print(f"  Equity       : {metrics['equity']:.6f}")
    if ec:
        parts = "  ".join(f"{k}={v}" for k, v in sorted(ec.items()))
        print(f"  Exit types   : {parts}")
    if pos:
        print(f"  Open pos     : {pos['direction']} @ {pos['entry_price']:.4f}  "
              f"size={pos.get('position_size', 1.0)*100:.0f}%")
    else:
        print("  Open pos     : none")
    print("  ─────────────────────────────────────────────────────")


def reset(fusion: bool = False) -> None:
    """Reset paper trading state.  Pass fusion=True to reset the fusion state."""
    state_path = FUSION_STATE_PATH if fusion else XGB_STATE_PATH
    log_path   = FUSION_LOG_PATH   if fusion else XGB_LOG_PATH
    label      = "fusion" if fusion else "xgb"
    for p in (state_path, log_path):
        if p.exists():
            p.unlink()
    print(f"Paper trading state reset ({label}).")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Wipe paper trading state and start fresh.")
    parser.add_argument("--reset-fusion", action="store_true",
                        help="Wipe fusion paper trading state and start fresh.")
    args = parser.parse_args()

    if args.reset:
        reset(fusion=False)
    elif args.reset_fusion:
        reset(fusion=True)
    else:
        raise SystemExit(main())
