"""
evaluate.py — Compare PPO models on the held-out test set.

Usage:
    python services/models/rl/evaluate.py

Compares:
  1. Production PPO     — RL_MODEL_PATH       (53-dim obs, no pipeline signal)
  2. Aligned fusion PPO — RL_ALIGNED_MODEL_PATH (54-dim obs, xgb_prob_up)  [if exists]
  3. Always-HOLD baseline
  4. Buy-and-hold baseline
  5. XGBoost-signal strategy (BUY/SELL driven by xgb_prob_up threshold)

Outputs:
    services/models/classification/plots/rl_comparison.png  — equity curves
    Printed comparison table: return, Sharpe, drawdown, win-rate, trades,
                              action distribution, invalid-action count
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NamedTuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    FEATURE_COLUMNS_PATH,
    FUSION_SIGNAL_DIM,
    INITIAL_EQUITY,
    PLOTS_DIR,
    PPO_DEVICE,
    PROCESSED_CSV,
    RL_ALIGNED_MODEL_PATH,
    RL_ALIGNED_SCALER_PATH,
    RL_FUSION_MODEL_PATH,
    RL_FUSION_SCALER_PATH,
    RL_MODEL_PATH,
    RL_SCALER_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
    XGB_CONFIDENCE_THRESHOLD,
)
from environment import TradingEnv
from train_ppo import _max_drawdown, _sharpe, compute_fusion_signals, compute_pipeline_signals

from stable_baselines3 import PPO


# ── Result container ──────────────────────────────────────────────────────────

class RolloutResult(NamedTuple):
    label:          str
    equity_curve:   np.ndarray
    total_return:   float   # fractional, e.g. 0.12 = 12 %
    bh_return:      float
    sharpe:         float
    max_dd:         float
    win_rate:       float
    trades:         int
    hold_count:     int
    buy_count:      int
    sell_count:     int
    invalid_buy:    int
    invalid_sell:   int
    fees_paid:      float   # estimated total round-trip cost as fraction of equity


# ── Rollout helpers ───────────────────────────────────────────────────────────

def _win_rate(actions: list[int], prices: np.ndarray) -> float:
    """Fraction of completed round-trips with positive PnL."""
    entry_price = None
    results: list[bool] = []
    for i, a in enumerate(actions):
        p = float(prices[min(i, len(prices) - 1)])
        if a == 1 and entry_price is None:
            entry_price = p
        elif a == 2 and entry_price is not None:
            results.append(p > entry_price)
            entry_price = None
    return sum(results) / max(len(results), 1)


def _run_ppo_rollout(
    label: str,
    agent: PPO,
    features: np.ndarray,
    prices: np.ndarray,
    pipeline_signals: np.ndarray | None,
) -> RolloutResult:
    """Run one deterministic episode and return a RolloutResult."""
    env = TradingEnv(features, prices, pipeline_signals)
    obs, _ = env.reset()

    equity_curve = [env._equity]
    bh_start     = prices[0]
    actions: list[int] = []
    invalid_buy  = 0
    invalid_sell = 0

    while True:
        action, _ = agent.predict(obs, deterministic=True)
        action_i  = int(action)
        obs, _reward, terminated, truncated, info = env.step(action_i)
        equity_curve.append(info["equity"])
        actions.append(action_i)
        if info.get("invalid"):
            if action_i == 1:
                invalid_buy  += 1
            elif action_i == 2:
                invalid_sell += 1
        if terminated or truncated:
            break

    eq = np.array(equity_curve)
    log_rets     = np.diff(np.log(np.maximum(eq, 1e-10)))
    total_return = (eq[-1] - eq[0]) / eq[0]
    bh_return    = (prices[min(len(actions), len(prices) - 1)] - bh_start) / bh_start

    hold_count = actions.count(0)
    buy_count  = actions.count(1)
    sell_count = actions.count(2)
    trades     = buy_count + sell_count
    # Rough fee estimate: each valid trade costs ~_COST/2 applied to equity
    from config import FEE_RT, SLIPPAGE_PCT, TRADE_PENALTY
    cost_per_side = FEE_RT / 2 + SLIPPAGE_PCT / 2 + TRADE_PENALTY
    fees_paid = (buy_count - invalid_buy + sell_count - invalid_sell) * cost_per_side

    return RolloutResult(
        label        = label,
        equity_curve = eq,
        total_return = total_return,
        bh_return    = bh_return,
        sharpe       = _sharpe(log_rets),
        max_dd       = _max_drawdown(eq),
        win_rate     = _win_rate(actions, prices),
        trades       = trades,
        hold_count   = hold_count,
        buy_count    = buy_count,
        sell_count   = sell_count,
        invalid_buy  = invalid_buy,
        invalid_sell = invalid_sell,
        fees_paid    = fees_paid,
    )


def _run_xgb_strategy(
    prices: np.ndarray,
    features: np.ndarray,
    pipeline_signals: np.ndarray,
    threshold: float,
) -> RolloutResult:
    """Simulate a rule-based strategy: BUY when xgb_prob_up > threshold, SELL otherwise.
    Executed through TradingEnv so cost model is identical to PPO rollouts."""
    env = TradingEnv(features, prices, pipeline_signals)
    obs, _ = env.reset()

    equity_curve = [env._equity]
    bh_start     = prices[0]
    actions: list[int] = []
    invalid_buy  = 0
    invalid_sell = 0

    t = 0
    while True:
        prob_up = float(pipeline_signals[min(t, len(pipeline_signals) - 1), 0])
        position = env._position

        if prob_up > threshold and position == 0.0:
            action_i = 1   # BUY
        elif prob_up <= threshold and position == 1.0:
            action_i = 2   # SELL
        else:
            action_i = 0   # HOLD

        obs, _reward, terminated, truncated, info = env.step(action_i)
        equity_curve.append(info["equity"])
        actions.append(action_i)
        if info.get("invalid"):
            if action_i == 1:
                invalid_buy  += 1
            elif action_i == 2:
                invalid_sell += 1
        t += 1
        if terminated or truncated:
            break

    eq = np.array(equity_curve)
    log_rets     = np.diff(np.log(np.maximum(eq, 1e-10)))
    total_return = (eq[-1] - eq[0]) / eq[0]
    bh_return    = (prices[min(len(actions), len(prices) - 1)] - bh_start) / bh_start

    hold_count = actions.count(0)
    buy_count  = actions.count(1)
    sell_count = actions.count(2)
    trades     = buy_count + sell_count
    from config import FEE_RT, SLIPPAGE_PCT, TRADE_PENALTY
    cost_per_side = FEE_RT / 2 + SLIPPAGE_PCT / 2 + TRADE_PENALTY
    fees_paid = (buy_count - invalid_buy + sell_count - invalid_sell) * cost_per_side

    return RolloutResult(
        label        = f"XGBoost-signal (thr={threshold:.2f})",
        equity_curve = eq,
        total_return = total_return,
        bh_return    = bh_return,
        sharpe       = _sharpe(log_rets),
        max_dd       = _max_drawdown(eq),
        win_rate     = _win_rate(actions, prices),
        trades       = trades,
        hold_count   = hold_count,
        buy_count    = buy_count,
        sell_count   = sell_count,
        invalid_buy  = invalid_buy,
        invalid_sell = invalid_sell,
        fees_paid    = fees_paid,
    )


def _run_fusion_only_strategy(
    prices: np.ndarray,
    features: np.ndarray,
    fusion_signals: np.ndarray,
    buy_threshold: float = 0.0,
    sell_threshold: float = 0.0,
) -> RolloutResult:
    """Fusion-only strategy: BUY when fused_score > buy_threshold, SELL when < -sell_threshold.

    Uses TradingEnv (same cost model as PPO) with rule-based actions derived from
    the retroactive fused_score (index 1 in fusion_signals).
    Importable by other scripts — no duplicate cost logic.
    """
    from config import FUSION_WEIGHT_CLASSIFICATION  # already imported at top-level but explicit here
    env = TradingEnv(features, prices, fusion_signals)
    obs, _ = env.reset()
    equity_curve = [env._equity]
    bh_start     = prices[0]
    actions: list[int] = []
    invalid_buy = invalid_sell = 0

    t = 0
    while True:
        fused = float(fusion_signals[min(t, len(fusion_signals) - 1), 1])   # PIPELINE_IDX_FUSED=1
        pos   = env._position

        if fused > buy_threshold and pos == 0.0:
            action_i = 1   # BUY
        elif fused < -sell_threshold and pos == 1.0:
            action_i = 2   # SELL
        else:
            action_i = 0   # HOLD

        obs, _r, terminated, truncated, info = env.step(action_i)
        equity_curve.append(info["equity"])
        actions.append(action_i)
        if info.get("invalid"):
            if action_i == 1:
                invalid_buy  += 1
            elif action_i == 2:
                invalid_sell += 1
        t += 1
        if terminated or truncated:
            break

    eq           = np.array(equity_curve)
    log_rets     = np.diff(np.log(np.maximum(eq, 1e-10)))
    total_return = (eq[-1] - eq[0]) / eq[0]
    bh_return    = (prices[min(len(actions), len(prices) - 1)] - bh_start) / bh_start
    hold_count   = actions.count(0)
    buy_count    = actions.count(1)
    sell_count   = actions.count(2)
    trades       = buy_count + sell_count
    from config import FEE_RT, SLIPPAGE_PCT, TRADE_PENALTY
    cost_per_side = FEE_RT / 2 + SLIPPAGE_PCT / 2 + TRADE_PENALTY
    fees_paid = (buy_count - invalid_buy + sell_count - invalid_sell) * cost_per_side

    return RolloutResult(
        label        = f"Fusion-only (thr±{buy_threshold:.2f})",
        equity_curve = eq,
        total_return = total_return,
        bh_return    = bh_return,
        sharpe       = _sharpe(log_rets),
        max_dd       = _max_drawdown(eq),
        win_rate     = _win_rate(actions, prices),
        trades       = trades,
        hold_count   = hold_count,
        buy_count    = buy_count,
        sell_count   = sell_count,
        invalid_buy  = invalid_buy,
        invalid_sell = invalid_sell,
        fees_paid    = fees_paid,
    )


def _bh_result(label: str, equity_curve: np.ndarray, prices: np.ndarray) -> RolloutResult:
    """Buy-and-hold result (synthetic — no TradingEnv)."""
    log_rets     = np.diff(np.log(np.maximum(equity_curve, 1e-10)))
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    return RolloutResult(
        label        = label,
        equity_curve = equity_curve,
        total_return = total_return,
        bh_return    = total_return,
        sharpe       = _sharpe(log_rets),
        max_dd       = _max_drawdown(equity_curve),
        win_rate     = float("nan"),
        trades       = 0,
        hold_count   = len(equity_curve) - 1,
        buy_count    = 0,
        sell_count   = 0,
        invalid_buy  = 0,
        invalid_sell = 0,
        fees_paid    = 0.0,
    )


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_table(results: list[RolloutResult]) -> None:
    col_w = max(len(r.label) for r in results) + 2

    def row(label: str, fmt, getter) -> None:
        vals = "  ".join(fmt.format(getter(r)) for r in results)
        print(f"  {label:<30}  {vals}")

    header = "  ".join(f"{r.label:>{col_w}}" for r in results)
    print(f"\n  {'':30}  {header}")
    print(f"  {'─'*80}")
    row("Total return (%)",      f"{{:>{col_w}.2f}}",  lambda r: r.total_return * 100)
    row("Buy-and-hold (%)",      f"{{:>{col_w}.2f}}",  lambda r: r.bh_return * 100)
    row("Alpha vs BH (%)",       f"{{:>{col_w}.2f}}",  lambda r: (r.total_return - r.bh_return) * 100)
    row("Annualised Sharpe",     f"{{:>{col_w}.3f}}",  lambda r: r.sharpe)
    row("Max drawdown (%)",      f"{{:>{col_w}.2f}}",  lambda r: r.max_dd * 100)
    row("Win rate (%)",          f"{{:>{col_w}}}",     lambda r: f"{r.win_rate*100:.1f}" if not (isinstance(r.win_rate, float) and r.win_rate != r.win_rate) else "  n/a")
    row("Total trades",          f"{{:>{col_w},}}",    lambda r: r.trades)
    row("  BUY actions",         f"{{:>{col_w},}}",    lambda r: r.buy_count)
    row("  SELL actions",        f"{{:>{col_w},}}",    lambda r: r.sell_count)
    row("  HOLD actions",        f"{{:>{col_w},}}",    lambda r: r.hold_count)
    row("  Invalid BUY",         f"{{:>{col_w},}}",    lambda r: r.invalid_buy)
    row("  Invalid SELL",        f"{{:>{col_w},}}",    lambda r: r.invalid_sell)
    row("Est. fees (cost frac)", f"{{:>{col_w}.4f}}",  lambda r: r.fees_paid)
    row("Final equity ($)",      f"{{:>{col_w},.2f}}", lambda r: r.equity_curve[-1])
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("PPO Model Comparison — held-out test set")
    print("=" * 60)

    for path, label in [
        (PROCESSED_CSV,        "processed CSV"),
        (FEATURE_COLUMNS_PATH, "feature columns"),
        (RL_MODEL_PATH,        "production RL model"),
        (RL_SCALER_PATH,       "production RL scaler"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    df = pd.read_csv(PROCESSED_CSV)
    feature_cols: list[str] = json.loads(FEATURE_COLUMNS_PATH.read_text())
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    n       = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    print(f"  Test rows : {len(test_df):,}  (of {n:,} total)")

    # ── Pipeline signals — reuse existing builders, no duplication ───────────
    print("  Computing 1-dim pipeline signals (xgb_prob_up)…")
    pipeline_signals_all = compute_pipeline_signals(df, feature_cols)
    test_sig_1d = pipeline_signals_all[n_train + n_val:]
    print(f"  xgb_prob_up  min={test_sig_1d[:,0].min():.3f}  "
          f"max={test_sig_1d[:,0].max():.3f}  mean={test_sig_1d[:,0].mean():.3f}")

    print("  Computing 3-dim fusion signals (xgb_prob_up, fused_score, risk_encoded)…")
    fusion_signals_all = compute_fusion_signals(df, feature_cols)
    test_sig_3d = fusion_signals_all[n_train + n_val:]
    print(f"  fused_score  min={test_sig_3d[:,1].min():.3f}  "
          f"max={test_sig_3d[:,1].max():.3f}  mean={test_sig_3d[:,1].mean():.3f}")

    test_prices = test_df["close"].values

    # ── Production model (53-dim, no pipeline signal) ─────────────────────────
    prod_scaler   = joblib.load(RL_SCALER_PATH)
    test_feat_old = prod_scaler.transform(test_df[feature_cols].fillna(0.0).values)
    prod_agent    = PPO.load(str(RL_MODEL_PATH), device=PPO_DEVICE)
    print("\n  Running: Production PPO (53-dim)…")
    prod_result = _run_ppo_rollout(
        "Production PPO",
        prod_agent,
        test_feat_old,
        test_prices,
        pipeline_signals=None,   # legacy 53-dim obs
    )

    results: list[RolloutResult] = [prod_result]

    # ── Aligned model (54-dim: xgb_prob_up) — skip if not trained yet ─────────
    if RL_ALIGNED_MODEL_PATH.exists() and RL_ALIGNED_SCALER_PATH.exists():
        aligned_scaler = joblib.load(RL_ALIGNED_SCALER_PATH)
        test_feat_al   = aligned_scaler.transform(test_df[feature_cols].fillna(0.0).values)
        aligned_agent  = PPO.load(str(RL_ALIGNED_MODEL_PATH), device=PPO_DEVICE)
        print("  Running: Aligned PPO (54-dim)…")
        results.append(_run_ppo_rollout(
            "Aligned PPO",
            aligned_agent,
            test_feat_al,
            test_prices,
            pipeline_signals=test_sig_1d,
        ))
    else:
        print(f"  Aligned model not found ({RL_ALIGNED_MODEL_PATH.name}) — skipping.")

    # ── Fusion→PPO v2 (56-dim: xgb_prob_up + fused_score + risk_encoded) ─────
    if RL_FUSION_MODEL_PATH.exists() and RL_FUSION_SCALER_PATH.exists():
        fusion_scaler = joblib.load(RL_FUSION_SCALER_PATH)
        test_feat_fu  = fusion_scaler.transform(test_df[feature_cols].fillna(0.0).values)
        fusion_agent  = PPO.load(str(RL_FUSION_MODEL_PATH), device=PPO_DEVICE)
        print("  Running: Fusion→PPO v2 (56-dim)…")
        results.append(_run_ppo_rollout(
            "Fusion→PPO v2",
            fusion_agent,
            test_feat_fu,
            test_prices,
            pipeline_signals=test_sig_3d,
        ))
    else:
        print(f"  Fusion v2 model not found ({RL_FUSION_MODEL_PATH.name}) — skipping.")

    # ── Always-HOLD baseline ──────────────────────────────────────────────────
    hold_equity = np.full(len(test_prices) + 1, INITIAL_EQUITY, dtype=np.float64)
    results.append(RolloutResult(
        label        = "Always-HOLD",
        equity_curve = hold_equity,
        total_return = 0.0,
        bh_return    = (test_prices[-1] - test_prices[0]) / test_prices[0],
        sharpe       = 0.0,
        max_dd       = 0.0,
        win_rate     = float("nan"),
        trades       = 0,
        hold_count   = len(test_prices),
        buy_count    = 0,
        sell_count   = 0,
        invalid_buy  = 0,
        invalid_sell = 0,
        fees_paid    = 0.0,
    ))

    # ── Buy-and-hold baseline ─────────────────────────────────────────────────
    bh_equity = INITIAL_EQUITY * (test_prices / test_prices[0])
    results.append(_bh_result("Buy-and-hold", bh_equity, test_prices))

    # ── XGBoost-signal strategy ───────────────────────────────────────────────
    print("  Running: XGBoost-signal strategy…")
    results.append(_run_xgb_strategy(
        test_prices,
        test_feat_old,
        test_sig_1d,
        threshold=XGB_CONFIDENCE_THRESHOLD,
    ))

    # ── Fusion-only strategy ──────────────────────────────────────────────────
    print("  Running: Fusion-only strategy…")
    from config import FUSION_WEIGHT_CLASSIFICATION  # imported from fusion/config via rl/config
    # Threshold = fraction of the classification weight so that at XGBoost thr=0.69,
    # the continuous fused_score contribution alone exceeds the BUY threshold.
    fusion_thr = 0.0   # use any positive fused_score as BUY signal (softest threshold)
    results.append(_run_fusion_only_strategy(
        test_prices,
        test_feat_old,
        test_sig_3d,
        buy_threshold=fusion_thr,
        sell_threshold=fusion_thr,
    ))

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON TABLE — Test Set")
    print("=" * 60)
    _print_table(results)

    # ── Equity curve plot ─────────────────────────────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    styles = [
        {"linewidth": 2.0,  "linestyle": "-",   "color": "steelblue"},
        {"linewidth": 2.0,  "linestyle": "-",   "color": "darkorange"},
        {"linewidth": 1.0,  "linestyle": ":",   "color": "gray"},
        {"linewidth": 1.5,  "linestyle": "--",  "color": "black"},
        {"linewidth": 1.5,  "linestyle": "-.",  "color": "green"},
    ]
    for i, r in enumerate(results):
        style = styles[i % len(styles)]
        ax.plot(r.equity_curve, label=r.label, **style)

    ax.set_title("PPO Model Comparison — Test Set Equity Curves")
    ax.set_xlabel("Step (15-min bars)")
    ax.set_ylabel("Equity (USD)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plot_path = PLOTS_DIR / "rl_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Equity curve saved → {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
