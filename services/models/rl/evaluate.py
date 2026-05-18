"""
evaluate.py — Backtest the trained PPO agent on the held-out test set.

Usage:
    python services/models/rl/evaluate.py

Outputs:
    services/models/classification/plots/rl_equity_curve.png
    Printed metrics: total return, Sharpe, max drawdown, win rate, trades
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    PLOTS_DIR,
    PPO_DEVICE,
    PROCESSED_CSV,
    RL_MODEL_PATH,
    RL_SCALER_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
)
from environment import TradingEnv

from stable_baselines3 import PPO


def _sharpe(returns: np.ndarray, periods_per_year: int = 35_040) -> float:
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(((peak - equity) / np.where(peak > 0, peak, 1)).max())


def main() -> None:
    print("=" * 60)
    print("PPO Evaluation — held-out test set")
    print("=" * 60)

    for path, label in [(RL_MODEL_PATH, "model"), (RL_SCALER_PATH, "scaler"),
                        (PROCESSED_CSV, "data"), (FEATURE_COLUMNS_PATH, "features")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}\nRun train_ppo.py first.")

    df = pd.read_csv(PROCESSED_CSV)
    feature_cols: list[str] = json.loads(FEATURE_COLUMNS_PATH.read_text())
    scaler = joblib.load(RL_SCALER_PATH)

    n       = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    print(f"  Test rows: {len(test_df):,}")

    test_features = scaler.transform(test_df[feature_cols].fillna(0.0).values)
    test_prices   = test_df["close"].values

    agent = PPO.load(str(RL_MODEL_PATH), device=PPO_DEVICE)
    env   = TradingEnv(test_features, test_prices)
    obs, _ = env.reset()

    equity_curve  = [env._equity]
    step_returns  = []
    actions_taken = []

    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        equity_curve.append(info["equity"])
        step_returns.append(reward)
        actions_taken.append(int(action))
        if terminated or truncated:
            break

    equity_curve  = np.array(equity_curve)
    step_returns  = np.array(step_returns)
    buy_hold      = test_prices / test_prices[0] * equity_curve[0]

    total_return  = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    bh_return     = (test_prices[-1] - test_prices[0]) / test_prices[0]
    trades        = sum(1 for a in actions_taken if a in (1, 2))

    # Sharpe from equity log-returns (not reward signal) — correct annualisation
    eq_log_returns = np.diff(np.log(np.where(equity_curve > 0, equity_curve, 1e-9)))
    sharpe = _sharpe(eq_log_returns)

    # Win rate: fraction of completed round-trips with positive PnL
    entry_price = None
    round_trip_results = []
    for i, a in enumerate(actions_taken):
        if a == 1 and entry_price is None:
            entry_price = test_prices[min(i, len(test_prices) - 1)]
        elif a == 2 and entry_price is not None:
            exit_price = test_prices[min(i, len(test_prices) - 1)]
            round_trip_results.append(exit_price > entry_price)
            entry_price = None
    win_rate = sum(round_trip_results) / max(len(round_trip_results), 1)

    print(f"\n  {'Metric':<28} {'Agent':>10}   {'Buy-and-hold':>12}")
    print(f"  {'─'*52}")
    print(f"  {'Total return':<28} {total_return*100:>9.2f}%   {bh_return*100:>11.2f}%")
    print(f"  {'Annualised Sharpe':<28} {sharpe:>10.3f}")
    print(f"  {'Max drawdown':<28} {_max_drawdown(equity_curve)*100:>9.2f}%")
    print(f"  {'Win rate (round-trips)':<28} {win_rate*100:>9.2f}%")
    print(f"  {'Trade count':<28} {trades:>10,}")
    print(f"  {'Final equity':<28} ${equity_curve[-1]:>9,.2f}")

    # ── Equity curve plot ──────────────────────────────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity_curve, label="PPO Agent", linewidth=1.5)
    ax.plot(buy_hold[:len(equity_curve)], label="Buy & Hold", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title("PPO Agent vs Buy-and-Hold — Test Set")
    ax.set_xlabel("Step (15-min bars)")
    ax.set_ylabel("Equity (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = PLOTS_DIR / "rl_equity_curve.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Equity curve saved → {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
