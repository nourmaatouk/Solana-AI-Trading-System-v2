"""
train_ppo.py — Train a PPO agent on historical SOL/USDT 15-minute data.

Usage:
    python services/models/rl/train_ppo.py

Outputs:
    services/models/models/ppo_trading_agent.zip   — best checkpoint
    services/models/models/rl_scaler.pkl           — fitted StandardScaler
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    FEATURE_COLUMNS_PATH,
    MODEL_DIR,
    PLOTS_DIR,
    PROCESSED_CSV,
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
    PPO_DEVICE,
    PPO_NET_ARCH,
    PPO_TOTAL_TIMESTEPS,
    PPO_EVAL_FREQ_MIN,
    RL_MODEL_PATH,
    RL_SCALER_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
)
from environment import TradingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray, periods_per_year: int = 35040) -> float:
    """Annualised Sharpe from per-step log returns (15-min = 35 040 bars/year)."""
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd   = (peak - equity_curve) / np.where(peak > 0, peak, 1)
    return float(dd.max())


def evaluate_agent(agent: PPO, features: np.ndarray, prices: np.ndarray) -> dict:
    env = TradingEnv(features, prices)
    obs, _ = env.reset()
    equity_curve    = [env._equity]
    bh_equity_curve = [env._bh_equity]
    trades = 0

    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(int(action))
        equity_curve.append(info["equity"])
        bh_equity_curve.append(info["bh_equity"])
        if action in (1, 2):
            trades += 1
        if terminated or truncated:
            break

    equity_curve    = np.array(equity_curve)
    bh_equity_curve = np.array(bh_equity_curve)

    # Sharpe from actual log-returns of equity curve (not shaped rewards)
    log_rets    = np.diff(np.log(np.maximum(equity_curve, 1e-10)))
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    bh_return    = (bh_equity_curve[-1] - bh_equity_curve[0]) / bh_equity_curve[0]

    return {
        "total_return_pct":    round(total_return * 100, 2),
        "bh_return_pct":       round(bh_return * 100, 2),
        "alpha_pct":           round((total_return - bh_return) * 100, 2),
        "sharpe":              round(_sharpe(log_rets), 3),
        "max_drawdown_pct":    round(_max_drawdown(equity_curve) * 100, 2),
        "trades":              trades,
        "final_equity":        round(equity_curve[-1], 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("PPO Training — SOL/USDT 15m")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(f"Processed CSV not found: {PROCESSED_CSV}")
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature columns not found: {FEATURE_COLUMNS_PATH}")

    df = pd.read_csv(PROCESSED_CSV)
    feature_cols: list[str] = json.loads(FEATURE_COLUMNS_PATH.read_text())
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if "close" not in df.columns:
        raise ValueError("Column 'close' not found — needed for price series.")

    print(f"  Rows: {len(df):,}   Features: {len(feature_cols)}")

    # ── Chronological split ────────────────────────────────────────────────────
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df   = df.iloc[n_train : n_train + n_val].reset_index(drop=True)

    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  "
          f"Test: {n - n_train - n_val:,}")

    # ── Fit scaler on train only ───────────────────────────────────────────────
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[feature_cols].fillna(0.0).values)
    val_features   = scaler.transform(val_df[feature_cols].fillna(0.0).values)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, RL_SCALER_PATH)
    print(f"  Scaler saved → {RL_SCALER_PATH.name}")

    # ── Build environments ─────────────────────────────────────────────────────
    # random_start=True exposes the agent to all market regimes (bull, bear,
    # recovery) rather than always starting from the beginning of the training
    # slice — critical when the slice contains regime changes.
    train_env = Monitor(TradingEnv(train_features, train_df["close"].values, random_start=True))
    val_env   = Monitor(TradingEnv(val_features,   val_df["close"].values))

    # ── Train ──────────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(MODEL_DIR),
        log_path=str(MODEL_DIR),
        eval_freq=max(PPO_N_STEPS, PPO_EVAL_FREQ_MIN),
        n_eval_episodes=1,
        deterministic=True,
        verbose=0,
    )

    agent = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        policy_kwargs={"net_arch": PPO_NET_ARCH},
        device=PPO_DEVICE,
        verbose=1,
    )

    print(f"\n  Training for {PPO_TOTAL_TIMESTEPS:,} timesteps …")
    agent.learn(total_timesteps=PPO_TOTAL_TIMESTEPS, callback=eval_callback)

    # EvalCallback saves best_model.zip — rename to our canonical path
    best_path = MODEL_DIR / "best_model.zip"
    if best_path.exists():
        best_path.rename(RL_MODEL_PATH)
    else:
        agent.save(str(RL_MODEL_PATH))

    print(f"\n  Model saved → {RL_MODEL_PATH.name}")

    # ── Evaluate on validation set ─────────────────────────────────────────────
    trained = PPO.load(str(RL_MODEL_PATH))
    metrics = evaluate_agent(trained, val_features, val_df["close"].values)

    print("\n  ── Validation metrics ──────────────────────────────")
    for k, v in metrics.items():
        print(f"    {k:25s} {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
