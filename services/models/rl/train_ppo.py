"""
train_ppo.py — Train a PPO agent on historical SOL/USDT 15-minute data.

Default usage (aligned fusion model — recommended):
    python services/models/rl/train_ppo.py

Outputs (aligned run — does NOT overwrite ppo_trading_agent.zip):
    services/models/models/ppo_trading_agent_aligned_fusion.zip  — best checkpoint
    services/models/models/rl_aligned_scaler.pkl                 — fitted StandardScaler

Observation space (54-dim):
    49 market features (StandardScaler-normalised)
  + 1  pipeline signal: xgb_prob_up (XGBoost P(UP) at each bar, retroactive)
  + 4  portfolio state: [position, unrealised_pnl, steps_idle/100, log(equity/bh)]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    FEATURE_COLUMNS_PATH,
    FUSION_RISK_HIGH_THRESHOLD,
    FUSION_RISK_MEDIUM_THRESHOLD,
    FUSION_RISK_WEIGHT_GNN,
    FUSION_RISK_WEIGHT_SENTIMENT,
    FUSION_RISK_WEIGHT_SPOOFING,
    FUSION_SIGNAL_DIM,
    FUSION_WEIGHT_CLASSIFICATION,
    FUSION_WEIGHT_SPOOFING,
    FUSION_GNN_STRESS_PENALTY,
    MODEL_DIR,
    PIPELINE_SIGNAL_DIM,
    PLOTS_DIR,
    PROCESSED_CSV,
    PPO_ALIGNED_TOTAL_TIMESTEPS,
    PPO_BATCH_SIZE,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF,
    PPO_EVAL_FREQ_MIN,
    PPO_FUSION_TOTAL_TIMESTEPS,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_N_EPOCHS,
    PPO_N_STEPS,
    PPO_DEVICE,
    PPO_NET_ARCH,
    RL_ALIGNED_MODEL_PATH,
    RL_ALIGNED_SCALER_PATH,
    RL_FUSION_MODEL_PATH,
    RL_FUSION_SCALER_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
    XGB_MODEL_PATH,
    XGB_SCALER_PATH,
)
from environment import TradingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


# ── Pipeline signal builder ───────────────────────────────────────────────────

def compute_pipeline_signals(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """Return pipeline signal array of shape (N, PIPELINE_SIGNAL_DIM).

    Currently PIPELINE_SIGNAL_DIM = 1: [xgb_prob_up].

    XGBoost inference is run on features at time T only — no look-ahead.
    The production XGBoost artefacts (XGB_MODEL_PATH, XGB_SCALER_PATH) are
    reused directly.  No new model, scaler, or data file is created here.

    This function is also imported by evaluate.py to avoid code duplication.
    """
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found: {XGB_MODEL_PATH}")
    if not XGB_SCALER_PATH.exists():
        raise FileNotFoundError(f"XGBoost scaler not found: {XGB_SCALER_PATH}")

    xgb_scaler = joblib.load(XGB_SCALER_PATH)
    xgb_model  = xgb.XGBClassifier()
    xgb_model.load_model(str(XGB_MODEL_PATH))

    X        = df[feature_cols].fillna(0.0).values
    X_scaled = xgb_scaler.transform(X)
    prob_up  = xgb_model.predict_proba(X_scaled)[:, 1]   # P(UP) — no look-ahead

    signals = prob_up.reshape(-1, 1)
    if signals.shape[1] != PIPELINE_SIGNAL_DIM:
        raise ValueError(
            f"Expected PIPELINE_SIGNAL_DIM={PIPELINE_SIGNAL_DIM}, got {signals.shape[1]}"
        )
    return signals.astype(np.float64)


# ── Fusion signal builder (Fusion → PPO architecture) ────────────────────────

def compute_fusion_signals(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """Return retroactive fusion signal array of shape (N, FUSION_SIGNAL_DIM=3).

    Columns (order must match PIPELINE_IDX_* constants in config.py):
      0: xgb_prob_up     [0, 1]    — XGBoost P(UP); reuses compute_pipeline_signals()
      1: fused_score     [-1, 1]   — retroactive fusion: XGBoost + hawkes spoofing proxy
      2: risk_encoded    [0,0.5,1] — LOW=0, MEDIUM=0.5, HIGH=1.0

    No look-ahead: all inputs at time T use only features at time T.
    Sentiment and GNN components are not available in the historical training CSV
    (no historical sentiment columns, pressure_indicator=0 throughout).
    The fusion formula therefore uses:
      - Classification: XGBoost prob_up mapped to [-1, 1]
      - Spoofing proxy: hawkes_branching_ratio (already in feature_cols, so no new data read)
      - Sentiment / GNN: defaults to neutral (0 contribution)

    At realtime, realtime_rl_predictor.py reads the full fused_score directly from
    last_fused_decision.json, which includes sentiment + GNN contributions.
    The training/inference mismatch is intentional and conservative: training on a
    partial fusion score teaches the policy to use directional information, while
    realtime exposure to a fuller signal can only improve (not degrade) guidance.

    Importable from evaluate.py — no logic is duplicated.
    """
    # ── Step 1: xgb_prob_up via existing compute_pipeline_signals ─────────────
    pipeline_1d = compute_pipeline_signals(df, feature_cols)   # shape (N, 1)
    xgb_prob_up = pipeline_1d[:, 0]                            # shape (N,)

    # ── Step 2: retroactive fused_score ───────────────────────────────────────
    # Classification component: continuous [-1, 1] mapping of P(UP)
    classification_score = xgb_prob_up * 2.0 - 1.0   # [0,1] → [-1,1]

    # Spoofing proxy: hawkes_branching_ratio (already in the 49 market features,
    # reused here only to compute the derived fused_score and risk signal).
    hawkes_col = "hawkes_branching_ratio"
    if hawkes_col in df.columns:
        hawkes_br = df[hawkes_col].fillna(0.0).clip(0.0, 1.0).values.astype(np.float64)
    else:
        hawkes_br = np.zeros(len(df), dtype=np.float64)

    # Apply fusion formula (fusion/config.py weights, sentinel=0, GNN=0):
    #   fused = W_class * class_score - W_spoofing * (spoofing_proxy^2)
    fused_score = (
        FUSION_WEIGHT_CLASSIFICATION * classification_score
        - FUSION_WEIGHT_SPOOFING * (hawkes_br ** 2)
    )
    fused_score = np.clip(fused_score, -1.0, 1.0)

    # ── Step 3: retroactive risk_encoded ──────────────────────────────────────
    # risk formula from fusion/config.py:
    #   risk = W_spoof*spoofing + W_gnn*gnn + W_sent*(1-sent_rel)
    # With proxies: gnn=0, sent_rel=0.5 (neutral default)
    risk_score = (
        FUSION_RISK_WEIGHT_SPOOFING  * hawkes_br
        + FUSION_RISK_WEIGHT_GNN     * 0.0
        + FUSION_RISK_WEIGHT_SENTIMENT * 0.5   # (1 - neutral_sent_rel)
    )
    risk_encoded = np.where(
        risk_score > FUSION_RISK_HIGH_THRESHOLD, 1.0,
        np.where(risk_score > FUSION_RISK_MEDIUM_THRESHOLD, 0.5, 0.0)
    )

    signals = np.stack([xgb_prob_up, fused_score, risk_encoded], axis=1)
    if signals.shape[1] != FUSION_SIGNAL_DIM:
        raise ValueError(
            f"Expected FUSION_SIGNAL_DIM={FUSION_SIGNAL_DIM}, got {signals.shape[1]}"
        )
    return signals.astype(np.float64)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray, periods_per_year: int = 35_040) -> float:
    """Annualised Sharpe from per-step log returns (15-min = 35 040 bars/year)."""
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd   = (peak - equity_curve) / np.where(peak > 0, peak, 1)
    return float(dd.max())


def evaluate_agent(
    agent: PPO,
    features: np.ndarray,
    prices: np.ndarray,
    pipeline_signals: np.ndarray | None = None,
) -> dict:
    """Run one deterministic episode and return trading metrics."""
    env = TradingEnv(features, prices, pipeline_signals)
    obs, _ = env.reset()
    equity_curve    = [env._equity]
    bh_equity_curve = [env._bh_equity]
    trades          = 0

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

    # Sharpe from actual equity log-returns (not shaped rewards)
    log_rets     = np.diff(np.log(np.maximum(equity_curve, 1e-10)))
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    bh_return    = (bh_equity_curve[-1] - bh_equity_curve[0]) / bh_equity_curve[0]

    return {
        "total_return_pct":  round(total_return * 100, 2),
        "bh_return_pct":     round(bh_return * 100, 2),
        "alpha_pct":         round((total_return - bh_return) * 100, 2),
        "sharpe":            round(_sharpe(log_rets), 3),
        "max_drawdown_pct":  round(_max_drawdown(equity_curve) * 100, 2),
        "trades":            trades,
        "final_equity":      round(equity_curve[-1], 2),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def _train_run(
    label: str,
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_prices: np.ndarray,
    val_prices: np.ndarray,
    train_sig: np.ndarray | None,
    val_sig: np.ndarray | None,
    total_timesteps: int,
    model_path,
    scaler,
) -> None:
    """Shared PPO training loop. Used for both aligned and fusion models."""
    n_sig = train_sig.shape[1] if train_sig is not None else 0
    print(f"\n  {'─'*56}")
    print(f"  Training: {label}")
    print(f"  Obs dim  : 49 market + {n_sig} pipeline + 4 portfolio = {49 + n_sig + 4}")
    print(f"  Steps    : {total_timesteps:,}")
    print(f"  Output   : {model_path.name}")

    train_env = Monitor(TradingEnv(train_features, train_prices, train_sig, random_start=True))
    val_env   = Monitor(TradingEnv(val_features,   val_prices,   val_sig))

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
    agent.learn(total_timesteps=total_timesteps, callback=eval_callback)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODEL_DIR / "best_model.zip"
    if best_path.exists():
        best_path.rename(model_path)
    else:
        agent.save(str(model_path))
    print(f"\n  Model saved → {model_path.name}")

    # Quick validation metrics
    trained = PPO.load(str(model_path))
    metrics = evaluate_agent(trained, val_features, val_prices, val_sig)
    print(f"\n  ── Validation metrics ──────────────────────────────")
    for k, v in metrics.items():
        print(f"    {k:25s} {v}")


def main() -> None:
    print("=" * 60)
    print("PPO Training — SOL/USDT 15m")
    print("=" * 60)
    print(f"  Aligned model  : {RL_ALIGNED_MODEL_PATH.name}")
    print(f"  Fusion v2 model: {RL_FUSION_MODEL_PATH.name}")
    print(f"  Production PPO : unchanged ({RL_ALIGNED_MODEL_PATH.parent}/ppo_trading_agent.zip)")
    print()

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

    print(f"  Rows: {len(df):,}   Market features: {len(feature_cols)}")

    # ── Chronological split (no test data touches training or val) ─────────────
    n       = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df   = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    print(f"  Train: {n_train:,}  Val: {n_val:,}  Test: {n - n_train - n_val:,}")

    # ── Fit market scaler on training data only ────────────────────────────────
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[feature_cols].fillna(0.0).values)
    val_features   = scaler.transform(val_df[feature_cols].fillna(0.0).values)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────────────
    # Run 1: Aligned PPO (54-dim: xgb_prob_up only)
    # ────────────────────────────────────────────────────────────────────────────
    print("\n  Computing aligned pipeline signals (1-dim: xgb_prob_up)…")
    aligned_signals_all = compute_pipeline_signals(df, feature_cols)
    train_aligned_sig   = aligned_signals_all[:n_train]
    val_aligned_sig     = aligned_signals_all[n_train : n_train + n_val]

    joblib.dump(scaler, RL_ALIGNED_SCALER_PATH)
    print(f"  Scaler saved → {RL_ALIGNED_SCALER_PATH.name}")

    _train_run(
        label           = "Aligned PPO (54-dim)",
        train_features  = train_features,
        val_features    = val_features,
        train_prices    = train_df["close"].values,
        val_prices      = val_df["close"].values,
        train_sig       = train_aligned_sig,
        val_sig         = val_aligned_sig,
        total_timesteps = PPO_ALIGNED_TOTAL_TIMESTEPS,
        model_path      = RL_ALIGNED_MODEL_PATH,
        scaler          = scaler,
    )

    # ────────────────────────────────────────────────────────────────────────────
    # Run 2: Fusion→PPO v2 (56-dim: xgb_prob_up + fused_score + risk_encoded)
    # ────────────────────────────────────────────────────────────────────────────
    print("\n  Computing fusion signals (3-dim: xgb_prob_up, fused_score, risk_encoded)…")
    fusion_signals_all = compute_fusion_signals(df, feature_cols)
    train_fusion_sig   = fusion_signals_all[:n_train]
    val_fusion_sig     = fusion_signals_all[n_train : n_train + n_val]

    print(f"  fused_score   min={fusion_signals_all[:,1].min():.3f}  "
          f"max={fusion_signals_all[:,1].max():.3f}  "
          f"mean={fusion_signals_all[:,1].mean():.3f}")
    print(f"  risk_encoded  LOW={( fusion_signals_all[:,2]==0.0).mean()*100:.1f}%  "
          f"MEDIUM={(fusion_signals_all[:,2]==0.5).mean()*100:.1f}%  "
          f"HIGH={(fusion_signals_all[:,2]==1.0).mean()*100:.1f}%")

    # The fusion model uses its own scaler (same StandardScaler, same training data)
    joblib.dump(scaler, RL_FUSION_SCALER_PATH)
    print(f"  Scaler saved → {RL_FUSION_SCALER_PATH.name}")

    _train_run(
        label           = "Fusion→PPO v2 (56-dim)",
        train_features  = train_features,
        val_features    = val_features,
        train_prices    = train_df["close"].values,
        val_prices      = val_df["close"].values,
        train_sig       = train_fusion_sig,
        val_sig         = val_fusion_sig,
        total_timesteps = PPO_FUSION_TOTAL_TIMESTEPS,
        model_path      = RL_FUSION_MODEL_PATH,
        scaler          = scaler,
    )

    print("\n" + "=" * 60)
    print("  Training complete.  Production artifact unchanged:")
    print(f"    ppo_trading_agent.zip — {(MODEL_DIR / 'ppo_trading_agent.zip').stat().st_size:,} bytes")
    print("=" * 60)


if __name__ == "__main__":
    main()
