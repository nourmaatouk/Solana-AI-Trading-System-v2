"""
config.py — Single source of truth for all PPO RL pipeline constants.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# Load classification/config.py by absolute path — avoids generic-name collisions.
_CLF_CONFIG_PATH = Path(__file__).resolve().parents[1] / "classification" / "config.py"
_spec = importlib.util.spec_from_file_location("classification.config", _CLF_CONFIG_PATH)
_clf = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_clf)  # type: ignore[union-attr]

# Load fusion/config.py by absolute path — reuse thresholds, do not duplicate.
_FUSION_CONFIG_PATH = Path(__file__).resolve().parents[1] / "fusion" / "config.py"
_fspec = importlib.util.spec_from_file_location("fusion.config", _FUSION_CONFIG_PATH)
_fus = importlib.util.module_from_spec(_fspec)  # type: ignore[arg-type]
_fspec.loader.exec_module(_fus)  # type: ignore[union-attr]

# ── Constants shared with the classification pipeline ─────────────────────────
# Imported rather than copied so a change in one place propagates everywhere.
FEE_RT                   = _clf.FEE_RT
SLIPPAGE_PCT             = _clf.SLIPPAGE_PCT
TRAIN_RATIO              = _clf.TRAIN_RATIO
VAL_RATIO                = _clf.VAL_RATIO
SIGNAL_STALE_MINUTES     = _clf.SIGNAL_STALE_MINUTES
ACTIVE_HORIZON           = _clf.ACTIVE_HORIZON
XGB_CONFIDENCE_THRESHOLD = float(_clf.CONFIDENCE_THRESHOLD)

# ── Constants shared with the fusion layer ────────────────────────────────────
# Risk boundaries and weights imported from fusion/config.py — single source of truth.
FUSION_WEIGHT_CLASSIFICATION       = float(_fus.WEIGHT_CLASSIFICATION)
FUSION_WEIGHT_SPOOFING             = float(_fus.WEIGHT_SPOOFING)
FUSION_WEIGHT_GNN                  = float(_fus.WEIGHT_GNN)
FUSION_GNN_STRESS_PENALTY          = float(_fus.GNN_STRESS_PENALTY)
FUSION_RISK_HIGH_THRESHOLD         = float(_fus.RISK_HIGH_THRESHOLD)
FUSION_RISK_MEDIUM_THRESHOLD       = float(_fus.RISK_MEDIUM_THRESHOLD)
FUSION_RISK_WEIGHT_SPOOFING        = float(_fus.RISK_WEIGHT_SPOOFING)
FUSION_RISK_WEIGHT_GNN             = float(_fus.RISK_WEIGHT_GNN)
FUSION_RISK_WEIGHT_SENTIMENT       = float(_fus.RISK_WEIGHT_SENTIMENT_PENALTY)

# ── Directory layout ──────────────────────────────────────────────────────────
RL_DIR        = Path(__file__).resolve().parent
DATA_DIR      = RL_DIR.parent / "classification" / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REALTIME_DIR  = DATA_DIR / "realtime"
MODEL_DIR     = RL_DIR.parent / "models"
PLOTS_DIR     = RL_DIR.parent / "classification" / "plots"

# ── Artifact paths ────────────────────────────────────────────────────────────
RL_MODEL_PATH        = MODEL_DIR / "ppo_trading_agent.zip"
RL_SCALER_PATH       = MODEL_DIR / "rl_scaler.pkl"
FEATURE_COLUMNS_PATH = PROCESSED_DIR / "feature_columns.json"
PROCESSED_CSV        = PROCESSED_DIR / "sol_usdt_15m_features_h4_with_sentiment_orderbook.csv"
RL_SIGNAL_PATH        = REALTIME_DIR / "last_rl_signal.json"
RL_STATE_PATH         = REALTIME_DIR / "rl_position_state.json"
REALTIME_FEATURES_CSV = REALTIME_DIR / "latest_features_with_sentiment.csv"
# Path to the upstream XGBoost classification signal (contains probability_up)
XGB_SIGNAL_PATH       = REALTIME_DIR / "last_signal.json"

# ── Aligned fusion model artifacts (separate from production ppo_trading_agent.zip) ──
# These are produced by re-training with the expanded pipeline-signal observation.
XGB_MODEL_PATH           = MODEL_DIR / f"xgboost_h{ACTIVE_HORIZON}.json"
XGB_SCALER_PATH          = PROCESSED_DIR / f"scaler_h{ACTIVE_HORIZON}.pkl"
RL_ALIGNED_MODEL_PATH    = MODEL_DIR / "ppo_trading_agent_aligned_fusion.zip"
RL_ALIGNED_SCALER_PATH   = MODEL_DIR / "rl_aligned_scaler.pkl"

# ── Fusion→PPO v2 model artifacts ─────────────────────────────────────────────
# Full Fusion→PPO: obs = 49 market + 3 fusion signals + 4 portfolio = 56 dims
# Fusion signals: [xgb_prob_up, retroactive_fused_score, risk_encoded]
# Does NOT overwrite production ppo_trading_agent.zip or aligned model.
RL_FUSION_MODEL_PATH   = MODEL_DIR / "ppo_trading_agent_fusion_v2.zip"
RL_FUSION_SCALER_PATH  = MODEL_DIR / "rl_fusion_scaler.pkl"
XGB_FUSED_SIGNAL_PATH  = REALTIME_DIR / "last_fused_decision.json"

# ── Observation space ─────────────────────────────────────────────────────────
# Portfolio state features appended at the end of the observation vector.
PORTFOLIO_FEATURE_DIM = 4   # [position, unrealized_pnl_pct, steps_since_trade/100, rel_to_bh]

# Pipeline signal features inserted between market features and portfolio state.
# Order must be identical in train_ppo.py and realtime_rl_predictor.py.
#   Index 0: xgb_prob_up — XGBoost P(price UP) at current bar.
#     Available historically via retroactive XGBoost inference on the processed CSV.
#     Available at realtime from last_signal.json → probability_up.
PIPELINE_SIGNAL_DIM   = 1   # legacy aligned model (54-dim obs)

# Fusion→PPO v2 pipeline signal layout (56-dim obs total):
#   Index 0: xgb_prob_up     [0, 1]   — XGBoost P(UP)
#   Index 1: fused_score     [-1, 1]  — retroactive fusion (XGBoost + hawkes proxy)
#   Index 2: risk_encoded    [0,0.5,1]— LOW=0, MEDIUM=0.5, HIGH=1.0
FUSION_SIGNAL_DIM      = 3
PIPELINE_IDX_XGB       = 0
PIPELINE_IDX_FUSED     = 1
PIPELINE_IDX_RISK      = 2
INITIAL_EQUITY        = 10_000.0
MIN_EQUITY_FRACTION   = 0.50   # episode ends if equity < 50% of initial

# ── PPO hyperparameters ───────────────────────────────────────────────────────
PPO_LEARNING_RATE   = 3e-4
PPO_N_STEPS         = 2048
PPO_BATCH_SIZE      = 64
PPO_N_EPOCHS        = 10
PPO_GAMMA           = 0.99
PPO_GAE_LAMBDA      = 0.95
PPO_CLIP_RANGE      = 0.2
PPO_ENT_COEF        = 0.003  # low entropy — let policy converge to near-deterministic
PPO_TOTAL_TIMESTEPS = 2_000_000
PPO_NET_ARCH        = [256, 256]
PPO_DEVICE          = "cpu"  # MLP policy trains faster on CPU (SB3 recommendation)

# ── Inference gate ────────────────────────────────────────────────────────────
# Minimum action probability to emit BUY/SELL; otherwise NEUTRAL
RL_CONFIDENCE_THRESHOLD = float(os.environ.get("RL_CONFIDENCE_THRESHOLD", "0.60"))

# ── EvalCallback frequency floor ──────────────────────────────────────────────
PPO_EVAL_FREQ_MIN = 10_000
# Minimum valid round-trips a checkpoint must execute on the validation episode to
# be eligible as "best model". Stops TradingEvalCallback from selecting the
# degenerate do-nothing policy (which a plain reward-based EvalCallback picks,
# because under the absolute-return reward a flat episode scores 0 and beats every
# checkpoint that traded at a net loss).
MIN_EVAL_ROUND_TRIPS = 5

# ── Reward shaping ────────────────────────────────────────────────────────────
# Drawdown penalty applied only while in position.
DRAWDOWN_PENALTY_COEF  = 0.02   # small stop-loss incentive while holding
# Per-trade penalty applied to the REWARD only (never to equity — see TradingEnv).
# Anti-churn shaping under the absolute-return reward. Tuning history (200K-step
# smoke, val total-return / Sharpe / max-drawdown, after the equity-cost fix):
#   0.002 → churn, blew up to −50% / Sharpe −6.1 / 51% DD.
#   0.005 → stable +2.4% / Sharpe +0.21 / 9.8% DD, survived full val.  ← chosen
#   0.01+ → over-suppressed (agent stops participating).
TRADE_PENALTY          = 0.005
# Penalty for an invalid action: SELL when already flat, or BUY when already long.
# Raised from 0.002 to 0.01 — the aligned model had ~3987 invalid SELLs per episode.
INVALID_ACTION_PENALTY = 0.01
# Sharpe bonus on excess returns. Disabled until the basic signal produces a
# working policy — adding it complicates the learning signal.
SHARPE_WINDOW          = 20
SHARPE_REWARD_COEF     = 0.0

# ── Fusion→PPO v2 additional reward shaping ───────────────────────────────────
# Penalty for closing a position within MIN_HOLD_STEPS bars of opening it.
# Prevents the policy from churning (enter → exit → enter → exit …). A 12-bar
# (3-hour) minimum hold paired with TRADE_PENALTY=0.005 gave the stable positive
# policy; shorter windows let the flip-flop attractor (→ fee-bleed blowup) return.
OVERTRADING_PENALTY    = 0.005
MIN_HOLD_STEPS         = 12     # minimum bars to hold before SELL without penalty (3 hours)
# Extra penalty for BUY action when risk_encoded == 1.0 (HIGH risk regime).
RISKY_BUY_PENALTY      = 0.005
# Small bonus when the portfolio return sign matches the fused_score sign,
# teaching the policy to align trades with the upstream fusion signal.
SIGNAL_ALIGN_COEF      = 0.05

# ── Aligned training timesteps ────────────────────────────────────────────────
# Separate from PPO_TOTAL_TIMESTEPS so the canonical production constant is unchanged.
# Override via env var: PPO_ALIGNED_TIMESTEPS=<n> python train_ppo.py
PPO_ALIGNED_TOTAL_TIMESTEPS = int(os.environ.get("PPO_ALIGNED_TIMESTEPS", "500000"))

# ── Fusion→PPO v2 training timesteps ─────────────────────────────────────────
# Override via env var: PPO_FUSION_TIMESTEPS=<n> python train_ppo.py
PPO_FUSION_TOTAL_TIMESTEPS  = int(os.environ.get("PPO_FUSION_TIMESTEPS", "1000000"))
