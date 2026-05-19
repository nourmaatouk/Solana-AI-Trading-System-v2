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

# ── Constants shared with the classification pipeline ─────────────────────────
# Imported rather than copied so a change in one place propagates everywhere.
FEE_RT               = _clf.FEE_RT
SLIPPAGE_PCT         = _clf.SLIPPAGE_PCT
TRAIN_RATIO          = _clf.TRAIN_RATIO
VAL_RATIO            = _clf.VAL_RATIO
SIGNAL_STALE_MINUTES = _clf.SIGNAL_STALE_MINUTES

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
RL_SIGNAL_PATH       = REALTIME_DIR / "last_rl_signal.json"
REALTIME_FEATURES_CSV = REALTIME_DIR / "latest_features_with_sentiment.csv"

# ── Environment ───────────────────────────────────────────────────────────────
# Portfolio state features appended to market observation
PORTFOLIO_FEATURE_DIM = 3   # [current_position, unrealized_pnl_pct, steps_since_trade/100]
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
PPO_ENT_COEF        = 0.05   # higher entropy → more exploration, slower collapse
PPO_TOTAL_TIMESTEPS = 1_000_000
PPO_NET_ARCH        = [256, 256]
PPO_DEVICE          = "cpu"  # MLP policy trains faster on CPU (SB3 recommendation)

# ── Inference gate ────────────────────────────────────────────────────────────
# Minimum action probability to emit BUY/SELL; otherwise NEUTRAL
RL_CONFIDENCE_THRESHOLD = float(os.environ.get("RL_CONFIDENCE_THRESHOLD", "0.60"))

# ── EvalCallback frequency floor ──────────────────────────────────────────────
PPO_EVAL_FREQ_MIN = 10_000

# ── Reward shaping ────────────────────────────────────────────────────────────
# Drawdown penalty applied only while in position.
DRAWDOWN_PENALTY_COEF = 0.10
# Per-trade cost on top of fee/slippage. Reduced from 0.005 so the agent is
# not scared out of trades — the Sharpe term now handles overtrading.
TRADE_PENALTY = 0.001
# Rolling-Sharpe bonus: rewards consistent risk-adjusted returns over a window.
SHARPE_WINDOW     = 20
SHARPE_REWARD_COEF = 0.05
# Tiny bonus for holding a profitable open position — prevents premature exits.
HOLD_PROFIT_COEF  = 0.001
