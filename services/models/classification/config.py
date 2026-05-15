"""
config.py — Single source of truth for all XGBoost pipeline constants.

Every hard-coded value that was previously scattered across scripts now lives
here.  Scripts import what they need; nothing is duplicated.
"""

from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
CLASSIFICATION_DIR = Path(__file__).resolve().parent   # …/classification/
PROJECT_ROOT       = CLASSIFICATION_DIR.parent         # …/services/models/

DATA_DIR       = CLASSIFICATION_DIR / "data"
PROCESSED_DIR  = DATA_DIR / "processed"
REALTIME_DIR   = DATA_DIR / "realtime"
MODEL_DIR      = PROJECT_ROOT / "models"               # …/services/models/models/
PLOTS_DIR      = CLASSIFICATION_DIR / "plots"
REPORTS_DIR    = CLASSIFICATION_DIR / "reports"

# ── Data collection ───────────────────────────────────────────────────────────
EXCHANGE            = "binance"
SYMBOL              = "SOL/USDT"
BTC_SYMBOL          = "BTC/USDT"
TIMEFRAME           = "15m"
CANDLE_LIMIT_HISTORY  = 500   # max bars per Binance API call (historical download)
CANDLE_LIMIT_REALTIME = 200   # bars fetched in realtime (enough for all indicators)

# ── Raw data filenames (inside data/raw/) ────────────────────────────────────
BTC_RAW_CSV = "btc_usdt_15m_raw.csv"

# ── Intermediate CSV filenames (inside PROCESSED_DIR) ────────────────────────
TECH_FEATURES_CSV  = "sol_usdt_15m_features_h4.csv"
SENTIMENT_CSV      = "sol_usdt_15m_features_h4_with_sentiment.csv"
ORDERBOOK_CSV      = "sol_usdt_15m_features_h4_with_sentiment_orderbook.csv"

# ── Triple-barrier labeling ───────────────────────────────────────────────────
TAKE_PROFIT_PCT = 0.020   # +2.0 % take-profit barrier
STOP_LOSS_PCT   = 0.015   # −1.5 % stop-loss barrier

# ── Chronological train / val / test split ────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios must sum to 1."

# ── Prediction horizons (number of 15-min candles ahead) ─────────────────────
HORIZONS       = [4, 8, 16]   # H4 = 1 h, H8 = 2 h, H16 = 4 h
ACTIVE_HORIZON = 4            # horizon used in realtime inference

# ── Non-stationary columns — MUST be excluded from XGBoost features ──────────
# These are absolute price levels; they contain regime information and will
# hurt generalisation across market cycles.
NON_STATIONARY_COLUMNS: set[str] = {
    # Raw OHLCV — absolute price/volume levels
    "open", "high", "low", "close", "volume",
    # Absolute moving averages
    "sma_10", "sma_20", "sma_50",
    "ema_10", "ema_20", "ema_50",
    # Bollinger absolute levels (stationary versions: bb_width, close_vs_bb_mid)
    "bb_mid", "bb_upper", "bb_lower",
    # Absolute momentum — dollar diffs (stationary versions: return_3/6/12)
    "momentum_3", "momentum_6", "momentum_12",
    # Absolute candle geometry (stationary versions: body_to_range, *_pct)
    "candle_body", "upper_shadow", "lower_shadow",
    # Absolute ATR (stationary version: atr_pct_14)
    "atr_14",
    # Absolute volume average (stationary version: volume_vs_sma20)
    "volume_sma_20",
    # Absolute MACD (stationary versions: macd_*_pct)
    "macd_line", "macd_signal", "macd_hist",
}

# Substrings that mark future-information (leakage) columns
LEAKAGE_SUBSTRINGS: list[str] = [
    "future_return", "target", "lead", "future", "t+1",
]

# ── Feature category keywords (used in 07 for logging) ───────────────────────
SENTIMENT_KEYWORDS: list[str] = [
    "sentiment", "news_volume", "positive_ratio", "negative_ratio",
    "neutral_ratio", "strong_positive", "strong_negative", "max_confidence",
    "event_count", "negative_event", "positive_event", "high_impact",
    "relevance_weighted", "event_signal", "gdelt", "rss",
    "sentiment_momentum", "news_volume_change", "sentiment_spike",
]
ORDERBOOK_KEYWORDS: list[str] = [
    "imbalance", "spread_mean", "mid_price", "bid_volume",
    "ask_volume", "pressure_indicator",
]

# ── Orderbook feature names (used in realtime predictor gate logic) ───────────
ORDERBOOK_FEATURES: list[str] = [
    "imbalance_mean",
    "imbalance_std",
    "spread_mean",
    "mid_price",
    "bid_volume_mean",
    "ask_volume_mean",
    "pressure_indicator",
]

# ── Hawkes intensity feature names (added by 03_hawkes_intensity.py) ─────────
HAWKES_KEYWORDS: list[str] = ["hawkes"]
HAWKES_FEATURES: list[str] = [
    "hawkes_mu",
    "hawkes_branching_ratio",
]

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
XGB_N_ESTIMATORS         = 200
XGB_MAX_DEPTH            = 5
XGB_LEARNING_RATE        = 0.05
XGB_SUBSAMPLE            = 0.8
XGB_COLSAMPLE_BYTREE     = 0.8
XGB_EARLY_STOPPING_ROUNDS = 20
XGB_SEED                 = 42

# ── Inference gate ────────────────────────────────────────────────────────────
# P(up) >= CONFIDENCE_THRESHOLD          → UP   signal
# P(up) <= 1 − CONFIDENCE_THRESHOLD      → DOWN signal
# otherwise                              → NEUTRAL (no trade)
CONFIDENCE_THRESHOLD = 0.69

# ── Paper-trader risk management ──────────────────────────────────────────────
# Position size scales linearly from MIN_POSITION_SIZE (at the gate boundary)
# to 1.0 (at maximum confidence).  This mirrors fractional Kelly sizing without
# requiring a full Kelly calculation.
MIN_POSITION_SIZE = 0.25   # minimum fraction of equity deployed per trade
