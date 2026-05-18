"""
config.py — Centralized configuration for the TEN-GNN Spoofing Detection system
================================================================================

Single source of truth for feature dimensions, model hyperparameters, and
checkpoint paths.  Import from here rather than hardcoding constants.

Feature schema
--------------
Two data paths exist.  They produce different feature counts because the Kaggle
LOB data has fewer available columns than a full LOBSTER feed:

    Kaggle path (kaggle_loader.py → extract_lob_features):
        2  base features  (midpoint, spread)
        30 depth features (bid_notional + ask_notional × 15 levels, interleaved)
        1  imbalance      (bid_vol - ask_vol) / total_vol
        1  cancel_ratio   cancel_vol / (limit_vol + cancel_vol)
        ─────────────────
        34 total  →  N_FEATURES_KAGGLE

    Index map:  0=midpoint, 1=spread, 2=bid_1, 3=ask_1, ..., 30=bid_15, 31=ask_15,
                32=imbalance, 33=cancel_ratio

    LOBSTER path (feature_engineering.py → LOBFeatureEngineer):
        40 price/vol features (10 bid price, 10 ask price, 10 bid vol, 10 ask vol)
        3  order features     (price, vol, side)
        1  delta_t_ms
        1  order imbalance
        1  spread
        1  weighted mid
        1  rolling vol z-score
        1  volume z-score
        ─────────────────
        49 total  →  N_FEATURES_LOBSTER

Any model trained on Kaggle data uses d_input=N_FEATURES_KAGGLE=33.
Any model trained on LOBSTER data uses d_input=N_FEATURES_LOBSTER=49.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Feature dimensions ────────────────────────────────────────────────────────
N_FEATURES_KAGGLE: int = 34   # midpoint + spread + (bid+ask)×15 levels + imbalance + cancel_ratio
N_FEATURES_LOBSTER: int = 49  # full LOBFeatureEngineer output

# ── Windowing ─────────────────────────────────────────────────────────────────
DEFAULT_WINDOW_LENGTH: int = 100   # events per sequence
DEFAULT_WINDOW_STRIDE: int = 50    # 50% overlap

# ── LOB depth ─────────────────────────────────────────────────────────────────
DEFAULT_N_LEVELS: int = 15  # depth levels in Kaggle data; LOBSTER uses 10

# ── Assets ────────────────────────────────────────────────────────────────────
DEFAULT_ASSETS: list[str] = os.environ.get("DEFAULT_ASSETS", "BTC,ETH,SOL").split(",")
DEFAULT_FREQUENCY: str = "1min"

# ── Model hyperparameters (paper-aligned) ─────────────────────────────────────
MODEL_D_MODEL: int = 256
MODEL_N_HEADS: int = 8
MODEL_N_LAYERS: int = 6
MODEL_D_FF: int = 1024
MODEL_DROPOUT: float = 0.1

# ── Training defaults ─────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE: int = 32
TRAIN_MAX_EPOCHS: int = 50
TRAIN_LR: float = 5e-4
TRAIN_WEIGHT_DECAY: float = 1e-4
TRAIN_GRAD_CLIP: float = 1.0
TRAIN_PATIENCE: int = 8

# ── Checkpoint paths ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent  # spoofing_detection/

CHECKPOINT_BEST: Path = _ROOT / "checkpoints" / "tengnn_btc_eth_1min" / "best_model.pt"
CHECKPOINT_INT8: Path = _ROOT / "checkpoints" / "ten_gnn_int8.pt"
CHECKPOINT_SCRIPTED: Path = _ROOT / "checkpoints" / "ten_gnn_scripted.pt"

# ── Hawkes process ────────────────────────────────────────────────────────────
HAWKES_MAX_EVENTS: int = 2_000   # cap for O(n²) MLE
HAWKES_REFRESH_MS: float = 500.0  # live updater refresh interval

# ── Data directories ──────────────────────────────────────────────────────────
DATA_RAW_KAGGLE: Path = _ROOT / "data" / "raw" / "kaggle_crypto"
DATA_RAW_LOBSTER: Path = _ROOT / "data" / "raw"

# ── Chronological split ratios ────────────────────────────────────────────────
SPLIT_TRAIN: float = 0.65
SPLIT_VAL: float = 0.21
SPLIT_TEST: float = 0.14
