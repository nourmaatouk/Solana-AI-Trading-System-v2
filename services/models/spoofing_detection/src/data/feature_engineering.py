"""
feature_engineering.py — LOB Feature Extraction for Spoofing Detection
======================================================================

Computes a 49-dimensional feature vector per LOB event:

    Features  0–9  : Top 10 bid price levels
    Features 10–19 : Top 10 ask price levels
    Features 20–29 : Top 10 bid volume levels
    Features 30–39 : Top 10 ask volume levels
    Features 40    : Order flow price (incoming order)
    Features 41    : Order flow volume
    Features 42    : Order flow side (+1 buy / -1 sell)
    Features 43    : Δt — time since last event (milliseconds)
    Features 44    : Order imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    Features 45    : Spread = best_ask - best_bid
    Features 46    : Weighted mid-price
    Feature  47    : 1-minute rolling volatility (standardized)
    Feature  48    : 1-minute volume Z-score

→ Final output shape: (N_events, 49)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
N_LEVELS = 10
N_CORE_FEATURES = 49  # updated for macro-regime features


class LOBFeatureEngineer:
    """Extract, derive, and normalize LOB features for spoofing detection.

    Usage
    -----
    >>> eng = LOBFeatureEngineer(n_levels=10)
    >>> X_train = eng.fit_transform(train_df)   # fits scaler on train
    >>> X_val   = eng.transform(val_df)         # reuses train scaler
    """

    def __init__(self, n_levels: int = N_LEVELS, rolling_window: int = 20):
        self.n_levels = n_levels
        self.rolling_window = rolling_window
        self.scaler: Optional[StandardScaler] = None
        self._feature_names: list[str] = []

    # ── Column name helpers ──────────────────────────────────────────────────
    @property
    def bid_price_cols(self) -> list[str]:
        return [f"bid_price_{i}" for i in range(1, self.n_levels + 1)]

    @property
    def ask_price_cols(self) -> list[str]:
        return [f"ask_price_{i}" for i in range(1, self.n_levels + 1)]

    @property
    def bid_vol_cols(self) -> list[str]:
        return [f"bid_vol_{i}" for i in range(1, self.n_levels + 1)]

    @property
    def ask_vol_cols(self) -> list[str]:
        return [f"ask_vol_{i}" for i in range(1, self.n_levels + 1)]

    @property
    def feature_names(self) -> list[str]:
        """Return the ordered list of feature names after extraction."""
        return list(self._feature_names)

    # ── Core feature extraction ──────────────────────────────────────────────
    def _extract_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        for i, col in enumerate(self.bid_price_cols):
            features[f"f{i:02d}_bid_price_{i+1}"] = df[col].values
        for i, col in enumerate(self.ask_price_cols):
            features[f"f{i+10:02d}_ask_price_{i+1}"] = df[col].values
        for i, col in enumerate(self.bid_vol_cols):
            features[f"f{i+20:02d}_bid_vol_{i+1}"] = df[col].values
        for i, col in enumerate(self.ask_vol_cols):
            features[f"f{i+30:02d}_ask_vol_{i+1}"] = df[col].values

        features["f40_order_price"] = df["price"].values
        features["f41_order_volume"] = df["size"].astype(np.float64).values
        features["f42_order_side"] = df["direction"].astype(np.float64).values

        ts_ns = df["timestamp_ns"].values.astype(np.float64)
        delta_t_ns = np.diff(ts_ns, prepend=ts_ns[0])
        features["f43_delta_t_ms"] = delta_t_ns / 1e6

        total_bid_vol = df[self.bid_vol_cols].sum(axis=1).values
        total_ask_vol = df[self.ask_vol_cols].sum(axis=1).values
        safe_denom = np.where((total_bid_vol + total_ask_vol) == 0, 1.0, total_bid_vol + total_ask_vol)
        features["f44_order_imbalance"] = (total_bid_vol - total_ask_vol) / safe_denom

        best_ask, best_bid = df["ask_price_1"].values, df["bid_price_1"].values
        features["f45_spread"] = best_ask - best_bid

        bid_v1, ask_v1 = df["bid_vol_1"].values, df["ask_vol_1"].values
        safe_v = np.where((bid_v1 + ask_v1) == 0, 1.0, bid_v1 + ask_v1)
        features["f46_weighted_mid_price"] = (best_ask * bid_v1 + best_bid * ask_v1) / safe_v

        self._feature_names = list(features.columns)
        return features

    # ── Macro-regime features ────────────────────────────────────────────────
    def _add_macro_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Append 1-minute rolling volatility and volume Z-scores."""
        # Ensure we have a datetime index for time-based rolling
        if not isinstance(features.index, pd.DatetimeIndex):
            features.index = pd.to_datetime(df["timestamp_ns"], unit="ns")
        
        # 1. Feature 47: 1-minute rolling volatility of mid-price
        mid = features["f46_weighted_mid_price"]
        mid_returns = mid.diff().fillna(0)
        vol_1min = mid_returns.rolling("60s").std().fillna(0.0)
        
        # Normalize: Z-score using 1000-event rolling window
        vol_mean = vol_1min.rolling(window=1000, min_periods=1).mean()
        vol_std = vol_1min.rolling(window=1000, min_periods=1).std().replace(0, 1e-6)
        features["f47_vol_1min_zscore"] = (vol_1min - vol_mean) / vol_std
        
        # 2. Feature 48: 1-minute volume Z-score
        volume = features["f41_order_volume"]
        vol_roll_mean = volume.rolling("60s").mean()
        vol_roll_std = volume.rolling("60s").std().replace(0, 1e-6)
        features["f48_volume_zscore"] = (volume - vol_roll_mean) / vol_roll_std
        
        self._feature_names = list(features.columns)
        return features

    # ── Public API ───────────────────────────────────────────────────────────
    def fit_transform(
        self,
        df: pd.DataFrame,
        include_macro: bool = True,
        n_output_features: int = N_CORE_FEATURES,
    ) -> np.ndarray:
        """Extract features and fit the StandardScaler on training data."""
        features = self._extract_raw_features(df)
        if include_macro:
            # We need timestamp_ns for rolling windows
            features = self._add_macro_features(df, features)

        cols = features.columns[:n_output_features]
        raw = features[cols].values.astype(np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        self.scaler = StandardScaler()
        normalized = self.scaler.fit_transform(raw)

        logger.info("fit_transform: %d events × %d features", normalized.shape[0], normalized.shape[1])
        return normalized

    def transform(
        self,
        df: pd.DataFrame,
        include_macro: bool = True,
        n_output_features: int = N_CORE_FEATURES,
    ) -> np.ndarray:
        """Extract features and normalize using a *previously fitted* scaler."""
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted.")

        features = self._extract_raw_features(df)
        if include_macro:
            features = self._add_macro_features(df, features)

        cols = features.columns[:n_output_features]
        raw = features[cols].values.astype(np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        normalized = self.scaler.transform(raw)
        logger.info("transform: %d events × %d features", normalized.shape[0], normalized.shape[1])
        return normalized

    def get_mid_prices(self, df: pd.DataFrame) -> np.ndarray:
        return ((df["bid_price_1"].values + df["ask_price_1"].values) / 2.0)

    def get_timestamps_ms(self, df: pd.DataFrame) -> np.ndarray:
        return df["timestamp_ns"].values.astype(np.float64) / 1e6
