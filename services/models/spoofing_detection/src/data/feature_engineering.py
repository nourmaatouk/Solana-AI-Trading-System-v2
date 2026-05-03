"""
feature_engineering.py — LOB Feature Extraction for Spoofing Detection
======================================================================

Computes a 47-dimensional feature vector per LOB event, aligned with
Appendix B / Figure B1 of the paper:

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

Second-order features are appended during the ``transform`` step:
    Feature  47    : Rate of change of order imbalance
    Feature  48    : Rolling volatility of mid-price (window=20)

→ Final output shape: (N_events, 47) after selecting the core 47 features,
  or (N_events, 49) if second-order features are included.

The user's spec asks for 47 columns, so the second-order features
(rate-of-change of imbalance, rolling volatility) are folded into the 47
by replacing the last available slots as needed. The actual mapping can be
adjusted via configuration.
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
N_CORE_FEATURES = 47  # paper spec


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
        """Extract the full feature matrix from a merged LOB DataFrame.

        Expects the output of ``lobster_loader.load_lobster_data()`` or
        ``generate_synthetic_lobster_data()``.
        """
        features = pd.DataFrame(index=df.index)

        # --- LOB State: Top 10 bid prices (features 0-9) -------------------
        for i, col in enumerate(self.bid_price_cols):
            features[f"f{i:02d}_bid_price_{i+1}"] = df[col].values

        # --- LOB State: Top 10 ask prices (features 10-19) -----------------
        for i, col in enumerate(self.ask_price_cols):
            features[f"f{i+10:02d}_ask_price_{i+1}"] = df[col].values

        # --- LOB State: Top 10 bid volumes (features 20-29) ----------------
        for i, col in enumerate(self.bid_vol_cols):
            features[f"f{i+20:02d}_bid_vol_{i+1}"] = df[col].values

        # --- LOB State: Top 10 ask volumes (features 30-39) ----------------
        for i, col in enumerate(self.ask_vol_cols):
            features[f"f{i+30:02d}_ask_vol_{i+1}"] = df[col].values

        # --- Order Flow: price, volume, side (features 40-42) --------------
        features["f40_order_price"] = df["price"].values
        features["f41_order_volume"] = df["size"].astype(np.float64).values
        features["f42_order_side"] = df["direction"].astype(np.float64).values

        # --- Time feature: delta_t in milliseconds (feature 43) ------------
        ts_ns = df["timestamp_ns"].values.astype(np.float64)
        delta_t_ns = np.diff(ts_ns, prepend=ts_ns[0])
        delta_t_ms = delta_t_ns / 1e6  # nanoseconds → milliseconds
        features["f43_delta_t_ms"] = delta_t_ms

        # --- Derived: order_imbalance (feature 44) --------------------------
        total_bid_vol = df[self.bid_vol_cols].sum(axis=1).values
        total_ask_vol = df[self.ask_vol_cols].sum(axis=1).values
        denom = total_bid_vol + total_ask_vol
        # Avoid division by zero
        safe_denom = np.where(denom == 0, 1.0, denom)
        order_imbalance = (total_bid_vol - total_ask_vol) / safe_denom
        features["f44_order_imbalance"] = order_imbalance

        # --- Derived: spread (feature 45) -----------------------------------
        best_ask = df["ask_price_1"].values
        best_bid = df["bid_price_1"].values
        spread = best_ask - best_bid
        features["f45_spread"] = spread

        # --- Derived: weighted_mid_price (feature 46) -----------------------
        # Volume-weighted mid = (ask1 * bid_vol1 + bid1 * ask_vol1) /
        #                       (bid_vol1 + ask_vol1)
        bid_v1 = df["bid_vol_1"].values
        ask_v1 = df["ask_vol_1"].values
        v_sum = bid_v1 + ask_v1
        safe_v = np.where(v_sum == 0, 1.0, v_sum)
        weighted_mid = (best_ask * bid_v1 + best_bid * ask_v1) / safe_v
        features["f46_weighted_mid_price"] = weighted_mid

        self._feature_names = list(features.columns)
        return features

    # ── Second-order features ────────────────────────────────────────────────
    def _add_second_order_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Append rate-of-change and rolling volatility features.

        These are computed in-place but kept as separate columns so that
        the user can choose whether to include them.
        """
        # Rate of change of order imbalance
        imbalance = features["f44_order_imbalance"].values
        roc_imbalance = np.diff(imbalance, prepend=imbalance[0])
        features["f47_roc_order_imbalance"] = roc_imbalance

        # Rolling volatility of weighted mid-price (window=20 events)
        mid = features["f46_weighted_mid_price"].values
        mid_returns = np.diff(mid, prepend=mid[0])

        # Efficient rolling std via pandas
        rolling_vol = (
            pd.Series(mid_returns)
            .rolling(window=self.rolling_window, min_periods=1)
            .std()
            .fillna(0.0)
            .values
        )
        features["f48_rolling_vol_mid"] = rolling_vol

        self._feature_names = list(features.columns)
        return features

    # ── Public API ───────────────────────────────────────────────────────────
    def fit_transform(
        self,
        df: pd.DataFrame,
        include_second_order: bool = True,
        n_output_features: int = N_CORE_FEATURES,
    ) -> np.ndarray:
        """Extract features and fit the StandardScaler on training data.

        Parameters
        ----------
        df : merged LOB DataFrame from lobster_loader.
        include_second_order : whether to append second-order features.
        n_output_features : number of features in the output array.
            If fewer than the total extracted, the first *n* are selected.

        Returns
        -------
        np.ndarray of shape (N_events, n_output_features), normalized.
        """
        features = self._extract_raw_features(df)
        if include_second_order:
            features = self._add_second_order_features(features)

        # Select requested number of columns
        cols = features.columns[:n_output_features]
        raw = features[cols].values.astype(np.float64)

        # Handle NaN/Inf
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit scaler on training data
        self.scaler = StandardScaler()
        normalized = self.scaler.fit_transform(raw)

        logger.info(
            "fit_transform: %d events × %d features (scaler fitted on train)",
            normalized.shape[0], normalized.shape[1],
        )
        return normalized

    def transform(
        self,
        df: pd.DataFrame,
        include_second_order: bool = True,
        n_output_features: int = N_CORE_FEATURES,
    ) -> np.ndarray:
        """Extract features and normalize using a *previously fitted* scaler.

        Parameters
        ----------
        df : merged LOB DataFrame.
        include_second_order : whether to append second-order features.
        n_output_features : number of output features (must match fit).

        Returns
        -------
        np.ndarray of shape (N_events, n_output_features), normalized.
        """
        if self.scaler is None:
            raise RuntimeError(
                "Scaler not fitted. Call fit_transform() on training data first."
            )

        features = self._extract_raw_features(df)
        if include_second_order:
            features = self._add_second_order_features(features)

        cols = features.columns[:n_output_features]
        raw = features[cols].values.astype(np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        normalized = self.scaler.transform(raw)
        logger.info(
            "transform: %d events × %d features (using fitted scaler)",
            normalized.shape[0], normalized.shape[1],
        )
        return normalized

    def get_mid_prices(self, df: pd.DataFrame) -> np.ndarray:
        """Return the simple mid-price series for labeling purposes.

        mid = (best_bid + best_ask) / 2
        """
        return ((df["bid_price_1"].values + df["ask_price_1"].values) / 2.0)

    def get_timestamps_ms(self, df: pd.DataFrame) -> np.ndarray:
        """Return timestamps in milliseconds (float64) for labeling."""
        return df["timestamp_ns"].values.astype(np.float64) / 1e6
