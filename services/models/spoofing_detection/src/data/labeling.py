"""
labeling.py — Spoofing Label Generation & Sequence Windowing
=============================================================

Implements the L(t) labeling function from the paper:

    L(t) = 1  if ALL of:
        (1)  V_cancel / V_placed  >  τ₁ = 0.9   (high cancellation intent)
        (2)  Δt_duration          <  τ₂ = 500 ms  (rapid lifecycle)
        (3)  MIM                  >  τ₃ = 1.5     (price manipulation achieved)

    L(t) = 0  otherwise  (legitimate trading)

Where:
    MIM = |mid_price(t + 100ms) − mid_price(t)| / tick_size

Sequence windowing:
    - Window length = 100 events
    - Overlap = 50% (stride = 50 events)
    - Window label = 1 if ANY event in the window has L(t) = 1

Returns (X, y) arrays ready for dataset construction.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default thresholds (from paper) ──────────────────────────────────────────
TAU1_CANCEL_RATIO = 0.9    # cancel-to-placement ratio threshold
TAU2_DURATION_MS = 500.0   # maximum duration in milliseconds
TAU3_MIM = 1.5             # market impact measure threshold
MIM_LOOKAHEAD_MS = 100.0   # lookahead window for MIM calculation

# ── Windowing defaults ───────────────────────────────────────────────────────
WINDOW_LENGTH = 100        # events per sequence
WINDOW_STRIDE = 50         # 50% overlap → stride = window_length / 2


class SpoofingLabeler:
    """Label LOB events as spoofing (1) or legitimate (0).

    Parameters
    ----------
    tau1 : cancel-to-placement ratio threshold (default 0.9).
    tau2 : max duration in ms for spoof cycle (default 500).
    tau3 : MIM threshold (default 1.5).
    tick_size : minimum price increment (default 0.01).
    mim_lookahead_ms : lookahead window for MIM (default 100ms).
    window_length : events per sequence window (default 100).
    window_stride : stride between windows (default 50 for 50% overlap).
    """

    def __init__(
        self,
        tau1: float = TAU1_CANCEL_RATIO,
        tau2: float = TAU2_DURATION_MS,
        tau3: float = TAU3_MIM,
        tick_size: float = 0.01,
        mim_lookahead_ms: float = MIM_LOOKAHEAD_MS,
        window_length: int = WINDOW_LENGTH,
        window_stride: int = WINDOW_STRIDE,
    ):
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.tick_size = tick_size
        self.mim_lookahead_ms = mim_lookahead_ms
        self.window_length = window_length
        self.window_stride = window_stride

    # ── Per-event labeling ───────────────────────────────────────────────────
    def compute_event_labels(
        self,
        df: pd.DataFrame,
        mid_prices: np.ndarray,
        timestamps_ms: np.ndarray,
    ) -> np.ndarray:
        """Compute L(t) for every event in the DataFrame.

        Parameters
        ----------
        df : merged LOB DataFrame (needs 'event_type', 'size' columns).
        mid_prices : array of mid-prices per event.
        timestamps_ms : array of timestamps in milliseconds.

        Returns
        -------
        np.ndarray of shape (N,) with binary labels (0 or 1).
        """
        n = len(df)
        labels = np.zeros(n, dtype=np.int64)

        # --- Precompute per-event cancel/place volumes ---------------------
        event_types = df["event_type"].values
        sizes = df["size"].values.astype(np.float64)

        # Condition 1: Cancel-to-placement ratio
        # We compute this using a rolling window around each event.
        # For each event t, look at the surrounding 100-event context.
        cancel_ratios = self._compute_cancel_ratios(event_types, sizes)

        # Condition 2: Duration of order lifecycle (place → cancel)
        # Approximate as the time gap between consecutive events in a burst
        durations_ms = self._compute_order_durations(df, timestamps_ms)

        # Condition 3: Market Impact Measure (MIM)
        mim_values = self._compute_mim(mid_prices, timestamps_ms)

        # --- Apply the three-condition conjunction -------------------------
        cond1 = cancel_ratios > self.tau1
        cond2 = durations_ms < self.tau2
        cond3 = mim_values > self.tau3

        labels = (cond1 & cond2 & cond3).astype(np.int64)

        n_pos = labels.sum()
        logger.info(
            "Event labels: %d / %d positive (%.2f%%)",
            n_pos, n, 100.0 * n_pos / max(n, 1),
        )
        return labels

    def _compute_cancel_ratios(
        self,
        event_types: np.ndarray,
        sizes: np.ndarray,
        context_window: int = 100,
    ) -> np.ndarray:
        """Compute rolling cancel-to-placement volume ratio.

        event_type 1 = submission (placed)
        event_type 2,3 = partial/full cancellation
        """
        n = len(event_types)
        cancel_ratios = np.zeros(n, dtype=np.float64)

        # Precompute placed and cancelled volume per event
        placed_vol = np.where(event_types == 1, sizes, 0.0)
        cancel_vol = np.where(np.isin(event_types, [2, 3]), sizes, 0.0)

        # Rolling sums using cumulative sums for efficiency
        cum_placed = np.cumsum(placed_vol)
        cum_cancel = np.cumsum(cancel_vol)

        half_w = context_window // 2
        for i in range(n):
            lo = max(0, i - half_w)
            hi = min(n - 1, i + half_w)

            total_placed = cum_placed[hi] - (cum_placed[lo - 1] if lo > 0 else 0)
            total_cancel = cum_cancel[hi] - (cum_cancel[lo - 1] if lo > 0 else 0)

            if total_placed > 0:
                cancel_ratios[i] = total_cancel / total_placed
            else:
                cancel_ratios[i] = 0.0

        return cancel_ratios

    def _compute_order_durations(
        self,
        df: pd.DataFrame,
        timestamps_ms: np.ndarray,
    ) -> np.ndarray:
        """Estimate spoofing cycle duration as time between placement and
        nearest subsequent cancellation for the same order_id.

        For events that are not placements, uses the inter-event delta_t.
        """
        n = len(df)
        durations = np.full(n, np.inf, dtype=np.float64)

        order_ids = df["order_id"].values
        event_types = df["event_type"].values

        # Build a lookup: order_id → list of (index, event_type, timestamp)
        # For efficiency with large datasets, limit lookback
        placement_times: dict[int, float] = {}

        for i in range(n):
            eid = int(event_types[i])
            oid = int(order_ids[i])

            if eid == 1:  # submission
                placement_times[oid] = timestamps_ms[i]
            elif eid in (2, 3):  # cancellation
                if oid in placement_times:
                    durations[i] = timestamps_ms[i] - placement_times[oid]

        # For non-placement/cancellation events, use delta_t
        delta_t = np.diff(timestamps_ms, prepend=timestamps_ms[0])
        no_duration = np.isinf(durations)
        durations[no_duration] = delta_t[no_duration]

        # Clip negatives (shouldn't happen but be safe)
        durations = np.clip(durations, 0.0, None)

        return durations

    def _compute_mim(
        self,
        mid_prices: np.ndarray,
        timestamps_ms: np.ndarray,
    ) -> np.ndarray:
        """Compute Market Impact Measure:
        MIM(t) = |mid_price(t + lookahead) − mid_price(t)| / tick_size

        Finds the event closest to t + mim_lookahead_ms for each event.
        """
        n = len(mid_prices)
        mim = np.zeros(n, dtype=np.float64)

        # For each event, find the index of the event at t + lookahead
        target_times = timestamps_ms + self.mim_lookahead_ms

        # Use searchsorted for efficient lookup
        future_indices = np.searchsorted(timestamps_ms, target_times, side="left")
        future_indices = np.clip(future_indices, 0, n - 1)

        future_mids = mid_prices[future_indices]
        mim = np.abs(future_mids - mid_prices) / self.tick_size

        return mim

    # ── Sequence windowing ───────────────────────────────────────────────────
    def create_sequences(
        self,
        features: np.ndarray,
        event_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Window the feature matrix into overlapping sequences.

        Parameters
        ----------
        features : shape (N_events, n_features) — normalized feature array.
        event_labels : shape (N_events,) — per-event binary labels.

        Returns
        -------
        X : np.ndarray of shape (N_windows, window_length, n_features)
        y : np.ndarray of shape (N_windows,) — 1 if any event in window is spoofing
        """
        n_events, n_features = features.shape
        wl = self.window_length
        stride = self.window_stride

        if n_events < wl:
            logger.warning(
                "Not enough events (%d) for window length %d. "
                "Returning single padded window.", n_events, wl
            )
            padded_x = np.zeros((1, wl, n_features), dtype=np.float32)
            padded_x[0, :n_events, :] = features
            y = np.array([int(event_labels.any())], dtype=np.int64)
            return padded_x, y

        # Compute window start indices
        starts = list(range(0, n_events - wl + 1, stride))
        n_windows = len(starts)

        X = np.zeros((n_windows, wl, n_features), dtype=np.float32)
        y = np.zeros(n_windows, dtype=np.int64)

        for idx, start in enumerate(starts):
            end = start + wl
            X[idx] = features[start:end]
            # Window label = 1 if ANY event in the window is spoofing
            y[idx] = int(event_labels[start:end].any())

        n_pos = y.sum()
        logger.info(
            "Created %d windows (length=%d, stride=%d): %d positive (%.1f%%)",
            n_windows, wl, stride, n_pos, 100.0 * n_pos / max(n_windows, 1),
        )
        return X, y

    # ── End-to-end convenience method ────────────────────────────────────────
    def label_and_window(
        self,
        df: pd.DataFrame,
        features: np.ndarray,
        mid_prices: np.ndarray,
        timestamps_ms: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full pipeline: compute event labels → window into sequences.

        Parameters
        ----------
        df : merged LOB DataFrame.
        features : normalized feature array (N_events, n_features).
        mid_prices : mid-price per event.
        timestamps_ms : timestamp in ms per event.

        Returns
        -------
        X : (N_windows, window_length, n_features)
        y : (N_windows,) binary labels
        """
        event_labels = self.compute_event_labels(df, mid_prices, timestamps_ms)
        return self.create_sequences(features, event_labels)
