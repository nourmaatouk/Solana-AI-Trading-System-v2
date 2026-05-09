"""
hawkes_updater.py — Live Hawkes adjacency refresh thread
=========================================================

Runs a background daemon thread that accumulates incoming event timestamps
and re-fits the Hawkes MLE every ``refresh_interval_ms`` milliseconds
(default 500 ms).  The inferred adjacency matrix is exposed via a
thread-safe property so the inference engine can always read the latest
estimate without blocking.

Usage
-----
    from src.deployment.hawkes_updater import HawkesUpdater

    updater = HawkesUpdater(n_assets=3, refresh_interval_ms=500)
    updater.start()

    # Push new events as they arrive (asset_idx is 0-based)
    updater.push_event(asset_idx=0, timestamp_minutes=1.234)

    # Read latest adjacency (safe from any thread)
    adj = updater.adjacency   # numpy (n_assets, n_assets) float32

    updater.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from src.models.hawkes import HawkesEstimator

logger = logging.getLogger(__name__)

# Maximum events kept per asset for the sliding window fed to Hawkes MLE.
_MAX_EVENTS_PER_ASSET = 2_000


class HawkesUpdater:
    """Background thread that continuously refreshes the Hawkes adjacency matrix.

    Parameters
    ----------
    n_assets : int
        Number of assets in the graph (must match the trained model).
    refresh_interval_ms : float
        How often (in ms) to re-run Hawkes MLE.  Default 500 ms.
    min_events_per_asset : int
        Minimum number of events needed per asset before the first fit.
        Below this threshold the identity matrix is returned.
    """

    def __init__(
        self,
        n_assets: int = 3,
        refresh_interval_ms: float = 500.0,
        min_events_per_asset: int = 50,
    ) -> None:
        self._n_assets = n_assets
        self._interval = refresh_interval_ms / 1000.0  # seconds
        self._min_events = min_events_per_asset

        # Per-asset sliding window of event timestamps (minutes since global t0)
        self._buffers: Dict[int, Deque[float]] = {
            i: deque(maxlen=_MAX_EVENTS_PER_ASSET) for i in range(n_assets)
        }
        self._lock = threading.Lock()

        # Latest adjacency — starts as identity, updated in background
        self._adjacency: np.ndarray = np.eye(n_assets, dtype=np.float32)
        self._adj_lock = threading.RLock()

        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ── Public API ────────────────────────────────────────────────────────────

    def push_event(self, asset_idx: int, timestamp_minutes: float) -> None:
        """Add a new event for the given asset (thread-safe)."""
        if asset_idx < 0 or asset_idx >= self._n_assets:
            raise ValueError(f"asset_idx {asset_idx} out of range [0, {self._n_assets})")
        with self._lock:
            self._buffers[asset_idx].append(timestamp_minutes)

    def push_events_batch(self, asset_idx: int, timestamps_minutes: np.ndarray) -> None:
        """Add multiple events at once (thread-safe)."""
        with self._lock:
            for t in timestamps_minutes:
                self._buffers[asset_idx].append(float(t))

    @property
    def adjacency(self) -> np.ndarray:
        """Return the most recently computed adjacency matrix (read-only copy)."""
        with self._adj_lock:
            return self._adjacency.copy()

    def start(self) -> None:
        """Start the background refresh thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="HawkesUpdater")
        self._thread.start()
        logger.info("HawkesUpdater started (refresh=%.0f ms, n_assets=%d)", self._interval * 1000, self._n_assets)

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the background thread to stop and wait for it."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("HawkesUpdater stopped")

    # ── Background loop ───────────────────────────────────────────────────────

    def _run(self) -> None:
        while self._running:
            t0 = time.monotonic()
            self._refresh()
            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, self._interval - elapsed)
            time.sleep(sleep_time)

    def _refresh(self) -> None:
        with self._lock:
            snapshot: Dict[int, list] = {
                i: list(buf) for i, buf in self._buffers.items()
            }

        # Check we have enough events
        if any(len(v) < self._min_events for v in snapshot.values()):
            return  # Not enough data yet — keep identity

        try:
            estimator = HawkesEstimator(n_assets=self._n_assets)
            estimator.fit(snapshot)
            new_adj = estimator.get_adjacency().astype(np.float32)
            with self._adj_lock:
                self._adjacency = new_adj
            logger.debug("HawkesUpdater: adjacency refreshed\n%s", np.round(new_adj, 3))
        except Exception as exc:
            logger.warning("HawkesUpdater: MLE failed (%s) — keeping previous adjacency", exc)
