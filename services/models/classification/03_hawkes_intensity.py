"""
03_hawkes_intensity.py — Layer 2: Hawkes process intensity features

Reads : PROCESSED_DIR / TECH_FEATURES_CSV   (output of 02_build_technical_features.py)
Writes: same file in-place, adding two columns:
    hawkes_mu              — baseline (background) intensity from the Hawkes MLE
    hawkes_branching_ratio — alpha/beta; near 1 means highly self-exciting order flow

Method
------
Volume is used as a proxy for event arrivals.  For each rolling window of
HAWKES_WINDOW candles ending at row t, the candle volumes are converted into
synthetic event arrival times (candle i contributes round(vol[i]/scale) events
spread uniformly within [i, i+0.999]).  The univariate Hawkes MLE is then fit
on those event times; μ and branching ratio are recorded for row t.

The fit_hawkes / _precompute_R / _nll functions are imported directly from
realtime_spoofing_detector.py (single source of truth) rather than duplicated.

Prerequisites : 02_build_technical_features.py must have run.
Runtime       : ~4–5 minutes for ~186 k rows (≈1.4 ms / window).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ───────────────────────────────────────────────────────────
CLASSIFICATION_DIR = Path(__file__).resolve().parent
if str(CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_DIR))

from config import PROCESSED_DIR, TECH_FEATURES_CSV, HAWKES_FEATURES  # noqa: E402
from realtime_spoofing_detector import fit_hawkes  # noqa: E402

# ── Hawkes window ─────────────────────────────────────────────────────────────
HAWKES_WINDOW = 64    # rolling window in candles
MIN_EVENTS    = 10    # minimum synthetic events for a stable MLE
TARGET_EVENTS = 200   # target total events per window (controls α/β scale)


# ─────────────────────────────────────────────────────────────────────────────
def _volume_to_event_times(vol: np.ndarray) -> np.ndarray:
    """Convert a 1-D volume array into synthetic Hawkes arrival times.

    Each candle i contributes n_i = round(vol[i] / scale) events placed
    at i + linspace(0, 0.999, n_i).  The scale is chosen so that the
    total across the window is ≈ TARGET_EVENTS.
    """
    total = float(vol.sum())
    if total <= 0:
        return np.array([], dtype=np.float64)

    scale = max(total / TARGET_EVENTS, 1.0)
    times: list[float] = []
    for i, v in enumerate(vol):
        n = max(1, int(round(float(v) / scale)))
        if n == 1:
            times.append(float(i))
        else:
            times.extend((float(i) + np.linspace(0.0, 0.999, n)).tolist())

    return np.sort(np.asarray(times, dtype=np.float64))


def _fit_window(vol: np.ndarray) -> tuple[float, float]:
    """Fit Hawkes on a single volume window.

    Returns (hawkes_mu, hawkes_branching_ratio) or (0.0, 0.0) when the
    window has insufficient data or the optimiser fails.
    """
    times = _volume_to_event_times(vol)
    if len(times) < MIN_EVENTS:
        return 0.0, 0.0
    try:
        mu, _alpha, branching = fit_hawkes(times)
        return float(mu), float(branching)
    except Exception:
        return 0.0, 0.0


def compute_hawkes_features(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Hawkes features over the full volume series.

    Returns (hawkes_mu, hawkes_branching_ratio) arrays of length len(volume).
    Rows before index HAWKES_WINDOW are filled with 0.0.
    """
    n = len(volume)
    mu_out  = np.zeros(n, dtype=np.float64)
    br_out  = np.zeros(n, dtype=np.float64)

    report_step = max(1, (n - HAWKES_WINDOW) // 20)  # ~5% progress intervals
    for i in range(HAWKES_WINDOW, n):
        if (i - HAWKES_WINDOW) % report_step == 0:
            pct = 100.0 * (i - HAWKES_WINDOW) / (n - HAWKES_WINDOW)
            print(f"  {pct:5.1f}%  row {i}/{n}", end="\r", flush=True)
        vol_window = volume[i - HAWKES_WINDOW : i]
        mu_out[i], br_out[i] = _fit_window(vol_window)

    print()  # newline after progress
    return mu_out, br_out


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("03_hawkes_intensity.py")
    print("=" * 60)

    csv_path = PROCESSED_DIR / TECH_FEATURES_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Technical features CSV not found: {csv_path}\n"
            "Run 02_build_technical_features.py first."
        )

    print(f"  Loading {csv_path.name} …")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")

    if "volume" not in df.columns:
        raise KeyError("Column 'volume' not found in CSV.")

    # Warn if columns already exist — will overwrite
    existing = [c for c in HAWKES_FEATURES if c in df.columns]
    if existing:
        print(f"  WARNING: overwriting existing columns: {existing}")

    volume = df["volume"].fillna(0.0).to_numpy(dtype=np.float64)

    print(f"  Computing rolling Hawkes features  (window={HAWKES_WINDOW} candles) …")
    mu_arr, br_arr = compute_hawkes_features(volume)

    df["hawkes_mu"]              = mu_arr
    df["hawkes_branching_ratio"] = br_arr

    # ── stats ─────────────────────────────────────────────────────────────────
    non_zero = (br_arr > 0).sum()
    print(f"\n  hawkes_mu             : mean={mu_arr[mu_arr>0].mean():.4f}  "
          f"max={mu_arr.max():.4f}")
    print(f"  hawkes_branching_ratio: mean={br_arr[br_arr>0].mean():.4f}  "
          f"max={br_arr.max():.4f}  "
          f"({non_zero:,}/{len(br_arr):,} non-zero rows)")

    # ── NON_STATIONARY check ──────────────────────────────────────────────────
    for col, arr in [("hawkes_mu", mu_arr), ("hawkes_branching_ratio", br_arr)]:
        nz = arr[arr > 0]
        if len(nz) and abs(nz.mean()) > 10:
            print(f"  WARNING: |mean({col})| > 10 — consider adding to "
                  f"NON_STATIONARY_COLUMNS in config.py")

    print(f"\n  Saving → {csv_path.name} …")
    df.to_csv(csv_path, index=False)
    print(f"  Done.  Shape: {df.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
