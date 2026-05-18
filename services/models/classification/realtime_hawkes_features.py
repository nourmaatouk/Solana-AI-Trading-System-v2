"""
realtime_hawkes_features.py — add Hawkes intensity features to the realtime feature CSV.

Reads : data/realtime/latest_sol_15m_features.csv  (output of realtime_feature_builder.py)
Writes: same file in-place, adding / updating two columns:
    hawkes_mu              — baseline (background) intensity from the Hawkes MLE
    hawkes_branching_ratio — alpha/beta; near 1 means highly self-exciting order flow

Only rows with at least HAWKES_WINDOW candles of history are computed; earlier
rows are set to 0.0.  With CANDLE_LIMIT_REALTIME=200 and HAWKES_WINDOW=64 this
runs ~136 fits at ≈1.4 ms each — under 0.2 s total, safe for the 15-min cycle.

The logic (volume → synthetic event times → univariate Hawkes MLE) is identical
to 03_hawkes_intensity.py so the realtime features are on the same scale as the
features the XGBoost model was trained on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "realtime" / "latest_sol_15m_features.csv"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from realtime_spoofing_detector import fit_hawkes  # noqa: E402

# ── Hawkes hyper-params (must match 03_hawkes_intensity.py) ───────────────────
HAWKES_WINDOW = 64     # rolling window in candles
MIN_EVENTS    = 10     # minimum synthetic events for a stable MLE
TARGET_EVENTS = 200    # target total events per window (controls α/β scale)


# ─────────────────────────────────────────────────────────────────────────────
def _volume_to_event_times(vol: np.ndarray) -> np.ndarray:
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
    """Return (hawkes_mu, hawkes_branching_ratio) for one volume window."""
    times = _volume_to_event_times(vol)
    if len(times) < MIN_EVENTS:
        return 0.0, 0.0
    try:
        mu, _alpha, branching = fit_hawkes(times)
        return float(mu), float(branching)
    except Exception:
        return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("realtime_hawkes_features.py")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature CSV not found: {INPUT_PATH}\n"
            "Run realtime_feature_builder.py first."
        )

    df = pd.read_csv(INPUT_PATH)
    print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}")

    if "volume" not in df.columns:
        raise KeyError("Column 'volume' not found — cannot compute Hawkes features.")

    volume = df["volume"].fillna(0.0).to_numpy(dtype=np.float64)
    n = len(volume)

    mu_arr = np.zeros(n, dtype=np.float64)
    br_arr = np.zeros(n, dtype=np.float64)

    for i in range(HAWKES_WINDOW, n):
        mu_arr[i], br_arr[i] = _fit_window(volume[i - HAWKES_WINDOW : i])

    df["hawkes_mu"]              = mu_arr
    df["hawkes_branching_ratio"] = br_arr

    latest_mu = mu_arr[-1]
    latest_br = br_arr[-1]
    print(f"  hawkes_mu={latest_mu:.4f}   hawkes_branching_ratio={latest_br:.4f}")

    df.to_csv(INPUT_PATH, index=False)
    print(f"  Saved → {INPUT_PATH.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
