"""
realtime_spoofing_detector.py — Pre-trade spoofing gate

Reads the live orderbook snapshot (collected by realtime_orderbook_collector.py),
fits a univariate Hawkes process on the event timestamps, and reports the
branching ratio as a spoofing proxy.

A branching ratio near 1 indicates self-exciting, reflexive order flow:
each burst of activity triggers another, consistent with spoofing (orders
placed and cancelled rapidly to create false impressions of liquidity).

Optionally loads SpoofingInferenceEngine (TEN-GNN) if the checkpoint exists at
  services/models/spoofing_detection/checkpoints/tengnn_btc_eth_sol_1min/best_model.pt
(the TEN-GNN requires trained weights; when absent the Hawkes score alone gates.)

Writes: data/realtime/spoofing_score.json
    {
      "generated_at_utc": "...",
      "status":           "ok" | "no_data" | "error",
      "n_events":         <int>,
      "branching_ratio":  <float | null>,
      "is_suspicious":    <bool>,
      "spoofing_probability": <float | null>,   # TEN output if available
      "ten_threshold":        <float | null>,   # calibrated threshold embedded in checkpoint
      "message":          "..."
    }
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

BASE_DIR  = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]

ORDERBOOK_CSV  = BASE_DIR / "data" / "realtime" / "orderbook_features.csv"
SPOOFING_OUT   = BASE_DIR / "data" / "realtime" / "spoofing_score.json"
TEN_SPOOF_PATH = (
    REPO_ROOT / "services" / "models" / "spoofing_detection"
    / "checkpoints" / "tengnn_btc_eth_sol_1min" / "best_model.pt"
)

# ── Hawkes hyperparameters ────────────────────────────────────────────────────
# β controls the decay of self-excitation.  With 1-minute aggregated orderbook
# rows, β=0.1 per minute gives a decay half-life of ≈7 minutes — each event
# influences the next 7+ rows, which is long enough to detect bursty periods.
BETA                = 0.1    # per minute
BRANCHING_THRESHOLD = 0.85   # flag suspicious if α/β >= this
MIN_EVENTS          = 10     # minimum rows for a stable Hawkes estimate
LOOKBACK_ROWS       = 90     # use last N rows of orderbook CSV (≈90 min)
MAX_ORDERBOOK_AGE_MINUTES = int(os.getenv("SPOOFING_MAX_ORDERBOOK_AGE_MINUTES", "30"))


# ─────────────────────────────────────────────────────────────────────────────
# UNIVARIATE HAWKES MLE
# ─────────────────────────────────────────────────────────────────────────────
def _precompute_R(times: np.ndarray, beta: float) -> np.ndarray:
    """R[k] = Σ_{j<k} exp(-β·(t_k − t_j)) via O(N) recurrence."""
    R = np.zeros(len(times))
    for k in range(1, len(times)):
        R[k] = np.exp(-beta * (times[k] - times[k - 1])) * (R[k - 1] + 1.0)
    return R


def _nll(theta: np.ndarray, times: np.ndarray, R: np.ndarray, beta: float) -> float:
    mu, alpha = float(theta[0]), float(theta[1])
    if mu <= 0.0 or alpha < 0.0:
        return 1e12
    T      = times[-1]
    comp   = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - times)))
    lam    = np.maximum(mu + alpha * R, 1e-10)
    return float(comp - np.sum(np.log(lam)))


def fit_hawkes(times: np.ndarray) -> tuple[float, float, float]:
    """Fit univariate Hawkes (μ, α) with fixed β.

    Returns (mu_hat, alpha_hat, branching_ratio=alpha/beta).
    β is fixed to avoid identifiability issues on short windows.
    """
    R    = _precompute_R(times, BETA)
    T    = times[-1]
    rate = len(times) / max(T, 1.0)

    res = minimize(
        _nll,
        x0=[rate * 0.8, rate * 0.05],
        args=(times, R, BETA),
        method="L-BFGS-B",
        # Keep α < 0.99·β so the process stays sub-critical (stationary)
        bounds=[(1e-6, None), (0.0, 0.99 * BETA)],
        options={"maxiter": 300, "ftol": 1e-7},
    )
    mu_hat    = float(res.x[0])
    alpha_hat = float(res.x[1])
    branching = alpha_hat / BETA
    return mu_hat, alpha_hat, branching


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL TEN SPOOFING DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
def _load_ten_spoof():
    """Load SpoofingInferenceEngine from the TEN-GNN checkpoint. Returns None if unavailable."""
    if not TEN_SPOOF_PATH.exists():
        return None
    try:
        spoof_src = REPO_ROOT / "services" / "models" / "spoofing_detection"
        if str(spoof_src) not in sys.path:
            sys.path.insert(0, str(spoof_src))
        from src.deployment.inference_engine import SpoofingInferenceEngine

        engine = SpoofingInferenceEngine.from_checkpoint(TEN_SPOOF_PATH)
        return engine
    except Exception as e:
        print(f"  TEN spoofing load failed: {e}")
        return None


# Training feature indices (n_levels=15 → 34 features):
#   0: midpoint  1: spread  2..31: bid/ask notional at levels 1-15  32: imbalance  33: cancel_ratio
# Map the realtime CSV columns to their closest training-feature positions.
_REALTIME_FEATURE_MAP: dict[str, int] = {
    "mid_price":       0,   # midpoint
    "spread_mean":     1,   # spread
    "bid_volume_mean": 2,   # best-bid notional (level-1 proxy)
    "ask_volume_mean": 3,   # best-ask notional (level-1 proxy)
    "imbalance_mean":  32,  # order imbalance
}
# Asset slot order must match training: --assets BTC ETH SOL
_SOL_ASSET_IDX = 2


def _ten_inference(engine, df: pd.DataFrame) -> float | None:
    """Run SpoofingInferenceEngine (TEN-GNN) on the last 100 orderbook rows.

    The checkpoint was trained on 3-asset (BTC/ETH/SOL) LOB data with 34
    features per timestep.  In realtime only SOL data is available (~5 mapped
    features), so:
      - SOL slot (index 2) receives the mapped and per-window z-scored features.
      - BTC/ETH slots (indices 0, 1) remain zeros — honest absence, not
        fabricated values.  The cross-asset graph edges contribute zero, which
        is correct given missing data; the SOL self-attention still operates.
    Per-window z-scoring aligns the scale with training (StandardScaler).

    Returns P(spoofing) in [0, 1] or None on error.
    """
    try:
        import torch

        SEQ      = 100
        d_input  = engine._model.d_input   # read from checkpoint (34 after retrain)
        n_assets = engine._model.n_assets  # 3 from checkpoint config

        window = df.tail(SEQ).reset_index(drop=True)
        n_rows = len(window)

        sol_seq = np.zeros((SEQ, d_input), dtype=np.float32)

        for col, feat_idx in _REALTIME_FEATURE_MAP.items():
            if col not in window.columns:
                continue
            vals = window[col].fillna(0.0).to_numpy(dtype=np.float32)
            # Place values right-aligned in the SEQ dimension
            arr = np.zeros(SEQ, dtype=np.float32)
            arr[SEQ - n_rows:] = vals
            # Per-window z-score — matches StandardScaler used during training
            std = arr.std()
            if std > 1e-8:
                arr = (arr - arr.mean()) / std
            sol_seq[:, feat_idx] = arr

        # Shape: (1, n_assets, SEQ, d_input)
        x = np.zeros((1, n_assets, SEQ, d_input), dtype=np.float32)
        x[0, _SOL_ASSET_IDX] = sol_seq

        probs = engine.infer(torch.tensor(x), target_asset_idx=_SOL_ASSET_IDX)  # np.ndarray (1,)
        return float(probs[0])
    except Exception as e:
        print(f"  TEN inference error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 60)
    print("SPOOFING DETECTOR")
    print("=" * 60)

    out: dict = {
        "generated_at_utc":    datetime.now(timezone.utc).isoformat(),
        "status":              "ok",
        "n_events":            0,
        "branching_ratio":     None,
        "is_suspicious":       False,
        "spoofing_probability": None,
        "ten_threshold":       None,
        "latest_orderbook_timestamp": None,
        "orderbook_age_minutes": None,
        "max_orderbook_age_minutes": MAX_ORDERBOOK_AGE_MINUTES,
        "message":             "",
    }

    # ── Guard: orderbook data ─────────────────────────────────────────────────
    if not ORDERBOOK_CSV.exists():
        _finish(out, "no_data", f"Orderbook CSV not found: {ORDERBOOK_CSV}")
        return 0

    df = pd.read_csv(ORDERBOOK_CSV)
    if df.empty or "timestamp" not in df.columns:
        _finish(out, "no_data", "Orderbook CSV is empty or missing timestamp column")
        return 0

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["timestamp"]).copy()
    if df.empty:
        _finish(out, "no_data", "Orderbook CSV has no parseable timestamps")
        return 0

    latest_ts = df["timestamp"].max()
    age_minutes = (datetime.now(timezone.utc) - latest_ts.to_pydatetime()).total_seconds() / 60.0
    out["latest_orderbook_timestamp"] = latest_ts.isoformat()
    out["orderbook_age_minutes"] = round(float(age_minutes), 2)
    if age_minutes > MAX_ORDERBOOK_AGE_MINUTES:
        _finish(
            out,
            "stale_orderbook",
            (
                f"Orderbook data is {age_minutes:.1f} min old "
                f"(max={MAX_ORDERBOOK_AGE_MINUTES}); spoofing score skipped"
            ),
        )
        return 0

    lookback_cutoff = latest_ts - pd.Timedelta(minutes=LOOKBACK_ROWS)
    df = df[df["timestamp"] >= lookback_cutoff].copy()
    df = df.tail(LOOKBACK_ROWS).copy()

    # ── Parse timestamps → minutes since first event ──────────────────────────
    ts = df["timestamp"].dropna()
    if len(ts) < MIN_EVENTS:
        msg = f"Only {len(ts)} valid timestamps — need ≥{MIN_EVENTS} for stable estimate"
        print(f"  WARNING: {msg}")
        out["n_events"] = int(len(ts))
        _finish(out, "ok", msg)
        return 0

    times = (ts - ts.iloc[0]).dt.total_seconds().values / 60.0   # minutes
    times = np.sort(times)
    n     = int(len(times))
    out["n_events"] = n

    print(f"  Orderbook rows : {n}  (window={times[-1]:.1f} min)")

    # ── Hawkes fit ────────────────────────────────────────────────────────────
    try:
        mu, alpha, branching = fit_hawkes(times)
    except Exception as e:
        _finish(out, "error", f"Hawkes fit failed: {e}")
        return 0

    out["branching_ratio"] = round(float(branching), 4)
    is_suspicious          = branching >= BRANCHING_THRESHOLD
    out["is_suspicious"]   = bool(is_suspicious)

    print(f"  Hawkes μ={mu:.4f}  α={alpha:.4f}  β={BETA}  →  "
          f"branching_ratio={branching:.4f}  (threshold={BRANCHING_THRESHOLD})")

    msg = (
        f"SUSPICIOUS: branching_ratio={branching:.4f} >= {BRANCHING_THRESHOLD}"
        if is_suspicious
        else f"OK: branching_ratio={branching:.4f} < {BRANCHING_THRESHOLD}"
    )
    print(f"  Status : {msg}")
    out["message"] = msg

    # ── TEN path (optional) ───────────────────────────────────────────────────
    ten_model = _load_ten_spoof()
    if ten_model is not None:
        prob_spoof = _ten_inference(ten_model, df)
        if prob_spoof is not None:
            threshold = getattr(ten_model, "_threshold", 0.5)
            out["spoofing_probability"] = round(prob_spoof, 4)
            out["ten_threshold"] = round(threshold, 4)
            print(f"  TEN P(spoof) = {prob_spoof:.4f}  (threshold={threshold:.2f})")
            if prob_spoof >= threshold:
                out["is_suspicious"] = True
                out["message"] += f" | TEN P(spoof)={prob_spoof:.4f} >= {threshold:.2f}"
    else:
        print("  TEN spoofing model : not available (Hawkes only)")

    _write(out)
    print(f"\n  Written → {SPOOFING_OUT}")
    print("=" * 60)
    return 0


def _finish(out: dict, status: str, message: str) -> None:
    out["status"]  = status
    out["message"] = message
    print(f"  {message}")
    _write(out)


def _write(out: dict) -> None:
    SPOOFING_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SPOOFING_OUT, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
