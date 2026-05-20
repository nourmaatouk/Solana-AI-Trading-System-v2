"""
realtime_gnn_predictor.py — Live GNN spatial encoder inference.

Reads OHLCV-derived microstructure proxy features from the latest features CSV,
runs the trained Isolation Forest to produce a graph anomaly score, derives
liquidity_stress from volatility and volume metrics, and derives
community_sentiment from BTC momentum features.

Writes:
    data/realtime/last_gnn_signal.json

Exit codes:
    0 — signal written
    1 — model not found (run train_orderbook_anomaly.py first)
    2 — data error
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR     = Path(__file__).resolve().parent
MODEL_DIR    = BASE_DIR.parent / "models"
REALTIME_DIR = BASE_DIR.parent / "classification" / "data" / "realtime"

GNN_DETECTOR_PATH   = MODEL_DIR / "gnn_anomaly_detector.pkl"
GNN_SCALER_PATH     = MODEL_DIR / "gnn_scaler.pkl"
GNN_THRESHOLDS_PATH = MODEL_DIR / "gnn_thresholds.json"

FEATURES_CSV = REALTIME_DIR / "latest_features_with_sentiment.csv"
OUTPUT_PATH  = REALTIME_DIR / "last_gnn_signal.json"

BTC_SCALE    = 20.0   # 5% BTC return → community_sentiment = ±1

GRAPH_FEATURES = [
    "high_low_spread",
    "atr_pct_14",
    "bb_width",
    "volume_z_score_72",
    "volume_change_1",
    "rsi_14",
    "open_close_spread",
    "volatility_12",
]


def _load_models():
    if not GNN_DETECTOR_PATH.exists() or not GNN_SCALER_PATH.exists():
        print("  [gnn] Model not found — run train_orderbook_anomaly.py first.")
        return None, None, None
    model      = joblib.load(GNN_DETECTOR_PATH)
    scaler     = joblib.load(GNN_SCALER_PATH)
    thresholds = json.loads(GNN_THRESHOLDS_PATH.read_text())
    return model, scaler, thresholds


def _latest_feature_row() -> pd.Series | None:
    if not FEATURES_CSV.exists():
        print("  [gnn] Features CSV not found.")
        return None
    df = pd.read_csv(FEATURES_CSV)
    if df.empty:
        print("  [gnn] Features CSV is empty.")
        return None
    return df.iloc[-1]


def _anomaly_score(model, scaler, thresholds, row: pd.Series) -> float:
    x = np.array([[float(row.get(f, 0.0)) for f in GRAPH_FEATURES]])
    x_scaled = scaler.transform(x)
    raw = float(-model.decision_function(x_scaled)[0])   # higher = more anomalous
    score_min = thresholds["score_min"]
    score_max = thresholds["score_max"]
    return float(np.clip((raw - score_min) / max(score_max - score_min, 1e-10), 0.0, 1.0))


def _liquidity_stress(row: pd.Series, thresholds: dict) -> float:
    """Derive liquidity stress from OHLCV microstructure proxies."""
    p95 = thresholds.get("train_p95", {})

    atr       = abs(float(row.get("atr_pct_14",       0.0)))
    bb_w      = abs(float(row.get("bb_width",          0.0)))
    vol_z     = abs(float(row.get("volume_z_score_72", 0.0)))

    atr_norm   = np.clip(atr  / max(p95.get("atr_pct_14",       0.01), 1e-8), 0.0, 1.0)
    bb_norm    = np.clip(bb_w / max(p95.get("bb_width",          0.01), 1e-8), 0.0, 1.0)
    vol_norm   = np.clip(vol_z / 2.0, 0.0, 1.0)   # z > 2 = full stress

    stress = 0.40 * float(atr_norm) + 0.35 * float(bb_norm) + 0.25 * float(vol_norm)
    return float(np.clip(stress, 0.0, 1.0))


def _community_sentiment(row: pd.Series) -> float:
    """Cross-asset sentiment from BTC short-term return (4-bar = 1h)."""
    btc_ret = float(row.get("btc_return_4", 0.0))
    return float(np.clip(btc_ret * BTC_SCALE, -1.0, 1.0))


def main() -> int:
    print("=" * 60)
    print("GNN Spatial Encoder — Realtime Inference")
    print("=" * 60)

    model, scaler, thresholds = _load_models()
    if model is None:
        return 1

    row = _latest_feature_row()
    if row is None:
        return 2

    anomaly   = _anomaly_score(model, scaler, thresholds, row)
    stress    = _liquidity_stress(row, thresholds)
    community = _community_sentiment(row)

    payload = {
        "generated_at_utc":    datetime.now(timezone.utc).isoformat(),
        "anomaly_score":       round(anomaly,   4),
        "liquidity_stress":    round(stress,    4),
        "community_sentiment": round(community, 4),
        "inputs": {
            f: round(float(row.get(f, 0.0)), 6) for f in GRAPH_FEATURES
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))

    print(f"  Anomaly score      : {anomaly:.4f}")
    print(f"  Liquidity stress   : {stress:.4f}")
    print(f"  Community sentiment: {community:+.4f}")
    print(f"  Saved → {OUTPUT_PATH}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
