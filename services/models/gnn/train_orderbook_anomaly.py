"""
train_orderbook_anomaly.py — Train an Isolation Forest on OHLCV-derived
market microstructure proxies to detect liquidity anomalies.

The orderbook is represented as a bipartite graph (bid side ↔ ask side) whose
node features are derived from publicly available OHLCV data:

  Bid-side proxy  : RSI extremes + directional bar (open_close_spread, rsi_14, rsi_7)
  Ask-side proxy  : Volume anomaly (volume_z_score_72, volume_change_1)
  Market breadth  : Volatility + range (atr_pct_14, bb_width, high_low_spread, volatility_12)

Training on 80 K historical bars gives robust anomaly thresholds without
requiring live orderbook history.

Usage:
    python services/models/gnn/train_orderbook_anomaly.py

Outputs:
    services/models/models/gnn_anomaly_detector.pkl   — IsolationForest
    services/models/models/gnn_scaler.pkl             — StandardScaler
    services/models/models/gnn_thresholds.json        — p95/p99 + normalization bounds
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

PROCESSED_CSV = (
    BASE_DIR.parent
    / "classification"
    / "data"
    / "processed"
    / "sol_usdt_15m_features_h4_with_sentiment_orderbook.csv"
)

GNN_DETECTOR_PATH   = MODEL_DIR / "gnn_anomaly_detector.pkl"
GNN_SCALER_PATH     = MODEL_DIR / "gnn_scaler.pkl"
GNN_THRESHOLDS_PATH = MODEL_DIR / "gnn_thresholds.json"

# OHLCV-derived microstructure proxy features (always available historically and live)
GRAPH_FEATURES = [
    "high_low_spread",     # intraday range / price  → bid-ask spread proxy
    "atr_pct_14",          # ATR as % of price       → volatility / stress proxy
    "bb_width",            # Bollinger band width    → regime volatility
    "volume_z_score_72",   # volume vs 72-bar mean   → unusual activity
    "volume_change_1",     # 1-bar volume delta      → sudden pressure
    "rsi_14",              # RSI(14)                 → directional imbalance
    "open_close_spread",   # (close-open)/close      → bar-level pressure indicator
    "volatility_12",       # realised vol 12-bar     → short-term stress
]


def main() -> None:
    print("=" * 60)
    print("GNN Orderbook Anomaly Detector — Training")
    print("=" * 60)

    if not PROCESSED_CSV.exists():
        raise FileNotFoundError(f"Processed CSV not found: {PROCESSED_CSV}")

    df = pd.read_csv(PROCESSED_CSV, usecols=GRAPH_FEATURES)
    df = df.fillna(0.0)

    # Drop rows that are entirely zero (early padding rows)
    mask = (df != 0).any(axis=1)
    df = df[mask].reset_index(drop=True)
    print(f"  Training rows : {len(df):,}")
    print(f"  Features      : {GRAPH_FEATURES}")

    # Chronological 80/20 split — no future leakage
    n_train = int(len(df) * 0.80)
    X_train = df.values[:n_train]
    X_val   = df.values[n_train:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,   # expect ~5% anomalous bars
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s)

    # Flip sign so higher score = more anomalous
    train_raw = -model.decision_function(X_train_s)
    val_raw   = -model.decision_function(X_val_s)

    score_min = float(train_raw.min())
    score_max = float(train_raw.max())

    def normalise(raw: np.ndarray) -> np.ndarray:
        return np.clip((raw - score_min) / max(score_max - score_min, 1e-10), 0.0, 1.0)

    norm_val = normalise(val_raw)

    # Store percentile thresholds for normalization in inference
    train_df = pd.DataFrame(X_train, columns=GRAPH_FEATURES)
    thresholds = {
        "score_min":          score_min,
        "score_max":          score_max,
        "val_p50":            float(np.percentile(norm_val, 50)),
        "val_p95":            float(np.percentile(norm_val, 95)),
        "val_p99":            float(np.percentile(norm_val, 99)),
        "train_p95": {
            f: float(np.percentile(train_df[f].abs(), 95)) for f in GRAPH_FEATURES
        },
        "graph_features":     GRAPH_FEATURES,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  GNN_DETECTOR_PATH)
    joblib.dump(scaler, GNN_SCALER_PATH)
    GNN_THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))

    print(f"\n  Val  p50 anomaly score : {thresholds['val_p50']:.4f}")
    print(f"  Val  p95 anomaly score : {thresholds['val_p95']:.4f}")
    print(f"  Val  p99 anomaly score : {thresholds['val_p99']:.4f}")
    print(f"\n  Model   → {GNN_DETECTOR_PATH.name}")
    print(f"  Scaler  → {GNN_SCALER_PATH.name}")
    print(f"  Config  → {GNN_THRESHOLDS_PATH.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
