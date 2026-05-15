"""
realtime_fusion_predictor.py — Layer 3 Fusion Signal

Combines three independent signals into a single probability and trading signal:

  1. XGBoost   — P(UP) from the production gradient-boosted tree
  2. DNN       — P(UP) from the trained MLP (complementary inductive bias)
  3. TEN       — P(UP) from the Temporal Encoder Network (sequence model,
                 only when fusion metadata explicitly declares TEN support)
  4. Sentiment — small directional nudge from FinBERT news score

Fusion:
  P_stack  = logistic_stacker.predict_proba([P_xgb, P_dnn, P_ten])[:, 1]
  P_final  = clip(P_stack + sentiment_nudge, 0, 1)

Uses a 2-model stacker [P_xgb, P_dnn] unless fusion_meta_h{H}.json declares a
3-model stacker and contains the required TEN architecture metadata.  A
ten_h{H}.pt file alone is not enough to enable TEN.

The sentiment nudge is capped at ±0.04 so it can't flip a strong model
signal but can push borderline cases over/under the confidence gate.

Reads:
  data/realtime/last_signal.json                    — XGBoost P(UP)
  data/realtime/latest_features_with_sentiment.csv  — scaled features + sentiment
  models/dnn_h{H}.pt                               — MLP state dict
  models/ten_h{H}.pt                               — TEN state dict (optional)
  models/fusion_stacker_h{H}.pkl                   — logistic stacker
  models/fusion_meta_h{H}.json                     — architecture + threshold

Writes:
  data/realtime/last_fusion_signal.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    ACTIVE_HORIZON as HORIZON,
    CONFIDENCE_THRESHOLD,
    MODEL_DIR,
    PROCESSED_DIR,
    REALTIME_DIR,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURE_PATH       = REALTIME_DIR / "latest_features_with_sentiment.csv"
XGB_SIGNAL_PATH    = REALTIME_DIR / "last_signal.json"
FUSION_SIGNAL_PATH = REALTIME_DIR / "last_fusion_signal.json"
FEATURE_COLS_PATH  = PROCESSED_DIR / "feature_columns.json"
SCALER_PATH        = PROCESSED_DIR / f"scaler_h{HORIZON}.pkl"
DNN_PATH           = MODEL_DIR / f"dnn_h{HORIZON}.pt"
TEN_PATH           = MODEL_DIR / f"ten_h{HORIZON}.pt"
STACKER_PATH       = MODEL_DIR / f"fusion_stacker_h{HORIZON}.pkl"
META_PATH          = MODEL_DIR / f"fusion_meta_h{HORIZON}.json"

SENTIMENT_CAP = 0.04
TEN_META_KEYS = {
    "seq_len",
    "ten_d_model",
    "ten_n_heads",
    "ten_n_layers",
    "ten_d_ff",
    "ten_dropout",
}


# ─────────────────────────────────────────────────────────────────────────────
# MLP — must match train_dnn_fusion.py exactly
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# TEN — must match train_ten_fusion.py exactly
# ─────────────────────────────────────────────────────────────────────────────
class TENClassifier(nn.Module):
    def __init__(self, d_input: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float, seq_len: int):
        super().__init__()
        self.embed   = nn.Linear(d_input, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) + self.pos_emb[:, :x.size(1)]
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, Path):
        return str(v)
    return v


def _stacker_feature_count(stacker) -> int:
    """Return the number of inputs expected by the saved sklearn stacker."""
    count = getattr(stacker, "n_features_in_", None)
    if count is None and hasattr(stacker, "coef_"):
        count = stacker.coef_.shape[1]
    if count is None:
        raise ValueError(
            "Fusion artifact contract error: cannot determine stacker input "
            "dimension. Regenerate fusion_stacker_h4.pkl with current training code."
        )
    return int(count)


def _models_declared_by_meta(meta: dict) -> list[str]:
    """Return the model inputs declared by fusion metadata.

    Legacy metadata created by train_dnn_fusion.py has no ``models`` key and
    only stores xgb/dnn stacker weights.  Treat that as an explicit 2-input
    production contract.
    """
    models = meta.get("models")
    if isinstance(models, list) and models:
        return [str(m).lower() for m in models]

    weights = meta.get("stacker_weights", {})
    if isinstance(weights, dict) and "ten" in weights:
        return ["xgb", "dnn", "ten"]
    return ["xgb", "dnn"]


def _should_use_ten(meta: dict, stacker, ten_path: Path = TEN_PATH) -> bool:
    """Validate fusion artifact consistency and decide whether TEN is enabled."""
    declared_models = _models_declared_by_meta(meta)
    stacker_inputs = _stacker_feature_count(stacker)
    declares_ten = "ten" in declared_models

    if declares_ten:
        missing_meta = sorted(k for k in TEN_META_KEYS if k not in meta)
        errors = []
        if stacker_inputs != 3:
            errors.append(
                f"metadata declares TEN but stacker expects {stacker_inputs} input(s), not 3"
            )
        if missing_meta:
            errors.append(f"missing TEN metadata keys: {missing_meta}")
        if not ten_path.exists():
            errors.append(f"TEN weights missing: {ten_path}")
        if errors:
            raise ValueError(
                "Fusion artifact contract error: "
                + "; ".join(errors)
                + ". Regenerate fusion artifacts with train_ten_fusion.py."
            )
        return True

    if stacker_inputs == 3:
        raise ValueError(
            "Fusion artifact contract error: stacker expects 3 inputs but "
            "fusion metadata does not declare TEN. Regenerate fusion metadata "
            "with train_ten_fusion.py or restore a 2-input stacker."
        )
    if stacker_inputs != 2:
        raise ValueError(
            f"Fusion artifact contract error: unsupported stacker input count "
            f"{stacker_inputs}. Expected 2 (XGB+DNN) or 3 (XGB+DNN+TEN)."
        )

    if ten_path.exists():
        print(
            f"  NOTE: {ten_path.name} exists, but {META_PATH.name} declares a "
            "2-model stacker. Using XGB+DNN only."
        )
    return False


def load_dnn(meta: dict) -> tuple[MLP, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MLP(meta["in_dim"], tuple(meta["hidden"]), meta["dropout"])
    model.load_state_dict(torch.load(DNN_PATH, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, device


def load_ten(meta: dict, device: torch.device) -> TENClassifier:
    model = TENClassifier(
        d_input  = meta["in_dim"],
        d_model  = meta["ten_d_model"],
        n_heads  = meta["ten_n_heads"],
        n_layers = meta["ten_n_layers"],
        d_ff     = meta["ten_d_ff"],
        dropout  = meta["ten_dropout"],
        seq_len  = meta["seq_len"],
    )
    model.load_state_dict(torch.load(TEN_PATH, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


def dnn_proba(model: MLP, device: torch.device, X: np.ndarray) -> float:
    with torch.no_grad():
        t      = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(t).cpu().numpy()
    return float(1.0 / (1.0 + np.exp(-logits[0])))


def ten_proba(model: TENClassifier, device: torch.device,
              X_seq: np.ndarray) -> float:
    """X_seq shape: (1, seq_len, n_features)."""
    with torch.no_grad():
        t      = torch.tensor(X_seq, dtype=torch.float32, device=device)
        logits = model(t).cpu().numpy()
    return float(1.0 / (1.0 + np.exp(-logits[0])))


def _build_sequence(df_scaled: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Return (1, seq_len, n_features) array for TEN, zero-padded at left."""
    n = len(df_scaled)
    n_feat = df_scaled.shape[1]
    seq = np.zeros((seq_len, n_feat), dtype=np.float32)
    rows = df_scaled.tail(seq_len).to_numpy(dtype=np.float32)
    seq[-len(rows):] = rows
    return seq[np.newaxis]  # (1, seq_len, n_feat)


def sentiment_nudge(df: pd.DataFrame) -> float:
    sent_col = "sentiment_mean"
    vol_col  = "news_volume"
    if sent_col not in df.columns:
        return 0.0
    s   = float(df[sent_col].iloc[-1])
    vol = float(df[vol_col].iloc[-1]) if vol_col in df.columns else 0.0
    vol_weight = min(vol / 10.0, 1.0)
    raw = s * vol_weight * SENTIMENT_CAP
    return float(np.clip(raw, -SENTIMENT_CAP, SENTIMENT_CAP))


def score_to_signal(p: float, gate: float) -> tuple[str, str]:
    if p >= gate:
        return "UP", "FUSION_CONFIDENT_UP"
    if p <= 1.0 - gate:
        return "DOWN", "FUSION_CONFIDENT_DOWN"
    return "NEUTRAL", "FUSION_LOW_CONFIDENCE"


def main() -> int:
    print("=" * 60)
    print("FUSION PREDICTOR")
    print("=" * 60)

    # ── Guard: required artifacts ─────────────────────────────────────────────
    required = [FEATURE_PATH, XGB_SIGNAL_PATH, DNN_PATH, STACKER_PATH, META_PATH,
                FEATURE_COLS_PATH, SCALER_PATH]
    missing = [p for p in required if not p.exists()]
    if missing:
        for p in missing:
            print(f"  MISSING: {p}")
        print("  Run the full pipeline first (market fetch → feature build → XGB predict).")
        return 1

    # ── Load meta + models ────────────────────────────────────────────────────
    with open(META_PATH) as f:
        meta = json.load(f)
    gate = meta.get("confidence_threshold", CONFIDENCE_THRESHOLD)

    device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stacker             = joblib.load(STACKER_PATH)
    try:
        use_ten = _should_use_ten(meta, stacker)
    except ValueError as exc:
        print(f"  ERROR: {exc}")
        return 1

    dnn_model, _device  = load_dnn(meta)
    ten_model           = load_ten(meta, device) if use_ten else None

    # ── Load features ─────────────────────────────────────────────────────────
    df = pd.read_csv(FEATURE_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"  Missing feature columns: {missing_cols[:5]} ...")
        return 1

    scaler    = joblib.load(SCALER_PATH)
    X_raw     = df[feature_cols].tail(1).to_numpy(dtype=np.float64)
    X_raw     = np.where(np.isfinite(X_raw), X_raw, 0.0)
    X_scaled  = scaler.transform(X_raw)

    latest_ts    = str(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else "unknown"
    latest_close = float(df["close"].iloc[-1]) if "close" in df.columns else None

    # ── XGBoost P(UP) ─────────────────────────────────────────────────────────
    with open(XGB_SIGNAL_PATH) as f:
        xgb_sig = json.load(f)

    if xgb_sig.get("status") != "ok":
        print(f"  XGBoost signal status={xgb_sig.get('status')} — falling back to neutral.")
        p_xgb = 0.5
    else:
        p_xgb = float(xgb_sig.get("probability_up", 0.5))

    # ── DNN P(UP) ─────────────────────────────────────────────────────────────
    p_dnn = dnn_proba(dnn_model, device, X_scaled)

    # ── TEN P(UP) — 32-candle sequence ───────────────────────────────────────
    if use_ten:
        seq_len       = meta.get("seq_len", 32)
        df_feat_scaled = pd.DataFrame(
            scaler.transform(
                np.where(np.isfinite(df[feature_cols].to_numpy(dtype=np.float64)),
                         df[feature_cols].to_numpy(dtype=np.float64), 0.0)
            ),
            columns=feature_cols,
        )
        X_seq = _build_sequence(df_feat_scaled, seq_len)
        p_ten = ten_proba(ten_model, device, X_seq)
    else:
        p_ten = None

    # ── Fusion stacker ────────────────────────────────────────────────────────
    if use_ten:
        S = np.array([[p_xgb, p_dnn, p_ten]])
    else:
        S = np.array([[p_xgb, p_dnn]])
    p_stack = float(stacker.predict_proba(S)[0, 1])

    # ── Sentiment nudge ───────────────────────────────────────────────────────
    nudge   = sentiment_nudge(df)
    p_final = float(np.clip(p_stack + nudge, 0.0, 1.0))

    signal, reason = score_to_signal(p_final, gate)

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"  Timestamp : {latest_ts}")
    print(f"  Close     : {latest_close}")
    print(f"  P(UP) XGB : {p_xgb:.4f}")
    print(f"  P(UP) DNN : {p_dnn:.4f}")
    if use_ten:
        print(f"  P(UP) TEN : {p_ten:.4f}")
    print(f"  P(UP) stk : {p_stack:.4f}")
    print(f"  Sent nudge: {nudge:+.4f}")
    print(f"  P(UP) fin : {p_final:.4f}")
    print(f"  Gate      : >={gate:.2f} UP | <={1-gate:.2f} DOWN")
    print(f"  SIGNAL    : {signal}  ({reason})")

    # ── Write JSON ────────────────────────────────────────────────────────────
    components: dict[str, Any] = {
        "xgb":             round(p_xgb, 6),
        "dnn":             round(p_dnn, 6),
        "stack":           round(p_stack, 6),
        "sentiment_nudge": round(nudge, 6),
    }
    if use_ten:
        components["ten"] = round(float(p_ten), 6)

    payload = {
        "generated_at_utc":    datetime.now(timezone.utc).isoformat(),
        "status":              "ok",
        "signal":              signal,
        "reason":              reason,
        "probability_up":      round(p_final, 6),
        "probability_down":    round(1.0 - p_final, 6),
        "components":          components,
        "latest_timestamp":    latest_ts,
        "latest_close":        latest_close,
        "horizon":             HORIZON,
        "confidence_threshold": gate,
        "models_used":         ["xgb", "dnn", "ten"] if use_ten else ["xgb", "dnn"],
    }
    safe = {k: _json_safe(v) for k, v in payload.items()}
    safe["components"] = {k: _json_safe(v) for k, v in payload["components"].items()}

    FUSION_SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FUSION_SIGNAL_PATH, "w") as f:
        json.dump(safe, f, indent=2)

    print(f"\n  Fusion signal → {FUSION_SIGNAL_PATH}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
