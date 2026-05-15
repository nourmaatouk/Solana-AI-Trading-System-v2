"""
train_ten_fusion.py — Layer 3: TEN + 3-model Fusion Stacker

Trains a Temporal Encoder Network (TEN) on sliding windows of 15-minute
candles, then retrains the fusion stacker with a third model in the ensemble:

  Fusion input:  [P_xgb, P_dnn, P_ten]   (was [P_xgb, P_dnn])

Architecture (classification variant, lighter than spoofing TEN):
  Input:  (batch, SEQ_LEN=32, d_input=47)
    ├─ Linear 47 → d_model=128
    ├─ Learnable positional embedding  (1, 32, 128)
    ├─ 3× TransformerEncoderLayer  (4 heads, d_ff=512, dropout=0.1)
    ├─ Mean pooling  → (batch, 128)
    └─ Head: Linear(128→32) → GELU → Linear(32→1)

Sequence building:
  For each labeled candle at original index k, the input window is
  candle rows [k-31, …, k] of the FULL (unfiltered) ORDERBOOK_CSV,
  scaled with the same scaler_h{H}.pkl used by XGBoost and DNN.
  Near-start rows (k < 31) are zero-padded on the left.

Artifacts updated:
  MODEL_DIR / ten_h{H}.pt                  — TEN state dict
  MODEL_DIR / fusion_stacker_h{H}.pkl      — retrained 3-input stacker
  MODEL_DIR / fusion_meta_h{H}.json        — updated weights + metrics
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from config import (
    ACTIVE_HORIZON as HORIZON,
    CONFIDENCE_THRESHOLD,
    LEAKAGE_SUBSTRINGS,
    MODEL_DIR,
    NON_STATIONARY_COLUMNS,
    ORDERBOOK_CSV,
    PROCESSED_DIR,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRAIN_RATIO,
    VAL_RATIO,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
XGB_MODEL_PATH  = MODEL_DIR / f"xgboost_h{HORIZON}.json"
XGB_SCALER_PATH = PROCESSED_DIR / f"scaler_h{HORIZON}.pkl"
DNN_PATH        = MODEL_DIR / f"dnn_h{HORIZON}.pt"
TEN_PATH        = MODEL_DIR / f"ten_h{HORIZON}.pt"
STACKER_PATH    = MODEL_DIR / f"fusion_stacker_h{HORIZON}.pkl"
META_PATH       = MODEL_DIR / f"fusion_meta_h{HORIZON}.json"
FEAT_COLS_PATH  = PROCESSED_DIR / "feature_columns.json"

# ── TEN hyper-parameters ─────────────────────────────────────────────────────
SEQ_LEN   = 32        # 32 × 15 min = 8 h of temporal context
D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 3
D_FF      = 512
DROPOUT   = 0.1

LR           = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 256
MAX_EPOCHS   = 100
PATIENCE     = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# TEN model
# ─────────────────────────────────────────────────────────────────────────────
class TENClassifier(nn.Module):
    """Lightweight Temporal Encoder Network for candle-sequence classification."""

    def __init__(
        self,
        d_input:  int = 47,
        d_model:  int = D_MODEL,
        n_heads:  int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff:     int = D_FF,
        dropout:  float = DROPOUT,
        seq_len:  int = SEQ_LEN,
    ):
        super().__init__()
        self.embed   = nn.Linear(d_input, d_model)
        # Learnable position embedding — one vector per timestep
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_input)
        h = self.embed(x) + self.pos_emb[:, :x.size(1)]
        h = self.encoder(h)
        h = h.mean(dim=1)          # mean pool across time → (B, d_model)
        return self.head(h).squeeze(-1)

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        self.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_seq), BATCH_SIZE):
                batch = torch.tensor(X_seq[i:i+BATCH_SIZE], dtype=torch.float32, device=DEVICE)
                logits = self(batch).cpu().numpy()
                probs.append(1.0 / (1.0 + np.exp(-logits)))
        return np.concatenate(probs)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builder
# ─────────────────────────────────────────────────────────────────────────────
def build_sequences(seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_seq, y, orig_idx_arr) for all labeled candles.

    X_seq shape: (N_labeled, seq_len, n_features)
    Each window uses [k-seq_len+1, …, k] from the full scaled feature array.
    Rows before the start of the data are zero-padded.
    """
    print("  Loading ORDERBOOK_CSV …")
    df_full = pd.read_csv(PROCESSED_DIR / ORDERBOOK_CSV)
    n_full  = len(df_full)

    feat_cols = json.load(open(FEAT_COLS_PATH))

    # Scale the full array with the training scaler (fit was on training rows only)
    scaler   = joblib.load(XGB_SCALER_PATH)
    X_raw    = df_full[feat_cols].to_numpy(dtype=np.float64)
    X_raw    = np.where(np.isfinite(X_raw), X_raw, 0.0)
    X_scaled = scaler.transform(X_raw).astype(np.float32)   # (n_full, n_features)

    # Triple-barrier labeling (identical to training scripts)
    future_max_high = df_full["high"].rolling(HORIZON).max().shift(-HORIZON)
    future_min_low  = df_full["low"].rolling(HORIZON).min().shift(-HORIZON)
    tp_price = df_full["close"] * (1 + TAKE_PROFIT_PCT)
    sl_price = df_full["close"] * (1 - STOP_LOSS_PCT)
    hit_tp   = future_max_high >= tp_price
    hit_sl   = future_min_low  <= sl_price

    df_full["target"]    = float("nan")
    df_full["_orig_idx"] = np.arange(n_full)
    df_full.loc[hit_tp & ~hit_sl, "target"] = 1
    df_full.loc[hit_sl & ~hit_tp, "target"] = 0
    labeled = df_full.dropna(subset=["target"]).reset_index(drop=True)

    orig_indices = labeled["_orig_idx"].values.astype(int)
    y            = labeled["target"].values.astype(np.float32)
    n_feat       = X_scaled.shape[1]

    print(f"  Building {len(orig_indices):,} sequences of length {seq_len} × {n_feat} features …")
    X_seq = np.zeros((len(orig_indices), seq_len, n_feat), dtype=np.float32)

    for i, k in enumerate(orig_indices):
        start  = max(0, k - seq_len + 1)
        window = X_scaled[start : k + 1]      # shape ≤ (seq_len, n_feat)
        pad    = seq_len - len(window)
        if pad > 0:
            window = np.concatenate([np.zeros((pad, n_feat), dtype=np.float32), window])
        X_seq[i] = window

    return X_seq, y, orig_indices


def chronological_split(
    X_seq: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n         = len(X_seq)
    train_end = int(n * TRAIN_RATIO)
    val_end   = train_end + int(n * VAL_RATIO)
    return (
        X_seq[:train_end], y[:train_end],
        X_seq[train_end:val_end], y[train_end:val_end],
        X_seq[val_end:], y[val_end:],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_ten(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    d_input: int,
) -> TENClassifier:
    model = TENClassifier(d_input=d_input).to(DEVICE)

    n_neg  = int((y_train == 0).sum())
    n_pos  = int((y_train == 1).sum())
    pos_w  = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Cosine LR decay with warm-up (first 5 epochs)
    warmup_steps  = 5
    total_steps   = MAX_EPOCHS
    def lr_lambda(ep: int) -> float:
        if ep < warmup_steps:
            return (ep + 1) / warmup_steps
        progress = (ep - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_v  = torch.tensor(X_val,   dtype=torch.float32, device=DEVICE)
    y_v  = torch.tensor(y_val,   dtype=torch.float32, device=DEVICE)

    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_val  = float("inf")
    patience  = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_v), y_v).item()

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            patience   = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  val_loss={val_loss:.4f}  patience={patience}/{PATIENCE}")

        if patience >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost / DNN helpers
# ─────────────────────────────────────────────────────────────────────────────
def xgb_oos_proba(X_flat: np.ndarray) -> np.ndarray:
    import xgboost as xgb
    scaler = joblib.load(XGB_SCALER_PATH)
    clf    = xgb.XGBClassifier()
    clf.load_model(str(XGB_MODEL_PATH))
    return clf.predict_proba(scaler.transform(X_flat))[:, 1]


def load_dnn(meta: dict) -> tuple:
    from train_dnn_fusion import MLP
    m      = MLP(meta["in_dim"], tuple(meta["hidden"]), meta["dropout"]).to(DEVICE)
    m.load_state_dict(torch.load(DNN_PATH, map_location=DEVICE, weights_only=True))
    return m.eval()


def dnn_oos_proba(dnn, X_flat: np.ndarray) -> np.ndarray:
    return dnn.predict_proba(X_flat)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def eval_metrics(name: str, proba: np.ndarray, y: np.ndarray) -> dict:
    gate = CONFIDENCE_THRESHOLD
    lo   = 1.0 - gate
    mask = (proba >= gate) | (proba <= lo)
    n    = int(mask.sum())
    acc  = float(accuracy_score(y[mask], (proba[mask] >= gate).astype(int))) if n > 0 else 0.0
    auc  = float(roc_auc_score(y, proba))
    print(f"  [{name:12s}]  AUC={auc:.4f}  Gated-acc={acc:.4f}  signals={n}/{len(y)}")
    return {"name": name, "auc": round(auc, 4), "gated_acc": round(acc, 4),
            "n_signals": n, "n_total": len(y)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print(f"TEN + 3-MODEL FUSION STACKER  (H{HORIZON}, device={DEVICE})")
    print("=" * 70)

    # ── Build sequences ───────────────────────────────────────────────────────
    print("\n── Building candle-sequence dataset ────────────────────────────")
    t0    = time.perf_counter()
    X_seq, y_all, _ = build_sequences()
    print(f"  Done in {time.perf_counter()-t0:.1f}s  |  shape: {X_seq.shape}")

    X_tr_s, y_tr, X_val_s, y_val, X_te_s, y_te = chronological_split(X_seq, y_all)
    print(f"  train:{len(X_tr_s):,}  val:{len(X_val_s):,}  test:{len(X_te_s):,}")
    d_input = X_seq.shape[2]

    # ── Load flat splits for XGB + DNN (same chronological order) ────────────
    X_tr_f = np.load(PROCESSED_DIR / f"X_train_h{HORIZON}.npy")
    X_val_f = np.load(PROCESSED_DIR / f"X_val_h{HORIZON}.npy")
    X_te_f  = np.load(PROCESSED_DIR / f"X_test_h{HORIZON}.npy")

    # ── Train TEN ─────────────────────────────────────────────────────────────
    print(f"\n── Training TEN ────────────────────────────────────────────────")
    t0  = time.perf_counter()
    ten = train_ten(X_tr_s, y_tr, X_val_s, y_val, d_input)
    print(f"  TEN trained in {time.perf_counter()-t0:.1f}s")

    # ── OOS probabilities on val set (for stacker training) ───────────────────
    print(f"\n── Computing OOS probabilities on val set ───────────────────────")
    meta  = json.load(open(META_PATH))
    dnn   = load_dnn(meta)
    p_xgb_val = xgb_oos_proba(X_val_f)
    p_dnn_val = dnn_oos_proba(dnn, X_val_f)
    p_ten_val = ten.predict_proba(X_val_s)

    # ── Retrain stacker with 3 inputs ─────────────────────────────────────────
    print(f"\n── Retraining stacker on [P_xgb, P_dnn, P_ten] ────────────────")
    S_val   = np.column_stack([p_xgb_val, p_dnn_val, p_ten_val])
    stacker = LogisticRegression(max_iter=1000, C=1.0)
    stacker.fit(S_val, y_val.astype(int))
    coefs   = stacker.coef_[0]
    print(f"  Weights — XGB: {coefs[0]:.3f}  DNN: {coefs[1]:.3f}  TEN: {coefs[2]:.3f}")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print(f"\n── Test-set evaluation ─────────────────────────────────────────")
    p_xgb_te  = xgb_oos_proba(X_te_f)
    p_dnn_te  = dnn_oos_proba(dnn, X_te_f)
    p_ten_te  = ten.predict_proba(X_te_s)
    S_te      = np.column_stack([p_xgb_te, p_dnn_te, p_ten_te])
    p_fus_te  = stacker.predict_proba(S_te)[:, 1]

    results = [
        eval_metrics("XGBoost",  p_xgb_te, y_te),
        eval_metrics("DNN",      p_dnn_te,  y_te),
        eval_metrics("TEN",      p_ten_te,  y_te),
        eval_metrics("Fusion-3", p_fus_te,  y_te),
    ]

    # ── Save artifacts ────────────────────────────────────────────────────────
    print(f"\n── Saving artifacts ─────────────────────────────────────────────")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(ten.state_dict(), TEN_PATH)
    print(f"  TEN        → {TEN_PATH}")

    joblib.dump(stacker, STACKER_PATH)
    print(f"  Stacker    → {STACKER_PATH}")

    meta.update({
        "models":           ["xgb", "dnn", "ten"],
        "seq_len":          SEQ_LEN,
        "ten_d_model":      D_MODEL,
        "ten_n_heads":      N_HEADS,
        "ten_n_layers":     N_LAYERS,
        "ten_d_ff":         D_FF,
        "ten_dropout":      DROPOUT,
        "stacker_weights":  {"xgb": float(coefs[0]), "dnn": float(coefs[1]), "ten": float(coefs[2])},
        "test_metrics":     results,
    })
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta       → {META_PATH}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
