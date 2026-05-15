"""
train_dnn_fusion.py — Layer 3: DNN + Fusion Stacker

Trains two complementary models on top of the existing XGBoost pipeline:

  1. MLP (DNN) — a 4-layer PyTorch neural network trained on the same 47 scaled
     features as XGBoost.  Adds a complementary inductive bias (continuous
     gradient vs. tree splits) and independently calibrated P(UP).

  2. Fusion stacker — a logistic regression trained on the validation-set
     out-of-sample probabilities from [XGBoost, DNN].  The stacker learns the
     optimal weighting of the two models on data neither model saw during fit.

Artifacts written:
  MODEL_DIR / dnn_h{HORIZON}.pt          — PyTorch state dict
  MODEL_DIR / fusion_stacker_h{HORIZON}.pkl — sklearn LogisticRegression
  MODEL_DIR / fusion_meta_h{HORIZON}.json  — feature count, threshold, metrics
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from config import ACTIVE_HORIZON as HORIZON, CONFIDENCE_THRESHOLD, MODEL_DIR, PROCESSED_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLIT = {
    split: {
        "X": PROCESSED_DIR / f"X_{split}_h{HORIZON}.npy",
        "y": PROCESSED_DIR / f"y_{split}_h{HORIZON}.npy",
    }
    for split in ("train", "val", "test")
}
XGB_MODEL_PATH  = MODEL_DIR / f"xgboost_h{HORIZON}.json"
XGB_SCALER_PATH = PROCESSED_DIR / f"scaler_h{HORIZON}.pkl"
DNN_PATH        = MODEL_DIR / f"dnn_h{HORIZON}.pt"
STACKER_PATH    = MODEL_DIR / f"fusion_stacker_h{HORIZON}.pkl"
META_PATH       = MODEL_DIR / f"fusion_meta_h{HORIZON}.json"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
HIDDEN       = (256, 128, 64)
DROPOUT      = 0.2
LR           = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 512
MAX_EPOCHS   = 150
PATIENCE     = 12            # early stopping on val loss
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = HIDDEN, dropout: float = DROPOUT):
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            logits = self(t).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))   # sigmoid


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    in_dim: int,
) -> MLP:
    model = MLP(in_dim).to(DEVICE)

    n_neg   = int((y_train == 0).sum())
    n_pos   = int((y_train == 1).sum())
    pos_w   = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_v  = torch.tensor(X_val,   dtype=torch.float32, device=DEVICE)
    y_v  = torch.tensor(y_val,   dtype=torch.float32, device=DEVICE)

    dataset   = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader    = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_v), y_v).item()
        sched.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  val_loss={val_loss:.4f}  patience={patience_cnt}/{PATIENCE}")

        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost OOS probabilities
# ─────────────────────────────────────────────────────────────────────────────
def xgb_oos_proba(X: np.ndarray) -> np.ndarray:
    import xgboost as xgb
    scaler = joblib.load(XGB_SCALER_PATH)
    clf    = xgb.XGBClassifier()
    clf.load_model(str(XGB_MODEL_PATH))
    X_s    = scaler.transform(X)
    return clf.predict_proba(X_s)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────
def eval_metrics(name: str, proba: np.ndarray, y: np.ndarray, gate: float = CONFIDENCE_THRESHOLD) -> dict:
    lo = 1.0 - gate
    mask  = (proba >= gate) | (proba <= lo)
    n_sig = int(mask.sum())
    acc   = float(accuracy_score(y[mask], (proba[mask] >= gate).astype(int))) if n_sig > 0 else 0.0
    auc   = float(roc_auc_score(y, proba))
    print(f"  [{name}]  AUC={auc:.4f}  Gated-acc={acc:.4f}  signals={n_sig}/{len(y)}")
    return {"name": name, "auc": round(auc, 4), "gated_acc": round(acc, 4), "n_signals": n_sig, "n_total": len(y)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print(f"DNN + FUSION STACKER TRAINING  (H{HORIZON}, device={DEVICE})")
    print("=" * 70)

    # ── Load splits ───────────────────────────────────────────────────────────
    X_train = np.load(SPLIT["train"]["X"])
    y_train = np.load(SPLIT["train"]["y"]).astype(np.float32)
    X_val   = np.load(SPLIT["val"]["X"])
    y_val   = np.load(SPLIT["val"]["y"]).astype(np.float32)
    X_test  = np.load(SPLIT["test"]["X"])
    y_test  = np.load(SPLIT["test"]["y"]).astype(np.float32)

    in_dim = X_train.shape[1]
    print(f"\nSplit sizes — train:{len(X_train):,}  val:{len(X_val):,}  test:{len(X_test):,}")
    print(f"Features: {in_dim}   UP ratio: {y_train.mean():.3f}")

    # ── Train DNN ─────────────────────────────────────────────────────────────
    print(f"\n── Training MLP (DNN) ──────────────────────────────────────────")
    t0  = time.perf_counter()
    dnn = train_dnn(X_train, y_train, X_val, y_val, in_dim)
    print(f"  DNN trained in {time.perf_counter()-t0:.1f}s")

    # ── OOS probabilities for stacker training (val set) ─────────────────────
    print(f"\n── Computing OOS probabilities on val set ──────────────────────")
    p_xgb_val = xgb_oos_proba(X_val)
    p_dnn_val = dnn.predict_proba(X_val)

    # ── Train fusion stacker on val OOS proba ─────────────────────────────────
    print(f"\n── Training logistic stacker on [P_xgb, P_dnn] ─────────────────")
    S_val    = np.column_stack([p_xgb_val, p_dnn_val])
    stacker  = LogisticRegression(max_iter=1000, C=1.0)
    stacker.fit(S_val, y_val.astype(int))
    coefs = stacker.coef_[0]
    print(f"  Stacker weights — XGB: {coefs[0]:.3f}  DNN: {coefs[1]:.3f}")

    # ── Evaluate all three on test set ────────────────────────────────────────
    print(f"\n── Test-set evaluation ─────────────────────────────────────────")
    p_xgb_test    = xgb_oos_proba(X_test)
    p_dnn_test    = dnn.predict_proba(X_test)
    S_test        = np.column_stack([p_xgb_test, p_dnn_test])
    p_fusion_test = stacker.predict_proba(S_test)[:, 1]

    results = [
        eval_metrics("XGBoost",     p_xgb_test,    y_test),
        eval_metrics("DNN",         p_dnn_test,     y_test),
        eval_metrics("Fusion",      p_fusion_test,  y_test),
    ]

    # ── Save artifacts ────────────────────────────────────────────────────────
    print(f"\n── Saving artifacts ─────────────────────────────────────────────")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(dnn.state_dict(), DNN_PATH)
    print(f"  DNN        → {DNN_PATH}")

    joblib.dump(stacker, STACKER_PATH)
    print(f"  Stacker    → {STACKER_PATH}")

    meta = {
        "horizon":              HORIZON,
        "in_dim":               in_dim,
        "hidden":               list(HIDDEN),
        "dropout":              DROPOUT,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "stacker_weights":      {"xgb": float(coefs[0]), "dnn": float(coefs[1])},
        "test_metrics":         results,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta       → {META_PATH}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
