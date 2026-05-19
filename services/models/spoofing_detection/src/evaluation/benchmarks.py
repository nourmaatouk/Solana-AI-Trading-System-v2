"""
benchmarks.py — Spoofing detection benchmark suite
====================================================

Evaluates all models on the SAME Kaggle multi-asset test split used during
TEN-GNN training (BTC/ETH/SOL 1-min LOB, 34 features, window=100, stride=10).

Model lineup
------------
  Rule-Based          threshold heuristic on imbalance + cancel_ratio
  Logistic Regression mean-pooled LOB features (single-asset SOL)
  Random Forest       mean-pooled LOB features (single-asset SOL)
  LSTM-Attention      trained 20 epochs on Kaggle SOL windows
  TEN                 trained 20 epochs on Kaggle SOL windows
  TEN-GNN             loaded from trained checkpoint (multi-asset)

All F1 / precision / recall numbers are measured on the held-out test split
(last 15% of windows, chronological).  Latency is wall-clock per sequence.

Usage:
    cd services/models/spoofing_detection
    python -m src.evaluation.benchmarks
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_point_metrics
from src.models.ten import TENSpoofingDetector
from src.models.ten_gnn import TENGNNModel

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
_SPOOF_ROOT = Path(__file__).resolve().parents[2]   # .../spoofing_detection/
_CKPT = _SPOOF_ROOT / "checkpoints" / "tengnn_btc_eth_sol_1min" / "best_model.pt"

# Feature index inside the 34-dim LOB feature vector (n_levels=15)
#   0: midpoint  1: spread  2..31: bid/ask notional  32: imbalance  33: cancel_ratio
_FEAT_IMBALANCE   = 32
_FEAT_CANCEL_RATIO = 33
_SOL_ASSET_IDX    = 2   # matches --assets BTC ETH SOL


# ── LSTM-Attention baseline ────────────────────────────────────────────────────
class LSTMAttention(nn.Module):
    """2-layer BiLSTM + dot-product attention baseline."""

    def __init__(self, d_input: int = 34, hidden_size: int = 128,
                 num_layers: int = 2, n_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_input, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc   = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x, adj=None):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        return self.fc((w * out).sum(dim=1))


# ── Rule-based baseline ────────────────────────────────────────────────────────
def _rule_based_multi(
    X_test: np.ndarray,    # (W, N, T, D) multi-asset
    X_train: np.ndarray,   # (W, N, T, D) for threshold calibration
) -> np.ndarray:
    """Flag windows where mean |imbalance| OR mean cancel_ratio exceeds the
    training-set 75th percentile for ANY asset.

    Uses the max-across-assets statistic so the rule fires when BTC, ETH, OR SOL
    shows elevated activity — matching the union spoofing label.  Thresholds are
    derived from the training split to work correctly on StandardScaler-normalized
    features (population mean≈0, std≈1 per asset).
    """
    # (W, N, T, D) → mean over T → (W, N, D) → max over N → (W, D)
    mean_imb_train = np.abs(X_train[:, :, :, _FEAT_IMBALANCE]).mean(axis=2).max(axis=1)  # (W_tr,)
    mean_cr_train  = X_train[:, :, :, _FEAT_CANCEL_RATIO].mean(axis=2).max(axis=1)

    t_imb = float(np.percentile(mean_imb_train, 75))
    t_cr  = float(np.percentile(mean_cr_train,  75))

    mean_imb_test = np.abs(X_test[:, :, :, _FEAT_IMBALANCE]).mean(axis=2).max(axis=1)
    mean_cr_test  = X_test[:, :, :, _FEAT_CANCEL_RATIO].mean(axis=2).max(axis=1)

    return ((mean_imb_test > t_imb) | (mean_cr_test > t_cr)).astype(int)


def _rule_based(X_test: np.ndarray, X_train: np.ndarray | None = None) -> np.ndarray:
    """Single-asset rule-based detector (synthetic data path).

    Flags windows where mean |imbalance| AND mean cancel_ratio exceed the
    training-set 75th percentile.  Uses fixed fallback thresholds when training
    data is not provided.
    """
    mean_imbalance    = np.abs(X_test[:, :, _FEAT_IMBALANCE]).mean(axis=1)
    mean_cancel_ratio = X_test[:, :, _FEAT_CANCEL_RATIO].mean(axis=1)

    if X_train is not None:
        t_imb = float(np.percentile(np.abs(X_train[:, :, _FEAT_IMBALANCE]).mean(axis=1), 75))
        t_cr  = float(np.percentile(X_train[:, :, _FEAT_CANCEL_RATIO].mean(axis=1), 75))
    else:
        t_imb, t_cr = 0.3, 0.3

    return ((mean_imbalance > t_imb) & (mean_cancel_ratio > t_cr)).astype(int)


# ── Quick single-asset trainer ─────────────────────────────────────────────────
def _train_baseline(
    model: nn.Module,
    X_train: np.ndarray,   # (N, T, D)
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Train a single-asset model for `epochs` epochs with AdamW + class weights."""
    model = model.to(device)
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    weights = torch.tensor(
        y_train.size / (2 * counts.clip(min=1)), dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    best_f1, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_val, dtype=torch.float32, device=device))
            preds = preds.argmax(dim=-1).cpu().numpy()
        m = compute_point_metrics(y_val, preds)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device)


# ── Kaggle data loader (matches train_tengnn_full.py exactly) ──────────────────
def _load_kaggle_splits(
    assets: list[str] = ("BTC", "ETH", "SOL"),
    frequency: str = "1min",
    window_stride: int = 10,
    seed: int = 42,
) -> Optional[dict]:
    """Return chronological 70/15/15 splits of multi-asset Kaggle LOB data.

    Returns None when the raw CSV files are absent.

    Output keys
    -----------
    X_train / X_val / X_test : (W, n_assets, 100, 34)
    y_train / y_val / y_test : (W,) int64
    adj                      : (n_assets, n_assets) float32 Hawkes adjacency
    n_features               : int — feature dimension (34)
    """
    try:
        # benchmarks.py lives at src/evaluation/; add src/ to path
        # so that data.kaggle_loader and models.hawkes are importable.
        src_dir = Path(__file__).resolve().parents[1]   # .../src/
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from data.kaggle_loader import (
            load_kaggle_crypto_data,
            extract_lob_features,
            label_real_spoofing,
        )
        from models.hawkes import HawkesEstimator

    except ImportError as e:
        logger.warning("Kaggle benchmark skipped — import error: %s", e)
        return None

    per_asset_X: list[np.ndarray] = []
    per_asset_y: list[np.ndarray] = []
    per_asset_ts: dict[str, np.ndarray] = {}
    _HAWKES_MAX_EVENTS = 5000

    import pandas as pd

    # data/raw/kaggle_crypto/ is relative to the spoofing_detection root
    _spoof_root = Path(__file__).resolve().parents[2]
    kaggle_data_dir = _spoof_root / "data" / "raw" / "kaggle_crypto"

    for asset in assets:
        try:
            df = load_kaggle_crypto_data(kaggle_data_dir, asset, frequency)
        except FileNotFoundError:
            logger.warning("Kaggle benchmark skipped — %s data not found", asset)
            return None

        ts_col = "system_time" if "system_time" in df.columns else None
        if ts_col:
            ts_seconds = (
                pd.to_datetime(df[ts_col], errors="coerce").astype(np.int64) / 1e9
            ).values
        else:
            ts_seconds = np.arange(len(df), dtype=np.float64) * 60.0

        features, _ = extract_lob_features(df, n_levels=15)
        X, y = label_real_spoofing(df, features, window_length=100,
                                   window_stride=window_stride, n_levels=15)
        per_asset_X.append(X)
        per_asset_y.append(y)
        per_asset_ts[asset] = ts_seconds
        logger.info("  %s: %d windows  %.1f%% spoofing", asset, len(X), 100 * y.mean())

    # Align to shortest asset
    min_w = min(len(X) for X in per_asset_X)
    X_multi = np.stack([X[:min_w] for X in per_asset_X], axis=1).astype(np.float32)
    y_union  = np.stack([y[:min_w] for y in per_asset_y], axis=1).any(axis=1).astype(np.int64)

    # Hawkes adjacency
    global_t0 = min(ts[0] for ts in per_asset_ts.values())
    hawkes_ts = {
        a: (per_asset_ts[a] - global_t0)[:_HAWKES_MAX_EVENTS] / 60.0
        for a in assets
    }
    try:
        estimator = HawkesEstimator(n_assets=len(assets))
        # fit() expects Dict[int, List[float]] and returns None (updates in-place)
        event_dict = {i: hawkes_ts[a].tolist() for i, a in enumerate(assets)}
        estimator.fit(event_dict)
        adj = estimator.get_adjacency()
    except Exception:
        adj = np.eye(len(assets), dtype=np.float32)

    # Chronological 70/15/15 split
    W = len(y_union)
    n_train = int(W * 0.70)
    n_val   = int(W * 0.15)

    return {
        "X_train": X_multi[:n_train],
        "X_val":   X_multi[n_train : n_train + n_val],
        "X_test":  X_multi[n_train + n_val:],
        "y_train": y_union[:n_train],
        "y_val":   y_union[n_train : n_train + n_val],
        "y_test":  y_union[n_train + n_val:],
        "adj":     adj.astype(np.float32),
        "n_features": X_multi.shape[-1],
        "assets":  list(assets),
    }


# ── Main benchmark runner ──────────────────────────────────────────────────────
def run_benchmarks(
    output_json: Path,
    checkpoint_path: Path = _CKPT,
    n_samples_synth: int = 1000,
) -> None:
    """Run the full benchmark suite and save results to ``output_json``."""
    from src.data.dataset import prepare_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results: list[dict] = []

    # ── Section 1: Kaggle real data ────────────────────────────────────────────
    logger.info("Loading Kaggle LOB data …")
    splits = _load_kaggle_splits()

    if splits is not None:
        source = "Kaggle (BTC/ETH/SOL)"
        D      = splits["n_features"]   # 34
        N      = len(splits["assets"])  # 3

        X_test_multi = splits["X_test"]                    # (W, 3, 100, 34)
        y_test       = splits["y_test"]                    # (W,)
        X_test_sol   = X_test_multi[:, _SOL_ASSET_IDX]    # (W, 100, 34)

        X_train_sol  = splits["X_train"][:, _SOL_ASSET_IDX]
        X_val_sol    = splits["X_val"][:, _SOL_ASSET_IDX]
        y_train      = splits["y_train"]
        y_val        = splits["y_val"]

        adj_t = torch.tensor(splits["adj"], dtype=torch.float32, device=device)

        logger.info(
            "  Test: %d windows  spoofing=%.1f%%",
            len(y_test), 100 * y_test.mean(),
        )

        # Rule-based (multi-asset: fires on ANY asset exceeding threshold) ────────
        t0 = time.perf_counter()
        y_pred_rule = _rule_based_multi(X_test_multi, splits["X_train"])
        lat_rule = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
        m = compute_point_metrics(y_test, y_pred_rule)
        m.update({"Model": "Rule-Based", "Data Source": source,
                  "trained": True, "latency_ms_per_seq": round(lat_rule, 5),
                  "note": "max-across-assets mean |imbalance| OR cancel_ratio > 75th-pctile"})
        all_results.append(m)

        # Logistic Regression ───────────────────────────────────────────────────
        X_tr_flat  = X_train_sol.mean(axis=1)   # (W, 34)
        X_val_flat = X_val_sol.mean(axis=1)
        X_te_flat  = X_test_sol.mean(axis=1)
        # Combine train+val for classical models
        X_clf_train = np.concatenate([X_tr_flat, X_val_flat], axis=0)
        y_clf_train = np.concatenate([y_train, y_val], axis=0)

        lr = LogisticRegression(max_iter=3000, class_weight="balanced")
        lr.fit(X_clf_train, y_clf_train)
        t0 = time.perf_counter()
        y_pred_lr = lr.predict(X_te_flat)
        lat_lr = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
        m = compute_point_metrics(y_test, y_pred_lr)
        m.update({"Model": "Logistic Regression", "Data Source": source,
                  "trained": True, "latency_ms_per_seq": round(lat_lr, 5)})
        all_results.append(m)

        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    random_state=42)
        rf.fit(X_clf_train, y_clf_train)
        t0 = time.perf_counter()
        y_pred_rf = rf.predict(X_te_flat)
        lat_rf = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
        m = compute_point_metrics(y_test, y_pred_rf)
        m.update({"Model": "Random Forest", "Data Source": source,
                  "trained": True, "latency_ms_per_seq": round(lat_rf, 5)})
        all_results.append(m)

        # LSTM-Attention (trained 20 epochs) ────────────────────────────────────
        logger.info("  Training LSTM-Attention baseline (20 epochs) …")
        lstm = _train_baseline(
            LSTMAttention(d_input=D), X_train_sol, y_train,
            X_val_sol, y_val, epochs=20, device=device,
        )
        lstm.eval()
        X_te_t = torch.tensor(X_test_sol, dtype=torch.float32, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            y_pred_lstm = lstm(X_te_t).argmax(dim=-1).cpu().numpy()
        lat_lstm = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
        m = compute_point_metrics(y_test, y_pred_lstm)
        m.update({"Model": "LSTM-Attention", "Data Source": source,
                  "trained": True, "latency_ms_per_seq": round(lat_lstm, 5),
                  "note": "single-asset SOL, 20 epochs"})
        all_results.append(m)

        # TEN (trained 20 epochs, compact config to match data size) ─────────────
        logger.info("  Training TEN baseline (20 epochs) …")
        ten = _train_baseline(
            TENSpoofingDetector(d_input=D, d_model=64, n_heads=4, d_ff=256, n_layers=2),
            X_train_sol, y_train, X_val_sol, y_val, epochs=20, device=device,
        )
        ten.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            y_pred_ten = ten(X_te_t).argmax(dim=-1).cpu().numpy()
        lat_ten = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
        m = compute_point_metrics(y_test, y_pred_ten)
        m.update({"Model": "TEN", "Data Source": source,
                  "trained": True, "latency_ms_per_seq": round(lat_ten, 5),
                  "note": "single-asset SOL, 20 epochs"})
        all_results.append(m)

        # TEN-GNN from trained checkpoint ────────────────────────────────────────
        if checkpoint_path.exists():
            logger.info("  Evaluating TEN-GNN from checkpoint %s …", checkpoint_path.name)
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            cfg  = ckpt.get("config", {})
            n_h  = cfg.get("n_heads", min(8, cfg.get("d_model", 128) // 32))
            tengnn = TENGNNModel(
                n_assets=cfg.get("n_assets", N),
                d_input=cfg.get("d_input", D),
                d_model=cfg.get("d_model", 128),
                n_heads=n_h,
                n_layers=cfg.get("n_layers", 3),
                d_ff=cfg.get("d_ff", 512),
            ).to(device)
            tengnn.load_state_dict(ckpt["model_state_dict"])
            tengnn.eval()

            best_threshold = float(ckpt.get("best_threshold", 0.5))
            ckpt_epoch     = int(ckpt.get("epoch", "?") or 0)
            ckpt_val_f1    = float(ckpt.get("val_f1", 0.0))

            X_te_gnn = torch.tensor(X_test_multi, dtype=torch.float32, device=device)
            # Collect probabilities in batches
            probs_list = []
            t0 = time.perf_counter()
            with torch.no_grad():
                for i in range(0, len(X_te_gnn), 64):
                    logits = tengnn(X_te_gnn[i:i+64], adj_t)
                    probs_list.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            lat_gnn = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
            probs_gnn = np.concatenate(probs_list)
            y_pred_gnn = (probs_gnn >= best_threshold).astype(int)

            m = compute_point_metrics(y_test, y_pred_gnn)
            m.update({
                "Model":               "TEN-GNN",
                "Data Source":         source,
                "trained":             True,
                "latency_ms_per_seq":  round(lat_gnn, 5),
                "note": (
                    f"checkpoint epoch={ckpt_epoch}  val_f1={ckpt_val_f1:.3f}"
                    f"  threshold={best_threshold:.2f}"
                    f"  multi-asset BTC/ETH/SOL"
                ),
            })
            all_results.append(m)
        else:
            logger.warning("TEN-GNN checkpoint not found at %s — skipped", checkpoint_path)

    else:
        logger.warning("Kaggle data unavailable — skipping real-data benchmarks")

    # ── Section 2: Synthetic data (reference / regression baseline) ────────────
    logger.info("Running synthetic benchmarks …")
    data_synth = prepare_data(use_real_data=False, n_sequences=n_samples_synth)
    _eval_synthetic(data_synth, all_results, device)

    # ── Save results ────────────────────────────────────────────────────────────
    output_json.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if output_json.exists():
        with open(output_json) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                pass

    existing["benchmarks_standardized"] = all_results
    existing["benchmark_metadata"] = {
        "generated_utc":   __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_used": checkpoint_path.exists(),
        "kaggle_data_used": splits is not None,
    }

    with open(output_json, "w") as f:
        json.dump(existing, f, indent=4)

    # ── Print table ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 108)
    print("BENCHMARK RESULTS")
    print("=" * 108)
    print(f"{'Model':<22} | {'Source':<24} | {'F1':>6} | {'Prec.':>6} | {'Recall':>6} | {'Latency ms':>11} | Trained")
    print("-" * 108)
    for r in all_results:
        trained_flag = "yes" if r.get("trained") else "NO (random)"
        print(
            f"{r['Model']:<22} | {r['Data Source']:<24} | {r['f1']:>6.3f} | "
            f"{r['precision']:>6.3f} | {r['recall']:>6.3f} | "
            f"{r['latency_ms_per_seq']:>11.4f} | {trained_flag}"
        )
    print("=" * 108)
    logger.info("Saved results → %s", output_json)


# ── Synthetic section (kept for regression / smoke testing) ───────────────────
def _eval_synthetic(datasets: dict, all_results: list, device: torch.device) -> None:
    source = "Synthetic"
    test_ds = datasets["test"]
    X_test  = test_ds.X.numpy()        # (N, T, D)
    y_test  = test_ds.y.numpy()
    D       = X_test.shape[2]

    # Rule-based uses LOBSTER indices 44 (imbalance) and 43 (delta_t)
    t0 = time.perf_counter()
    if D >= 45:
        peak_imb   = np.abs(X_test[:, :, 44]).max(axis=1)
        min_delta  = X_test[:, :, 43].min(axis=1)
        y_pred_rule = ((peak_imb > 0.7) & (min_delta < 10.0)).astype(int)
    else:
        y_pred_rule = np.zeros(len(y_test), dtype=int)
    lat_rule = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
    m = compute_point_metrics(y_test, y_pred_rule)
    m.update({"Model": "Rule-Based", "Data Source": source, "trained": True,
              "latency_ms_per_seq": round(lat_rule, 5),
              "note": "LOBSTER features 44+43"})
    all_results.append(m)

    X_tr_flat = datasets["train"].X.mean(dim=1).numpy()
    y_tr      = datasets["train"].y.numpy()
    X_te_flat = X_test.mean(axis=1)

    for name, clf in [("Logistic Regression", LogisticRegression(max_iter=3000, class_weight="balanced")),
                      ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))]:
        clf.fit(X_tr_flat, y_tr)
        t0 = time.perf_counter()
        y_pred = clf.predict(X_te_flat)
        lat = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
        m = compute_point_metrics(y_test, y_pred)
        m.update({"Model": name, "Data Source": source, "trained": True,
                  "latency_ms_per_seq": round(lat, 5)})
        all_results.append(m)

    # Deep models trained on synthetic data (20 epochs each)
    X_train_syn = datasets["train"].X.numpy()
    y_train_syn = datasets["train"].y.numpy()
    X_val_syn   = datasets["val"].X.numpy()
    y_val_syn   = datasets["val"].y.numpy()
    X_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    logger.info("  Training LSTM-Attention on synthetic data (20 epochs) …")
    lstm_syn = _train_baseline(
        LSTMAttention(d_input=D), X_train_syn, y_train_syn,
        X_val_syn, y_val_syn, epochs=20, device=device,
    )
    lstm_syn.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        y_pred_lstm = lstm_syn(X_t).argmax(dim=-1).cpu().numpy()
    lat = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
    m = compute_point_metrics(y_test, y_pred_lstm)
    m.update({"Model": "LSTM-Attention", "Data Source": source, "trained": True,
              "latency_ms_per_seq": round(lat, 5), "note": "synthetic data, 20 epochs"})
    all_results.append(m)

    logger.info("  Training TEN on synthetic data (20 epochs) …")
    ten_syn = _train_baseline(
        TENSpoofingDetector(d_input=D, d_model=64, n_heads=4, d_ff=256, n_layers=2),
        X_train_syn, y_train_syn, X_val_syn, y_val_syn, epochs=20, device=device,
    )
    ten_syn.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        y_pred_ten = ten_syn(X_t).argmax(dim=-1).cpu().numpy()
    lat = (time.perf_counter() - t0) * 1000 / max(len(y_test), 1)
    m = compute_point_metrics(y_test, y_pred_ten)
    m.update({"Model": "TEN", "Data Source": source, "trained": True,
              "latency_ms_per_seq": round(lat, 5), "note": "synthetic data, 20 epochs"})
    all_results.append(m)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Add src/ to path when running as __main__ (not as a module)
    _src_dir = Path(__file__).resolve().parents[1]
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    out_path = _SPOOF_ROOT / "results" / "evaluation_report.json"
    run_benchmarks(out_path)
