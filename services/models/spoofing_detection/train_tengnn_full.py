#!/usr/bin/env python3
"""
Train the full TEN-GNN model on Kaggle crypto data using multiple assets.

Unlike train_kaggle_full.py (which trains TEN on a single asset), this script:
  1. Loads N assets simultaneously (e.g. BTC + ETH + ADA)
  2. Runs the Hawkes Process estimator to build a cross-asset adjacency matrix
  3. Trains the TENGNNModel end-to-end with the real multi-asset input format

Usage:
    cd services/models/spoofing_detection
    python train_tengnn_full.py --assets BTC ETH ADA --frequency 1min --epochs 50

    # Quick smoke run (5 epochs, 5000 rows per asset):
    python train_tengnn_full.py --assets BTC ETH --frequency 1min \\
        --max-rows 5000 --epochs 5
"""

import sys
sys.path.insert(0, "src")

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.kaggle_loader import (
    load_kaggle_crypto_data,
    extract_lob_features,
    label_real_spoofing,
)
from data.dataset import create_data_splits, create_dataloaders, SpoofingDataset
from models.ten_gnn import TENGNNModel
from models.hawkes import HawkesEstimator
from training.trainer import TrainingConfig
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_tengnn")


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TEN-GNN on multi-asset Kaggle data")
    p.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL"],
                   choices=["BTC", "ETH", "ADA", "SOL"],
                   help="Assets to include in the graph (min 2)")
    p.add_argument("--frequency", default="1min",
                   choices=["1sec", "1min", "5min"])
    p.add_argument("--max-rows", type=int, default=None,
                   help="Max rows per asset (None = all)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-stride", type=int, default=10,
                   help="Sliding window stride (default 10 → ~10× more windows than stride=100)")
    p.add_argument("--d-model", type=int, default=128,
                   help="Transformer d_model (default 128 — smaller fits small datasets)")
    p.add_argument("--n-layers", type=int, default=3,
                   help="Number of transformer layers (default 3)")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate (default 0.1)")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Override checkpoint directory")
    return p.parse_args()


# ── Metrics helper ────────────────────────────────────────────────────────────
def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "f1":        float(f1_score(labels, preds, zero_division=0.0)),
        "precision": float(precision_score(labels, preds, zero_division=0.0)),
        "recall":    float(recall_score(labels, preds, zero_division=0.0)),
    }


# ── Hawkes adjacency builder ──────────────────────────────────────────────────
def build_hawkes_adjacency(
    asset_timestamps: dict[str, np.ndarray],
    n_assets: int,
) -> np.ndarray:
    """Run Hawkes MLE on per-asset event timestamps and return the adjacency matrix."""
    logger.info("Fitting Hawkes Process (%d assets)...", n_assets)
    estimator = HawkesEstimator(n_assets=n_assets)
    event_dict = {i: ts.tolist() for i, ts in enumerate(asset_timestamps.values())}
    estimator.fit(event_dict)
    adj = estimator.get_adjacency()
    logger.info("Hawkes adjacency:\n%s", np.round(adj, 3))
    return adj


# ── Multi-asset dataset builder ───────────────────────────────────────────────
# Max rows fed into Hawkes MLE (O(n²) complexity — keep well under 3 000).
_HAWKES_MAX_EVENTS = 2_000

def build_multi_asset_dataset(
    asset_names: list[str],
    frequency: str,
    max_rows: int | None,
    seed: int,
    window_stride: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load each asset, align windows by minimum count, stack into (B, N, T, D).

    Timestamps fed to the Hawkes estimator are real wall-clock minutes derived
    from the ``system_time`` column of the Kaggle CSV.  Different assets are
    stamped at slightly different sub-minute offsets (e.g. BTC at :41 s, ETH at
    :49 s), giving Hawkes the temporal separation it needs to detect lead-lag
    relationships.  All timestamps are normalised relative to the earliest
    observation across all assets so the built-in β parameters (10 and 0.5)
    represent fast/slow excitation windows of ~0.07 min and ~1.4 min.

    Returns
    -------
    X_multi : (n_windows, n_assets, 100, n_features)
    y_union : (n_windows,)  — 1 if ANY asset window is spoofing
    per_asset_ts : asset → 1-D array of minutes-since-start (for Hawkes)
    """
    per_asset_X: list[np.ndarray] = []
    per_asset_y: list[np.ndarray] = []
    per_asset_ts_ms: dict[str, np.ndarray] = {}

    for asset in asset_names:
        logger.info("Loading %s %s...", asset, frequency)
        df = load_kaggle_crypto_data(asset=asset, frequency=frequency, max_rows=max_rows)

        # ── Real wall-clock timestamps ────────────────────────────────────────
        if "system_time" in df.columns:
            parsed = pd.to_datetime(df["system_time"], errors="coerce", utc=True)
            _epoch = pd.Timestamp("1970-01-01", tz="UTC")
            # Always seconds regardless of internal pandas dtype
            ts_seconds = (parsed - _epoch).dt.total_seconds().values
        else:
            logger.warning("%s: no system_time column — falling back to synthetic timestamps", asset)
            ts_seconds = np.arange(len(df), dtype=np.float64) * 60.0

        features, _ = extract_lob_features(df, n_levels=15)
        X, y = label_real_spoofing(df, features, window_length=100, window_stride=window_stride, n_levels=15)

        per_asset_X.append(X)
        per_asset_y.append(y)
        per_asset_ts_ms[asset] = ts_seconds
        logger.info("  %s: %d windows, %.1f%% spoofing", asset, len(X), 100 * y.mean())

    # ── Normalise timestamps for Hawkes ───────────────────────────────────────
    # Global t0 = earliest second across all assets.
    # Convert to minutes so beta2=0.5 ↔ ~2-minute cross-asset excitation window.
    global_t0 = min(ts[0] for ts in per_asset_ts_ms.values() if len(ts) > 0)

    per_asset_ts: dict[str, np.ndarray] = {}
    for asset, ts_seconds in per_asset_ts_ms.items():
        ts_minutes = (ts_seconds - global_t0) / 60.0          # seconds → minutes
        per_asset_ts[asset] = ts_minutes[:_HAWKES_MAX_EVENTS]  # cap for O(n²) MLE

    # ── Align to shortest asset and stack ────────────────────────────────────
    min_windows = min(len(X) for X in per_asset_X)
    aligned_X = [X[:min_windows] for X in per_asset_X]        # each: (W, T, D)
    aligned_y = [y[:min_windows] for y in per_asset_y]

    X_multi = np.stack(aligned_X, axis=1).astype(np.float32)  # (W, N, T, D)
    y_union = np.stack(aligned_y, axis=1).any(axis=1).astype(np.int64)

    logger.info(
        "Multi-asset dataset: %d windows × %d assets × %d steps × %d features",
        *X_multi.shape,
    )
    logger.info("Union spoofing rate: %.1f%%", 100 * y_union.mean())

    return X_multi, y_union, per_asset_ts


# ── Training loop ─────────────────────────────────────────────────────────────
def train_tengnn(
    model: TENGNNModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    adj_tensor: torch.Tensor,
    config: TrainingConfig,
    device: torch.device,
) -> dict:
    model = model.to(device)
    adj_tensor = adj_tensor.to(device)

    # Inverse-frequency class weights
    all_labels = torch.cat([y for _, y in train_loader]).numpy()
    counts = np.bincount(all_labels, minlength=2).astype(np.float64)
    weights = torch.tensor(all_labels.size / (2 * counts.clip(min=1)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs
    )

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    patience_counter = 0
    best_metrics: dict = {}

    for epoch in range(1, config.max_epochs + 1):
        t0 = time.perf_counter()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        train_preds, train_labels = [], []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)   # (B, N, T, D)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch, adj_tensor)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            train_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

        train_m = compute_metrics(np.array(train_labels), np.array(train_preds))
        train_m["loss"] = total_loss / max(len(train_loader), 1)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch, adj_tensor)
                val_loss += criterion(logits, y_batch).item()
                val_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_m = compute_metrics(np.array(val_labels), np.array(val_preds))
        val_m["loss"] = val_loss / max(len(val_loader), 1)

        scheduler.step()
        elapsed = time.perf_counter() - t0
        lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %3d/%d | train loss=%.4f F1=%.3f | val loss=%.4f F1=%.3f P=%.3f R=%.3f | lr=%.2e | %.1fs",
            epoch, config.max_epochs,
            train_m["loss"], train_m["f1"],
            val_m["loss"], val_m["f1"], val_m["precision"], val_m["recall"],
            lr, elapsed,
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            patience_counter = 0
            best_metrics = {"epoch": epoch, **{f"val_{k}": v for k, v in val_m.items()}}
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "adj": adj_tensor.cpu().numpy(),
                "config": {
                    "n_assets": model.n_assets,
                    "d_input": model.d_input,
                    "d_model": model.d_model,
                    "n_layers": model.ten_encoder.n_layers,
                    "d_ff": model.ten_encoder.d_ff,
                    "n_heads": model.ten_encoder.n_heads,
                },
            }, ckpt_dir / "best_model.pt")
            logger.info("  ★ New best val F1=%.4f — saved checkpoint", best_val_f1)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, config.patience)
                break

    return best_metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert len(args.assets) >= 2, "Need at least 2 assets for the GNN graph."

    print("=" * 80)
    print("TRAINING TEN-GNN SPOOFING DETECTOR  (multi-asset)")
    print("=" * 80)
    print(f"Assets        : {args.assets}")
    print(f"Frequency     : {args.frequency}")
    print(f"Max rows      : {args.max_rows or 'all'}")
    print(f"Window stride : {args.window_stride}")
    print(f"Epochs        : {args.epochs}")
    print(f"d_model       : {args.d_model}  n_layers: {args.n_layers}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading and aligning multi-asset data...")
    X_multi, y_union, asset_ts = build_multi_asset_dataset(
        args.assets, args.frequency, args.max_rows, args.seed,
        window_stride=args.window_stride,
    )
    n_windows, n_assets, seq_len, n_features = X_multi.shape

    # ── 2. Hawkes adjacency ───────────────────────────────────────────────────
    print("\n[2] Estimating Hawkes cross-asset adjacency...")
    adj_np = build_hawkes_adjacency(asset_ts, n_assets)
    adj_tensor = torch.tensor(adj_np, dtype=torch.float32)

    # ── 3. Chronological split ────────────────────────────────────────────────
    print("\n[3] Splitting into train / val / test (chronological)...")
    i1 = int(n_windows * 0.65)
    i2 = int(n_windows * 0.86)

    def make_loader(X, y, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          pin_memory=torch.cuda.is_available())

    train_loader = make_loader(X_multi[:i1],   y_union[:i1],   shuffle=True)
    val_loader   = make_loader(X_multi[i1:i2], y_union[i1:i2], shuffle=False)
    test_loader  = make_loader(X_multi[i2:],   y_union[i2:],   shuffle=False)

    for name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        n = len(loader.dataset)
        pos = int(loader.dataset.tensors[1].sum().item())
        print(f"    {name}: {n} windows  ({pos} spoofing, {100*pos/max(n,1):.1f}%)")

    # ── 4. Build model ────────────────────────────────────────────────────────
    print(f"\n[4] Building TENGNNModel ({n_assets} assets, {n_features} features)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_ff = args.d_model * 4
    n_heads = min(8, args.d_model // 32)   # keep head_dim >= 32
    model = TENGNNModel(
        n_assets=n_assets,
        d_input=n_features,
        d_model=args.d_model,
        n_heads=n_heads,
        n_layers=args.n_layers,
        d_ff=d_ff,
        dropout=args.dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params:,}  |  Device: {device}")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    ckpt_dir = args.checkpoint_dir or f"./checkpoints/tengnn_{'_'.join(a.lower() for a in args.assets)}_{args.frequency}"
    config = TrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-4,
        grad_clip=1.0,
        patience=8,
        checkpoint_dir=ckpt_dir,
        log_file=f"training_log_tengnn_{'_'.join(a.lower() for a in args.assets)}.csv",
        device="auto",
    )

    print(f"\n[5] Training for up to {args.epochs} epochs (patience=8)...")
    best = train_tengnn(model, train_loader, val_loader, adj_tensor, config, device)

    # ── 6. Test evaluation ────────────────────────────────────────────────────
    print("\n[6] Loading best checkpoint and evaluating on test set...")
    ckpt = torch.load(Path(ckpt_dir) / "best_model.pt", map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_preds, test_labels = [], []
    adj_tensor = adj_tensor.to(device)
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(device), adj_tensor)
            test_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            test_labels.extend(y_batch.numpy())

    test_m = compute_metrics(np.array(test_labels), np.array(test_preds))

    # ── 7. Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest Validation (Epoch {best.get('epoch')}):")
    print(f"  F1:        {best.get('val_f1', 0):.4f}")
    print(f"  Precision: {best.get('val_precision', 0):.4f}")
    print(f"  Recall:    {best.get('val_recall', 0):.4f}")
    print(f"\nTest Set:")
    print(f"  F1:        {test_m['f1']:.4f}")
    print(f"  Precision: {test_m['precision']:.4f}")
    print(f"  Recall:    {test_m['recall']:.4f}")
    print(f"\nCheckpoint: {ckpt_dir}/best_model.pt")
    print(f"Adjacency matrix saved inside checkpoint['adj']")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    exit(main())
