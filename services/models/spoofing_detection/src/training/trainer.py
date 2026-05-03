"""
trainer.py — TEN Training Loop with Metrics Logging & Checkpointing
=====================================================================

Provides:
    - ``train_epoch()``: single training epoch with gradient clipping
    - ``eval_epoch()``: validation/test evaluation with full metrics
    - ``TENTrainer``: full training loop with:
        • AdamW optimizer, lr=5e-4
        • Cosine annealing scheduler
        • Gradient clipping (global norm=1.0)
        • Early stopping on val F1 (patience=8)
        • Best model checkpoint saving
        • Per-epoch CSV logging (loss, F1, precision, recall)

Training config (paper-aligned):
    batch_size   = 32
    max_epochs   = 50
    lr           = 5e-4
    grad_clip    = 1.0
    patience     = 8
    loss         = CrossEntropyLoss with inverse-frequency class weights
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class TrainingConfig:
    """Training hyperparameters (paper defaults)."""
    batch_size: int = 32
    max_epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 8         # early stopping patience on val F1
    checkpoint_dir: str = "checkpoints"
    log_file: str = "training_log.csv"
    device: str = "auto"      # "auto", "cuda", "cpu"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Epoch-level functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    hawkes_estimator = None,
) -> Dict[str, float]:
    """Run one training epoch.

    Parameters
    ----------
    model : the TEN or TEN-GNN model.
    loader : training DataLoader.
    criterion : loss function (CrossEntropyLoss with class weights).
    optimizer : AdamW optimizer.
    device : torch device.
    grad_clip : max gradient global norm.
    hawkes_estimator : optional Hawkes estimator to fetch adjacency.

    Returns
    -------
    Dict with keys: loss, f1, precision, recall.
    """
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Fetch dynamic adjacency from Hawkes background thread if model supports it
        if hawkes_estimator is not None:
            A = hawkes_estimator.get_adjacency()
            A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
        else:
            A_tensor = None

        optimizer.zero_grad()

        if A_tensor is not None:
            logits = model(batch_x, A_tensor)
        else:
            logits = model(batch_x)  # (batch, n_classes)
            
        loss = criterion(logits, batch_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        labels = batch_y.detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = _compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(n_batches, 1)

    return metrics


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    hawkes_estimator = None,
) -> Dict[str, float]:
    """Run one evaluation epoch (no gradient computation).

    Parameters
    ----------
    model : the TEN or TEN-GNN model.
    loader : validation or test DataLoader.
    criterion : loss function.
    device : torch device.
    hawkes_estimator : optional Hawkes estimator to fetch adjacency.

    Returns
    -------
    Dict with keys: loss, f1, precision, recall.
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if hawkes_estimator is not None:
                A = hawkes_estimator.get_adjacency()
                A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
                logits = model(batch_x, A_tensor)
            else:
                logits = model(batch_x)
                
            loss = criterion(logits, batch_y)

            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch_y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = _compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(n_batches, 1)

    return metrics


def _compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics with zero_division safety."""
    n_classes = len(np.unique(labels))

    if n_classes < 2 or len(labels) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    return {
        "f1": float(f1_score(labels, preds, zero_division=0.0)),
        "precision": float(precision_score(labels, preds, zero_division=0.0)),
        "recall": float(recall_score(labels, preds, zero_division=0.0)),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEN Trainer (full loop)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TENTrainer:
    """Full training loop for the TEN spoofing detector.

    Features:
        - AdamW + cosine annealing LR
        - Early stopping on val F1 with configurable patience
        - Best checkpoint saving
        - Per-epoch CSV logging

    Parameters
    ----------
    model : TENSpoofingDetector instance.
    config : TrainingConfig (uses paper defaults if not provided).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        hawkes_estimator = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.device = self.config.resolve_device()
        self.model = self.model.to(self.device)
        self.hawkes_estimator = hawkes_estimator

        # ── Optimizer ────────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # ── Scheduler ────────────────────────────────────────────────────────
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs,
        )

        # ── State ────────────────────────────────────────────────────────────
        self.best_val_f1: float = -1.0
        self.patience_counter: int = 0
        self.history: list[Dict[str, float]] = []

    def _build_criterion(self, train_loader: DataLoader) -> nn.Module:
        """Build CrossEntropyLoss with inverse-frequency class weights.

        Weights are proportional to 1 / class_frequency so the minority
        class (spoofing) gets higher weight.
        """
        # Collect all labels from the training set
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        all_labels = np.array(all_labels)

        n_classes = len(np.unique(all_labels))
        class_counts = np.bincount(all_labels, minlength=n_classes).astype(np.float64)
        # Inverse frequency: total / (n_classes × count_per_class)
        total = class_counts.sum()
        class_weights = total / (n_classes * class_counts.clip(min=1))

        weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        logger.info(
            "Class weights: %s (counts: %s)",
            class_weights.tolist(), class_counts.tolist(),
        )

        return nn.CrossEntropyLoss(weight=weights)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Run the full training loop.

        Parameters
        ----------
        train_loader : training DataLoader.
        val_loader : validation DataLoader.

        Returns
        -------
        Dict with best validation metrics.
        """
        criterion = self._build_criterion(train_loader)

        # Ensure checkpoint directory exists
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # CSV log setup
        log_path = ckpt_dir / self.config.log_file
        csv_fields = [
            "epoch", "lr",
            "train_loss", "train_f1", "train_precision", "train_recall",
            "val_loss", "val_f1", "val_precision", "val_recall",
            "elapsed_s",
        ]
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

        logger.info("Training on %s for up to %d epochs (patience=%d)",
                     self.device, self.config.max_epochs, self.config.patience)
        logger.info("Logging to %s", log_path)

        best_metrics: Dict[str, float] = {}

        for epoch in range(1, self.config.max_epochs + 1):
            t0 = time.perf_counter()

            # ── Train ────────────────────────────────────────────────────────
            train_metrics = train_epoch(
                model=self.model,
                loader=train_loader,
                criterion=criterion,
                optimizer=self.optimizer,
                device=self.device,
                grad_clip=self.config.grad_clip,
                hawkes_estimator=self.hawkes_estimator,
            )

            # ── Evaluate ─────────────────────────────────────────────────────
            val_metrics = eval_epoch(
                model=self.model,
                loader=val_loader,
                criterion=criterion,
                device=self.device,
                hawkes_estimator=self.hawkes_estimator,
            )

            # ── Scheduler step ───────────────────────────────────────────────
            self.scheduler.step()

            elapsed = time.perf_counter() - t0
            current_lr = self.scheduler.get_last_lr()[0]

            # ── Logging ──────────────────────────────────────────────────────
            row = {
                "epoch": epoch,
                "lr": f"{current_lr:.6f}",
                "train_loss": f"{train_metrics['loss']:.4f}",
                "train_f1": f"{train_metrics['f1']:.4f}",
                "train_precision": f"{train_metrics['precision']:.4f}",
                "train_recall": f"{train_metrics['recall']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "val_f1": f"{val_metrics['f1']:.4f}",
                "val_precision": f"{val_metrics['precision']:.4f}",
                "val_recall": f"{val_metrics['recall']:.4f}",
                "elapsed_s": f"{elapsed:.2f}",
            }
            with open(log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow(row)

            self.history.append({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            logger.info(
                "Epoch %3d/%d │ train loss=%.4f F1=%.3f │ "
                "val loss=%.4f F1=%.3f P=%.3f R=%.3f │ "
                "lr=%.2e │ %.1fs",
                epoch, self.config.max_epochs,
                train_metrics["loss"], train_metrics["f1"],
                val_metrics["loss"], val_metrics["f1"],
                val_metrics["precision"], val_metrics["recall"],
                current_lr, elapsed,
            )

            # ── Checkpoint (best val F1) ─────────────────────────────────────
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.patience_counter = 0
                best_metrics = {
                    "epoch": epoch,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }

                ckpt_path = ckpt_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "val_f1": self.best_val_f1,
                    "config": {
                        "d_input": getattr(self.model, "d_input", None),
                        "d_model": getattr(self.model, "d_model", None),
                        "n_heads": getattr(self.model, "n_heads", None),
                        "n_layers": getattr(self.model, "n_layers", None),
                        "d_ff": getattr(self.model, "d_ff", None),
                        "n_classes": getattr(self.model, "n_classes", None),
                    },
                }, ckpt_path)
                logger.info("  ★ New best val F1=%.4f → saved %s",
                            self.best_val_f1, ckpt_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d exhausted, "
                        "best val F1=%.4f at epoch %d)",
                        epoch, self.config.patience,
                        self.best_val_f1, best_metrics.get("epoch", -1),
                    )
                    break

        logger.info("Training complete. Best val F1=%.4f", self.best_val_f1)
        return best_metrics

    def load_best_checkpoint(self, path: Optional[str] = None) -> None:
        """Load the best model checkpoint."""
        if path is None:
            path = str(Path(self.config.checkpoint_dir) / "best_model.pt")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Loaded checkpoint from %s (epoch %d, val F1=%.4f)",
            path, checkpoint["epoch"], checkpoint["val_f1"],
        )

    def evaluate(
        self, test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate the model on a test set.

        Returns dict with loss, f1, precision, recall.
        """
        # Use a simple unweighted CE for evaluation
        criterion = nn.CrossEntropyLoss()
        return eval_epoch(
            model=self.model,
            loader=test_loader,
            criterion=criterion,
            device=self.device,
            hawkes_estimator=self.hawkes_estimator,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone smoke test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.models.ten import TENSpoofingDetector
    from src.data.dataset import SpoofingDataset, create_dataloaders

    print("=" * 70)
    print("TEN TRAINER — 2-EPOCH SMOKE TEST ON DUMMY DATA")
    print("=" * 70)

    # ── Create dummy data ────────────────────────────────────────────────────
    batch_size = 32
    seq_len = 100
    d_input = 47
    n_samples = 128

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, seq_len, d_input)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples).astype(np.int64)

    # Make it slightly imbalanced (70% negative, 30% positive)
    y[:int(0.7 * n_samples)] = 0
    y[int(0.7 * n_samples):] = 1
    rng.shuffle(y)

    # Split manually
    n_train = 80
    n_val = 24
    n_test = n_samples - n_train - n_val

    train_ds = SpoofingDataset(X[:n_train], y[:n_train])
    val_ds = SpoofingDataset(X[n_train:n_train + n_val], y[n_train:n_train + n_val])
    test_ds = SpoofingDataset(X[n_train + n_val:], y[n_train + n_val:])

    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    loaders = create_dataloaders(datasets, batch_size=batch_size)

    print(f"Train: {len(train_ds)} samples ({train_ds.n_positive} pos)")
    print(f"Val:   {len(val_ds)} samples ({val_ds.n_positive} pos)")
    print(f"Test:  {len(test_ds)} samples ({test_ds.n_positive} pos)")

    # ── Build model & trainer ────────────────────────────────────────────────
    model = TENSpoofingDetector(
        d_input=d_input,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        n_layers=6,
        n_classes=2,
    )

    config = TrainingConfig(
        max_epochs=2,
        patience=8,
        checkpoint_dir="/tmp/ten_smoke_test",
    )

    trainer = TENTrainer(model=model, config=config)

    # ── Train ────────────────────────────────────────────────────────────────
    best = trainer.train(loaders["train"], loaders["val"])
    print(f"\nBest val metrics: {best}")

    # ── Evaluate on test ─────────────────────────────────────────────────────
    trainer.load_best_checkpoint()
    test_metrics = trainer.evaluate(loaders["test"])
    print(f"Test metrics: {test_metrics}")

    # ── Verify CSV log ───────────────────────────────────────────────────────
    log_path = Path(config.checkpoint_dir) / config.log_file
    with open(log_path) as f:
        lines = f.readlines()
    print(f"\nCSV log ({log_path}): {len(lines)} lines (header + {len(lines)-1} epochs)")
    for line in lines:
        print(f"  {line.strip()}")

    print("\n" + "=" * 70)
    print("TRAINER SMOKE TEST PASSED ✓")
    print("=" * 70)
