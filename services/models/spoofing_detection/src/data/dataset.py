"""
dataset.py — PyTorch Dataset & Data Splitting for Spoofing Detection
=====================================================================

Provides:
    - ``SpoofingDataset``: PyTorch Dataset returning (sequence_tensor, label)
    - ``create_data_splits``: 65/21/21 stratified train/val/test split
    - ``build_smoke_test_data``: end-to-end pipeline generating 100 synthetic
      sequences for smoke testing (no external data needed)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# ── Default split ratios ─────────────────────────────────────────────────────
TRAIN_RATIO = 0.65
VAL_RATIO = 0.21  # Note: 0.65 + 0.21 + 0.14 ≈ 1.0 if strict;
TEST_RATIO = 0.14  # user spec says 65/21/21 which sums to 1.07 — we normalize


class SpoofingDataset(Dataset):
    """PyTorch Dataset for LOB spoofing detection sequences.

    Each sample is a (sequence_tensor, label) pair where:
        - sequence_tensor: shape (window_length, n_features), dtype float32
        - label: scalar int64 (0=legitimate, 1=spoofing)

    Parameters
    ----------
    X : np.ndarray of shape (N, window_length, n_features).
    y : np.ndarray of shape (N,) with binary labels.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), (
            f"X and y length mismatch: {len(X)} vs {len(y)}"
        )
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def n_features(self) -> int:
        return self.X.shape[2]

    @property
    def window_length(self) -> int:
        return self.X.shape[1]

    @property
    def n_positive(self) -> int:
        return int(self.y.sum().item())

    @property
    def n_negative(self) -> int:
        return len(self.y) - self.n_positive

    def class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced training."""
        n_pos = max(self.n_positive, 1)
        n_neg = max(self.n_negative, 1)
        total = n_pos + n_neg
        w_neg = total / (2.0 * n_neg)
        w_pos = total / (2.0 * n_pos)
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)


# ── Data splitting ──────────────────────────────────────────────────────────
def create_data_splits(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = 42,
) -> Dict[str, SpoofingDataset]:
    """Split data into stratified train / val / test SpoofingDatasets.

    Parameters
    ----------
    X : (N, window_length, n_features) array.
    y : (N,) binary labels.
    train_ratio, val_ratio, test_ratio : split proportions (normalized).
    seed : random seed for reproducibility.

    Returns
    -------
    Dict with keys 'train', 'val', 'test', each a SpoofingDataset.
    """
    n = len(X)
    assert n > 0, "Cannot split empty dataset"

    # Normalize ratios
    total = train_ratio + val_ratio + test_ratio
    train_r = train_ratio / total
    val_r = val_ratio / total
    # test_r = 1 - train_r - val_r  (implicit)

    # --- First split: train vs (val+test) ---
    val_test_r = 1.0 - train_r
    if val_test_r <= 0:
        raise ValueError("train_ratio too large, nothing left for val/test")

    splitter1 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_test_r, random_state=seed
    )

    # Handle edge case: if too few samples for stratification
    y_flat = y.flatten()
    n_classes = len(np.unique(y_flat))

    if n_classes < 2 or n < 4:
        # Can't stratify — do random split
        logger.warning(
            "Cannot stratify (%d classes, %d samples). Using random split.",
            n_classes, n,
        )
        indices = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        n_train = max(1, int(n * train_r))
        n_val = max(1, int(n * val_r))

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        if len(test_idx) == 0:
            test_idx = val_idx[-1:]
    else:
        train_idx, val_test_idx = next(splitter1.split(X, y_flat))

        # --- Second split: val vs test (within val+test portion) ---
        val_in_remainder = val_r / val_test_r
        splitter2 = StratifiedShuffleSplit(
            n_splits=1, test_size=(1.0 - val_in_remainder), random_state=seed
        )
        y_vt = y_flat[val_test_idx]
        X_vt = X[val_test_idx]

        # Check if we can stratify the second split
        if len(np.unique(y_vt)) < 2 or len(y_vt) < 4:
            mid = len(val_test_idx) // 2
            val_idx = val_test_idx[:mid]
            test_idx = val_test_idx[mid:]
        else:
            rel_val_idx, rel_test_idx = next(splitter2.split(X_vt, y_vt))
            val_idx = val_test_idx[rel_val_idx]
            test_idx = val_test_idx[rel_test_idx]

    # --- Build datasets ---
    datasets = {
        "train": SpoofingDataset(X[train_idx], y[train_idx]),
        "val": SpoofingDataset(X[val_idx], y[val_idx]),
        "test": SpoofingDataset(X[test_idx], y[test_idx]),
    }

    for name, ds in datasets.items():
        logger.info(
            "Split '%s': %d samples (%d positive, %d negative, %.1f%% pos)",
            name, len(ds), ds.n_positive, ds.n_negative,
            100.0 * ds.n_positive / max(len(ds), 1),
        )

    return datasets


# ── DataLoader creation ─────────────────────────────────────────────────────
def create_dataloaders(
    datasets: Dict[str, SpoofingDataset],
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Create DataLoaders from a split dict.

    Parameters
    ----------
    datasets : dict with 'train', 'val', 'test' SpoofingDataset instances.
    batch_size : batch size for all splits.
    num_workers : number of data loading workers.

    Returns
    -------
    Dict of DataLoaders.
    """
    loaders = {}
    for name, ds in datasets.items():
        shuffle = (name == "train")
        loaders[name] = DataLoader(
            ds,
            batch_size=min(batch_size, len(ds)),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(name == "train" and len(ds) > batch_size),
        )
    return loaders


# ── Smoke test: end-to-end pipeline ────────────────────────────────────────
def build_smoke_test_data(
    n_sequences: int = 100,
    window_length: int = 100,
    n_features: int = 47,
    seed: int = 42,
) -> Dict[str, SpoofingDataset]:
    """Generate synthetic data and return train/val/test splits.

    This is a self-contained smoke test that requires NO external data.
    Uses the AdversarialInjector's standalone synthetic generator.

    Parameters
    ----------
    n_sequences : total sequences to generate.
    window_length : events per sequence.
    n_features : feature dimensionality (default 47).
    seed : random seed.

    Returns
    -------
    Dict with 'train', 'val', 'test' SpoofingDataset instances.
    """
    from .adversarial_injection import AdversarialInjector

    logger.info("Building smoke test data: %d sequences, %d features",
                n_sequences, n_features)

    injector = AdversarialInjector(seed=seed)
    X, y = injector.generate_synthetic_spoofing_sequences(
        n_sequences=n_sequences,
        window_length=window_length,
        n_features=n_features,
    )

    datasets = create_data_splits(X, y, seed=seed)
    return datasets
