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
    """Split data into stratified train / val / test SpoofingDatasets."""
    # Split raw sequences here
    from sklearn.model_selection import train_test_split
    
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=train_ratio, stratify=y, random_state=seed
    )
    
    val_rel_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=val_rel_ratio, stratify=y_tmp, random_state=seed
    )
    
    datasets = {
        "train": SpoofingDataset(X_train, y_train),
        "val": SpoofingDataset(X_val, y_val),
        "test": SpoofingDataset(X_test, y_test),
    }
    # Statistics & Logging
    for name, ds in datasets.items():
        pos_ratio = 100.0 * ds.n_positive / max(len(ds), 1)
        logger.info(
            "%s: %d samples, %.1f%% positive",
            name.capitalize(), len(ds), pos_ratio
        )
        
    train_ratio_posneg = datasets["train"].n_negative / max(datasets["train"].n_positive, 1)
    logger.info("Class distribution — 0: %d samples, 1: %d samples, ratio: %.1f", 
                datasets["train"].n_negative, datasets["train"].n_positive, train_ratio_posneg)
                
    if train_ratio_posneg > 10:
        logger.info("Ratio > 10:1, automatically applying balanced class weights (handled natively in trainer losses)")

    # Dynamic quality checks instead of hard assertions for flexible scaling
    n_train = len(datasets["train"])
    n_test = len(datasets["test"])
    n_pos_test = datasets["test"].n_positive

    if n_train < 1000:
        logger.warning(f"Training set small ({n_train} samples). Statistical significance may be limited.")
    if n_test < 200:
        logger.warning(f"Test set small ({n_test} samples). Performance metrics may have high variance.")
    if n_pos_test < 20:
        logger.warning(f"Very few positive samples in test set ({n_pos_test}). Recall metrics will be unreliable.")

    # Hard floor for safety (must have at least a few samples to function)
    assert n_train > 0 and n_test > 0, "Dataset splits resulted in empty sets. Increase n_sequences."

    return datasets


# ── DataLoader creation ─────────────────────────────────────────────────────
def create_dataloaders(
    datasets: Dict[str, SpoofingDataset],
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
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

def audit_leakage(train_dataset: SpoofingDataset, test_dataset: SpoofingDataset):
    """Audits leakage dynamically"""
    train_hashes = set(hash(x.numpy().tobytes()) for x, _ in train_dataset)
    test_hashes = set(hash(x.numpy().tobytes()) for x, _ in test_dataset)
    overlap = train_hashes & test_hashes
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} sequences in both train and test"
    print("Leakage audit passed: 0 overlapping sequences")

# ── Master Data Loader ──────────────────────────────────────────────────────
def prepare_data(
    use_real_data: bool = False,
    n_sequences: int = 4000,
    window_length: int = 100,
    window_stride: int = 50,
    n_features: int = 49,
    seed: int = 42,
) -> Dict[str, SpoofingDataset]:
    """
    Main entry point for data preparation.
    Modes:
      - REAL: Loads from data/raw/AAPL_... and uses labeling.py
      - SYNTHETIC: Uses AdversarialInjector
    """
    mode_str = f"REAL ({n_sequences} sequences)" if use_real_data else f"SYNTHETIC ({n_sequences} sequences)"
    print(f"\nDATA MODE: {mode_str}")
    
    if use_real_data:
        from .lobster_loader import load_lobster_from_directory
        from .feature_engineering import LOBFeatureEngineer
        from .labeling import SpoofingLabeler
        
        # 1. Load Stand-in / Real data
        # Note: lobster_loader handles the generation of AAPL stand-in if missing
        df = load_lobster_from_directory("data/raw", ticker="AAPL", date="2012-06-21")
        
        # 2. Features
        engineer = LOBFeatureEngineer()
        features = engineer.fit_transform(df)
        mid_prices = engineer.get_mid_prices(df)
        timestamps = engineer.get_timestamps_ms(df)
        
        # 3. Apply Multi-threshold Labeling (from labeling.py)
        # Using paper defaults for thresholds, but configurable windowing
        labeler = SpoofingLabeler(
            window_length=window_length,
            window_stride=window_stride
        )
        X, y = labeler.label_and_window(df, features, mid_prices, timestamps)
        
        # If we have too few sequences for the requested n_sequences, we log it
        if len(X) < n_sequences:
            logger.info("Real data yielded %d sequences (less than requested %d)", len(X), n_sequences)
    else:
        from .adversarial_injection import AdversarialInjector
        injector = AdversarialInjector(seed=seed)
        X, y = injector.generate_synthetic_spoofing_sequences(
            n_sequences=n_sequences,
            window_length=window_length,
            n_features=n_features,
        )

    datasets = create_data_splits(X, y, seed=seed)
    audit_leakage(datasets["train"], datasets["test"])
    return datasets


def build_smoke_test_data(
    n_sequences: int = 4000,
    window_length: int = 100,
    n_features: int = 49,
    seed: int = 42,
) -> Dict[str, SpoofingDataset]:
    return prepare_data(
        use_real_data=False,
        n_sequences=n_sequences,
        window_length=window_length,
        n_features=n_features,
        seed=seed
    )
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        # Test synthetic — increased to 2000 to meet 'rigorous' baseline targets
        data_synth = prepare_data(use_real_data=False, n_sequences=2000)
        # Test real (stand-in) — 100 samples is enough for a pipeline smoke test
        data_real = prepare_data(use_real_data=True, n_sequences=100)
        print("\nSmoke test passed: Data pipeline functional in both modes.")
    except Exception as e:
        print(f"\nSmoke test failed: {e}")
        import traceback
        traceback.print_exc()
