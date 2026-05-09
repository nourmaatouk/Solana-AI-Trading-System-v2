"""
smoke_test.py — End-to-end smoke test for the LOB data pipeline
================================================================

Runs the full pipeline:
1. Generate synthetic LOBSTER data
2. Extract 47 features
3. Label events & window into sequences
4. Inject adversarial sequences
5. Build PyTorch datasets with stratified splits
6. Verify shapes, dtypes, and DataLoader iteration

Usage:
    python -m src.data.smoke_test
    # or from the spoofing_detection directory:
    python src/data/smoke_test.py
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import torch

# ── Setup logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_test")


def main() -> None:
    t0 = time.perf_counter()

    logger.info("=" * 70)
    logger.info("LOB SPOOFING DETECTION — DATA PIPELINE SMOKE TEST")
    logger.info("=" * 70)

    # ── Step 1: Generate synthetic LOBSTER data ─────────────────────────────
    logger.info("\n[1/6] Generating synthetic LOBSTER data...")
    from src.data.lobster_loader import generate_synthetic_lobster_data

    df = generate_synthetic_lobster_data(n_events=5_000, seed=42)
    logger.info("  → DataFrame shape: %s", df.shape)
    logger.info("  → Columns: %s", list(df.columns[:10]) + ["..."])
    logger.info("  → Event types: %s", dict(df["event_type"].value_counts()))
    assert len(df) == 5_000, f"Expected 5000 rows, got {len(df)}"
    assert "timestamp_ns" in df.columns
    assert "bid_price_1" in df.columns
    assert "ask_vol_10" in df.columns
    logger.info("  ✓ LOBSTER loader OK")

    # ── Step 2: Feature engineering ─────────────────────────────────────────
    logger.info("\n[2/6] Extracting features...")
    from src.data.feature_engineering import LOBFeatureEngineer

    engineer = LOBFeatureEngineer(n_levels=10, rolling_window=20)
    features = engineer.fit_transform(df)  # default n_output_features=49
    logger.info("  → Feature matrix shape: %s", features.shape)
    logger.info("  → Feature names (first 5): %s", engineer.feature_names[:5])
    assert features.shape == (5_000, 49), f"Expected (5000, 49), got {features.shape}"
    assert not np.isnan(features).any(), "NaNs in features!"
    assert not np.isinf(features).any(), "Infs in features!"
    # Verify scaler is fitted (mean ≈ 0, std ≈ 1)
    assert abs(features.mean()) < 0.01, f"Mean not near 0: {features.mean()}"
    assert abs(features.std() - 1.0) < 0.1, f"Std not near 1: {features.std()}"
    logger.info("  ✓ Feature engineering OK (49 features, normalized)")

    # ── Step 3: Labeling & windowing ────────────────────────────────────────
    logger.info("\n[3/6] Computing labels and creating windows...")
    from src.data.labeling import SpoofingLabeler

    labeler = SpoofingLabeler(tick_size=0.01)
    mid_prices = engineer.get_mid_prices(df)
    timestamps_ms = engineer.get_timestamps_ms(df)

    event_labels = labeler.compute_event_labels(df, mid_prices, timestamps_ms)
    logger.info("  → Per-event labels: %d positive / %d total",
                event_labels.sum(), len(event_labels))

    X, y = labeler.create_sequences(features, event_labels)
    logger.info("  → X shape: %s, y shape: %s", X.shape, y.shape)
    assert X.ndim == 3, f"Expected 3D array, got {X.ndim}D"
    assert X.shape[1] == 100, f"Expected window_length=100, got {X.shape[1]}"
    assert X.shape[2] == 49, f"Expected 49 features, got {X.shape[2]}"
    assert len(X) == len(y)
    logger.info("  ✓ Labeling & windowing OK")

    # ── Step 4: Adversarial injection (standalone synthetic) ────────────────
    logger.info("\n[4/6] Generating adversarial sequences...")
    from src.data.adversarial_injection import AdversarialInjector

    injector = AdversarialInjector(seed=123)
    X_adv, y_adv = injector.generate_synthetic_spoofing_sequences(
        n_sequences=100, window_length=100, n_features=49
    )
    logger.info("  → Adversarial X shape: %s, y shape: %s", X_adv.shape, y_adv.shape)
    assert X_adv.shape == (100, 100, 49), f"Expected (100,100,49), got {X_adv.shape}"
    assert y_adv.shape == (100,)
    n_spoof = y_adv.sum()
    n_legit = len(y_adv) - n_spoof
    logger.info("  → %d spoofing, %d legitimate", n_spoof, n_legit)
    logger.info("  ✓ Adversarial injection OK")

    # ── Step 5: Dataset & splits ────────────────────────────────────────────
    logger.info("\n[5/6] Building PyTorch datasets with stratified splits...")
    from src.data.dataset import (
        SpoofingDataset, create_data_splits, create_dataloaders,
        build_smoke_test_data
    )

    datasets = create_data_splits(X_adv, y_adv, seed=42)

    total = sum(len(ds) for ds in datasets.values())
    logger.info("  → Total samples across splits: %d", total)
    for name, ds in datasets.items():
        logger.info(
            "    %s: %d samples, %d pos, %d neg, window=%d, features=%d",
            name, len(ds), ds.n_positive, ds.n_negative,
            ds.window_length, ds.n_features,
        )
    assert total == 100, f"Expected 100 total samples, got {total}"
    logger.info("  ✓ Dataset splits OK (65/21/14 normalized)")

    # ── Step 6: DataLoader iteration ────────────────────────────────────────
    logger.info("\n[6/6] Testing DataLoader iteration...")
    loaders = create_dataloaders(datasets, batch_size=16)

    for name, loader in loaders.items():
        batch_count = 0
        for batch_x, batch_y in loader:
            assert batch_x.dtype == torch.float32
            assert batch_y.dtype == torch.long
            assert batch_x.ndim == 3  # (batch, window, features)
            batch_count += 1
        logger.info("  %s loader: %d batches", name, batch_count)

    logger.info("  ✓ DataLoader iteration OK")

    # ── Quick test of build_smoke_test_data shortcut ────────────────────────
    logger.info("\n[BONUS] Testing build_smoke_test_data() shortcut...")
    smoke_ds = build_smoke_test_data(n_sequences=50, seed=99)
    assert "train" in smoke_ds and "val" in smoke_ds and "test" in smoke_ds
    logger.info("  ✓ build_smoke_test_data OK")

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    logger.info("\n" + "=" * 70)
    logger.info("ALL SMOKE TESTS PASSED ✓  (%.2f seconds)", elapsed)
    logger.info("=" * 70)
    logger.info(
        "\nPipeline summary:\n"
        "  • lobster_loader    → 5,000 synthetic events\n"
        "  • feature_engineer  → (5000, 49) normalized features\n"
        "  • labeling          → %d windows (100 events, 50%% overlap)\n"
        "  • adversarial       → 100 synthetic sequences\n"
        "  • dataset           → train=%d / val=%d / test=%d\n",
        len(X),
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]),
    )


if __name__ == "__main__":
    # Ensure we can import from the package
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    main()
