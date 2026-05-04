#!/usr/bin/env python3
"""
Full training script for TEN spoofing detector on Kaggle crypto data.

Usage:
    python train_kaggle_full.py --asset BTC --frequency 1min --max-rows 100000 --epochs 50

For GPU training:
    CUDA_VISIBLE_DEVICES=0 python train_kaggle_full.py --asset BTC --frequency 1min
"""

import sys
sys.path.insert(0, 'src')

import argparse
from pathlib import Path

import torch
import numpy as np

from data.kaggle_loader import prepare_kaggle_data
from data.dataset import create_data_splits, create_dataloaders
from models.ten import TENSpoofingDetector
from training.trainer import TENTrainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train TEN on Kaggle crypto data")
    
    parser.add_argument("--asset", default="BTC", choices=["BTC", "ETH", "ADA"],
                        help="Cryptocurrency asset")
    parser.add_argument("--frequency", default="1min", choices=["1sec", "1min", "5min"],
                        help="Data frequency")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows to load (None = all)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("TRAINING TEN SPOOFING DETECTOR ON KAGGLE DATA")
    print("=" * 80)
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    print(f"\n[1] Loading {args.asset} {args.frequency} data...")
    if args.max_rows:
        print(f"    Max rows: {args.max_rows:,}")
    
    X, y = prepare_kaggle_data(
        asset=args.asset,
        frequency=args.frequency,
        max_rows=args.max_rows,
    )
    
    print(f"\n    Shape: {X.shape}")
    print(f"    Spoofing rate: {y.mean():.1%}")
    
    # Split
    print(f"\n[2] Creating train/val/test splits...")
    datasets = create_data_splits(X, y, seed=args.seed)
    
    for split, ds in datasets.items():
        print(f"    {split.upper()}: {len(ds)} samples ({ds.n_positive} spoofing)")
    
    # DataLoaders
    print(f"\n[3] Creating DataLoaders (batch_size={args.batch_size})...")
    loaders = create_dataloaders(datasets, batch_size=args.batch_size)
    
    # Model
    print(f"\n[4] Building TEN model...")
    print(f"    Input features: {X.shape[2]}")
    print(f"    Sequence length: {X.shape[1]}")
    
    model = TENSpoofingDetector(
        d_input=X.shape[2],
        d_model=256,
        n_heads=8,
        d_ff=1024,
        n_layers=6,
        n_classes=2,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params:,}")
    
    # Config
    print(f"\n[5] Training configuration...")
    
    checkpoint_dir = f"./checkpoints/kaggle_{args.asset.lower()}_{args.frequency}"
    log_file = f"training_log_kaggle_{args.asset.lower()}_{args.frequency}.csv"
    
    config = TrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-4,
        grad_clip=1.0,
        patience=8,
        checkpoint_dir=checkpoint_dir,
        log_file=log_file,
        device="auto",
    )
    
    print(f"    Epochs: {config.max_epochs}")
    print(f"    Learning rate: {config.lr}")
    print(f"    Device: {config.resolve_device()}")
    print(f"    Checkpoint dir: {checkpoint_dir}")
    
    # Train
    print(f"\n[6] Starting training...")
    print(f"    Training set: {len(loaders['train'].dataset)} samples")
    print(f"    Validation set: {len(loaders['val'].dataset)} samples")
    
    trainer = TENTrainer(model=model, config=config)
    best = trainer.train(loaders["train"], loaders["val"])
    
    # Evaluate
    print(f"\n[7] Loading best model and evaluating...")
    trainer.load_best_checkpoint()
    test_metrics = trainer.evaluate(loaders["test"])
    
    # Results
    print(f"\n" + "=" * 80)
    print(f"✅ TRAINING COMPLETE")
    print(f"=" * 80)
    
    print(f"\nBest Validation Metrics (Epoch {best.get('epoch')}):")
    print(f"  F1:        {best.get('val_f1'):.4f}")
    print(f"  Precision: {best.get('val_precision'):.4f}")
    print(f"  Recall:    {best.get('val_recall'):.4f}")
    print(f"  Loss:      {best.get('val_loss'):.4f}")
    
    print(f"\nTest Set Metrics:")
    print(f"  F1:        {test_metrics.get('f1'):.4f}")
    print(f"  Precision: {test_metrics.get('precision'):.4f}")
    print(f"  Recall:    {test_metrics.get('recall'):.4f}")
    print(f"  Loss:      {test_metrics.get('loss'):.4f}")
    
    print(f"\n📊 Improvement vs Synthetic (Expected):")
    print(f"  F1: 0.80 → {test_metrics.get('f1'):.2f} (actual)")
    print(f"  Precision: 0.81 → {test_metrics.get('precision'):.2f}")
    print(f"  Recall: 0.77 → {test_metrics.get('recall'):.2f}")
    
    print(f"\n💾 Model saved to: {checkpoint_dir}/best_model.pt")
    print(f"📈 Logs saved to: {log_file}")
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"SUMMARY")
    print(f"=" * 80)
    print(f"Asset: {args.asset}")
    print(f"Frequency: {args.frequency}")
    print(f"Total samples: {len(X)}")
    print(f"Training epochs: {best.get('epoch')}")
    print(f"Best validation F1: {best.get('val_f1'):.4f}")
    print(f"Test F1: {test_metrics.get('f1'):.4f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
