"""
ood_validation.py — Multi-Exchange Out-of-Distribution (OOD) Analysis
=====================================================================

Restores the paper-aligned breakdown of model robustness across three 
distinct market microstructures: Nasdaq (Equities), Eurex (Futures), 
and Binance (Crypto).
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import compute_point_metrics
from src.data.adversarial_injection import AdversarialInjector
from src.models.ten_gnn import TENGNNModel

logger = logging.getLogger(__name__)

class MultiExchangeOODGenerator(AdversarialInjector):
    """Extension of AdversarialInjector to simulate specific market microstructures."""
    
    def generate_nasdaq_equity_sim(self, n_sequences: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Nasdaq: Small ticks, Maker-Taker fees, Session Gaps."""
        self.tick_size = 0.0001
        X, y = self.generate_synthetic_spoofing_sequences(n_sequences=n_sequences)
        
        # 1. Maker-taker fee: Add random price offset (+/- 0.00005)
        fee_noise = self.rng.normal(0, 0.00005, size=X.shape[:2])
        X[..., :20] += fee_noise[..., np.newaxis] # add to all 20 price levels
        
        # 2. Session gaps: Zero out activity in a 'night' portion (simulated by zeroing first 20% of seq)
        # In this simplistic sim, we'll just zero out some features for the 'pre-market' part
        X[:, :20, 41] = 0 # zero volume
        
        return X, y

    def generate_eurex_futures_sim(self, n_sequences: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Eurex: CET Time shift, Lower participant mix, DAX-style pricing."""
        self.tick_size = 0.5
        # High base price (~18000)
        X, y = self.generate_synthetic_spoofing_sequences(n_sequences=n_sequences)
        X[..., :20] += 18000.0 
        X[..., 46] += 18000.0
        
        # Lower participant mix: reduce volumes by 40%
        X[..., 20:40] *= 0.6
        X[..., 41] *= 0.6
        
        return X, y

    def generate_binance_crypto_sim(self, n_sequences: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Binance: 24/7, high volatility, no market maker, extra feature."""
        self.tick_size = 0.01
        X, y = self.generate_synthetic_spoofing_sequences(n_sequences=n_sequences)
        
        # 1. Higher volatility: multiply price changes by 2.5x
        price_cols = list(range(20)) + [46]
        for b in range(X.shape[0]):
            for c in price_cols:
                diffs = np.diff(X[b, :, c])
                X[b, 1:, c] = X[b, 0, c] + np.cumsum(diffs * 2.5)
        
        # 2. No DMM: increase spread variability by 3x
        X[..., 45] *= self.rng.uniform(1, 3, size=X.shape[:2])
        
        # 3. Funding rate feature: Add 47th feature noise
        funding_rate = np.cumsum(self.rng.normal(0, 0.0001, size=X.shape[:2]), axis=1) + 0.0001
        X[..., 43] += funding_rate.astype(np.float32) 
        
        return X, y

def run_multi_ood_validation(output_json: Path, checkpoint_path: Path) -> None:
    """Run OOD validation across simulated exchange microstructures.

    Loads the model from checkpoint_path and reads n_assets/d_input from the
    saved config. Uses random weights + identity adjacency if the checkpoint
    does not exist (F1 numbers are meaningless in that case).

    NOTE: All OOD scenarios are synthetic simulations of different microstructures
    (Nasdaq/Eurex/Binance). They are NOT real exchange data. They test robustness
    to feature distribution shifts, not real cross-exchange generalisation.
    """
    logger.info("Initializing Multi-Exchange OOD Validation...")

    device = torch.device("cpu")

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        n_assets = cfg.get("n_assets", 3)
        d_input  = cfg.get("d_input", 33)
        adj_np   = ckpt.get("adj", None)
        model = TENGNNModel(n_assets=n_assets, d_input=d_input).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        adj = (torch.tensor(adj_np, dtype=torch.float32, device=device)
               if adj_np is not None else torch.eye(n_assets, device=device))
        logger.info("Loaded checkpoint: %s  (n_assets=%d, d_input=%d)", checkpoint_path, n_assets, d_input)
        checkpoint_note = "real model inference"
    else:
        logger.warning(
            "Checkpoint not found at %s — using random weights. "
            "F1 numbers will not be meaningful. Train the model first.",
            checkpoint_path,
        )
        n_assets, d_input = 3, 33
        model = TENGNNModel(n_assets=n_assets, d_input=d_input).to(device)
        adj = torch.eye(n_assets, device=device)
        checkpoint_note = "random-weight model — train before interpreting"

    model.eval()

    gen = MultiExchangeOODGenerator(seed=42)

    scenarios = [
        ("In-Distribution (synthetic)", gen.generate_synthetic_spoofing_sequences),
        ("OOD: Nasdaq equity sim",      gen.generate_nasdaq_equity_sim),
        ("OOD: Eurex futures sim",      gen.generate_eurex_futures_sim),
        ("OOD: Binance crypto sim",     gen.generate_binance_crypto_sim),
    ]

    summary_dict: dict = {}
    id_f1: float = 0.0

    print("\n" + "=" * 85)
    print("OOD VALIDATION  (synthetic microstructure simulations — NOT real exchange data)")
    print(f"{'Test Set':<35} | {'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'ΔF1 vs ID':<10}")
    print("-" * 85)

    for label, generator_fn in scenarios:
        X, y = generator_fn(n_sequences=200)
        # Generators default to n_features=49. Truncate or zero-pad to d_input.
        if X.shape[-1] > d_input:
            X = X[..., :d_input]
        elif X.shape[-1] < d_input:
            pad = np.zeros((*X.shape[:-1], d_input - X.shape[-1]), dtype=np.float32)
            X = np.concatenate([X, pad], axis=-1)

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            # Expand single-asset sequences to (B, n_assets, T, D)
            if X_tensor.ndim == 3:
                X_tensor = X_tensor.unsqueeze(1).expand(-1, n_assets, -1, -1)
            logits = model(X_tensor, adj)
            preds = logits.argmax(dim=-1).cpu().numpy()

        metrics = compute_point_metrics(y, preds)
        f1 = metrics["f1"]
        prec = metrics["precision"]
        rec = metrics["recall"]

        if "In-Distribution" in label:
            id_f1 = f1
            delta_str = "—"
        else:
            delta_str = f"{id_f1 - f1:+.3f}"

        row = {"Set": label, "Precision": prec, "Recall": rec, "F1": f1,
               "note": checkpoint_note}
        summary_dict[label] = row

        print(f"{label:<35} | {prec:.3f}     | {rec:.3f}  | {f1:.3f}  | {delta_str:<10}")

    print("-" * 85)
    if "random-weight" in checkpoint_note:
        print("\nWARNING: results above are from a randomly-initialised model.")
        print("         Run train_tengnn_full.py and pass the checkpoint to get real OOD numbers.")

    existing_data: dict = {}
    if output_json.exists():
        with open(output_json, "r") as f:
            existing_data = json.load(f)

    existing_data["multi_exchange_ood"] = summary_dict
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(existing_data, f, indent=4)

    logger.info("Saved Multi-Exchange OOD results to %s", output_json)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parent.parent.parent
    out_path = root / "results" / "evaluation_report.json"
    ckpt_path = root / "checkpoints" / "tengnn_btc_eth_sol_1min" / "best_model.pt"
    run_multi_ood_validation(out_path, ckpt_path)
