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
    """Synthetic OOD scenarios in Kaggle 33-feature normalized space.

    Feature layout (matches kaggle_loader.extract_lob_features, n_levels=15):
        0          : midpoint        (z-scored)
        1          : spread          (z-scored)
        2,4,...,30 : bid notionals   levels 1-15 (z-scored, interleaved with asks)
        3,5,...,31 : ask notionals   levels 1-15 (z-scored, interleaved with bids)
        32         : order_imbalance (z-scored)

    The previous implementation called generate_synthetic_spoofing_sequences (LOBSTER
    49-feature space) and then truncated to 33 features.  After truncation, features
    0-19 held raw bid/ask price levels (~150 USD), which are thousands of standard
    deviations outside the Kaggle training distribution.  The model returned all-zero
    predictions on every OOD scenario, carrying zero information about robustness.

    All scenarios here produce sequences whose feature distributions are consistent
    with the Kaggle training distribution, then apply controlled perturbations that
    model each microstructure's distinctive properties.
    """

    # Kaggle 34-feature schema constants (matches N_FEATURES_KAGGLE in config.py)
    _MIDPOINT_IDX     = 0
    _SPREAD_IDX       = 1
    _BID_IDXS         = list(range(2, 32, 2))   # 2,4,...,30  (15 bid notional levels)
    _ASK_IDXS         = list(range(3, 32, 2))   # 3,5,...,31  (15 ask notional levels)
    _IMBALANCE_IDX    = 32
    _CANCEL_RATIO_IDX = 33
    _N_FEATURES       = 34
    _WIN              = 100

    def _generate_kaggle_base_sequences(
        self, n_sequences: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic sequences in normalized Kaggle 33-feature space.

        Produces balanced spoofing / legit sequences with the same feature
        semantics as the Kaggle training data (z-scored after StandardScaler).
        Spoofing sequences contain a layering phase (imbalance spike + volume
        inflation on one side of the book) embedded in the middle of the window.
        """
        n_spoof = n_sequences // 2
        n_legit = n_sequences - n_spoof
        X = np.zeros((n_sequences, self._WIN, self._N_FEATURES), dtype=np.float32)
        y = np.zeros(n_sequences, dtype=np.int64)

        for i in range(n_sequences):
            is_spoof = i >= n_legit
            y[i] = int(is_spoof)

            # Midpoint: small random walk (z-scored units)
            X[i, :, self._MIDPOINT_IDX] = np.cumsum(
                self.rng.normal(0, 0.01, self._WIN)).astype(np.float32)
            # Spread: positive, moderately variable
            X[i, :, self._SPREAD_IDX] = np.abs(
                self.rng.normal(0, 0.8, self._WIN)).astype(np.float32)
            # Bid/ask notionals: roughly N(0,1) after z-scoring
            for idx in self._BID_IDXS + self._ASK_IDXS:
                X[i, :, idx] = self.rng.normal(0, 1, self._WIN).astype(np.float32)
            # Imbalance: near zero for benign activity
            X[i, :, self._IMBALANCE_IDX] = self.rng.normal(0, 0.3, self._WIN).astype(np.float32)
            # Cancel ratio: low baseline for legit (most orders rest and execute)
            X[i, :, self._CANCEL_RATIO_IDX] = np.abs(
                self.rng.normal(0, 0.3, self._WIN)).astype(np.float32)

            if is_spoof:
                spoof_side = float(self.rng.choice([-1.0, 1.0]))
                t0 = int(self.rng.integers(20, 55))
                t1 = min(t0 + 30, self._WIN)
                ph = t1 - t0

                # Imbalance spike ramping from +2σ to +3σ
                X[i, t0:t1, self._IMBALANCE_IDX] = (
                    spoof_side * np.linspace(2.0, 3.0, ph)).astype(np.float32)

                # Inflate notionals on the spoof side
                inflate_idxs = self._BID_IDXS if spoof_side > 0 else self._ASK_IDXS
                for lv, idx in enumerate(inflate_idxs):
                    X[i, t0:t1, idx] *= float(2.5 + 0.1 * lv)

                # Cancel ratio spike during layering phase (high cancellations)
                X[i, t0:t1, self._CANCEL_RATIO_IDX] = np.linspace(
                    2.0, 3.5, ph).astype(np.float32)

        perm = self.rng.permutation(n_sequences)
        return X[perm], y[perm]

    def generate_nasdaq_equity_sim(self, n_sequences: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Nasdaq equities: very tight spread (small tick), maker-taker fee noise,
        pre-market session gap (first 10 steps zeroed)."""
        X, y = self._generate_kaggle_base_sequences(n_sequences)
        # Tight tick: compress spread to ~10% of crypto width
        X[..., self._SPREAD_IDX] *= 0.3
        # Maker-taker fee: tiny noise on spread
        X[..., self._SPREAD_IDX] += self.rng.normal(0, 0.05, size=X.shape[:2]).astype(np.float32)
        # Pre-market session gap: flatten imbalance and thin both sides of book
        X[:, :10, self._IMBALANCE_IDX] = 0.0
        for idx in self._BID_IDXS[10:] + self._ASK_IDXS[10:]:  # deep levels go quiet
            X[:, :10, idx] *= 0.1
        return X, y

    def generate_eurex_futures_sim(self, n_sequences: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Eurex futures: lower participant mix (40% volume reduction),
        systematic midpoint shift (different price regime in z-score space)."""
        X, y = self._generate_kaggle_base_sequences(n_sequences)
        # Fewer market participants → thinner book on both sides
        for idx in self._BID_IDXS + self._ASK_IDXS:
            X[..., idx] *= 0.6
        # Systematic midpoint offset: futures trade at a different basis
        X[..., self._MIDPOINT_IDX] += 2.0
        return X, y

    def generate_binance_crypto_sim(self, n_sequences: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Binance spot crypto: 2.5× higher midpoint volatility, spread 1–3×
        wider than baseline, funding-rate pressure on imbalance."""
        X, y = self._generate_kaggle_base_sequences(n_sequences)
        # Higher volatility: amplify midpoint random walk
        for b in range(X.shape[0]):
            diffs = np.diff(X[b, :, self._MIDPOINT_IDX])
            X[b, 1:, self._MIDPOINT_IDX] = (
                X[b, 0, self._MIDPOINT_IDX] + np.cumsum(diffs * 2.5)).astype(np.float32)
        # Wider spread (no dedicated market maker)
        spread_mult = self.rng.uniform(1.0, 3.0, size=X.shape[:2]).astype(np.float32)
        X[..., self._SPREAD_IDX] *= spread_mult
        # Funding rate: slow drift on imbalance
        funding = np.cumsum(
            self.rng.normal(0, 0.01, size=X.shape[:2]), axis=1).astype(np.float32)
        X[..., self._IMBALANCE_IDX] += funding
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

    # All scenarios use the Kaggle 33-feature normalized space so the model receives
    # inputs with the same semantic meaning as its training data.  The in-distribution
    # baseline uses the same generator as the OOD scenarios (Kaggle space), not the
    # LOBSTER 49-feature generator — the two schemas are incompatible after truncation.
    scenarios = [
        ("In-Distribution (synthetic)", gen._generate_kaggle_base_sequences),
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
        # All generators now produce 33-feature Kaggle-space sequences.
        # Truncate or pad defensively in case d_input differs from the expected 33.
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
