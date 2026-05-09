"""
shap_explainer.py — SHAP-based feature attribution for TEN-GNN alerts
======================================================================

Provides:
    - ``TENGNNWrapper``: thin nn.Module wrapper so SHAP can treat adjacency
      as a constant and differentiate through the sequence tensor only.
    - ``explain_single_alert``: per-feature SHAP importance for one alert.
    - ``get_global_importance``: aggregate SHAP across a batch, saved to disk.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
import shap

logger = logging.getLogger(__name__)


class TENGNNWrapper(nn.Module):
    """Wraps TEN-GNN so SHAP differentiates through sequences, not adjacency."""

    def __init__(self, model: nn.Module, adj: torch.Tensor):
        super().__init__()
        self.model = model
        # Register as buffer so it moves with the module but is not a parameter
        self.register_buffer("adj", adj.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x, self.adj)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1].unsqueeze(-1)  # P(spoofing), shape (B, 1)


def explain_single_alert(
    model: nn.Module,
    sequence: torch.Tensor,
    adjacency: torch.Tensor,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute per-feature SHAP importance for a single spoofing alert.

    Parameters
    ----------
    model : TENGNNModel instance (eval mode).
    sequence : (1, n_assets, seq_len, d_input) tensor.
    adjacency : (n_assets, n_assets) tensor.
    feature_names : list of length d_input.

    Returns
    -------
    dict mapping feature_name → mean-absolute SHAP value.
    """
    d_input = sequence.size(-1)
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(d_input)]

    wrapper = TENGNNWrapper(model, adjacency).eval().to(sequence.device)

    # Zero-baseline background — shape matches sequence exactly
    background = torch.zeros_like(sequence)
    explainer = shap.GradientExplainer(wrapper, background)

    seq_input = sequence.clone().detach().requires_grad_(True)
    shap_values = explainer.shap_values(seq_input)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # shap_values: (1, n_assets, seq_len, d_input) — collapse all but last dim
    shap_abs = np.abs(shap_values).mean(axis=tuple(range(shap_values.ndim - 1)))  # (d_input,)

    return {name: float(val) for name, val in zip(feature_names, shap_abs)}


def get_global_importance(
    model: nn.Module,
    sequences: torch.Tensor,
    adjacency: torch.Tensor,
    out_path: Path,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Aggregate SHAP values across a batch and save to disk.

    Parameters
    ----------
    sequences : (B, n_assets, seq_len, d_input).
    out_path : .npy file path.

    Returns
    -------
    global_shap : (d_input,) array of mean absolute SHAP values.
    """
    logger.info("Computing global feature importance using SHAP (%d sequences)...", len(sequences))

    wrapper = TENGNNWrapper(model, adjacency).eval().to(sequences.device)

    n_background = min(10, sequences.size(0))
    background = torch.zeros_like(sequences[:n_background])
    explainer = shap.GradientExplainer(wrapper, background)

    x_input = sequences.clone().detach().requires_grad_(True)
    shap_values = explainer.shap_values(x_input)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Collapse batch, assets, seq_len → (d_input,)
    global_shap = np.abs(shap_values).mean(axis=tuple(range(shap_values.ndim - 1)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), global_shap)
    logger.info("Saved global SHAP values to %s", out_path)

    if feature_names is not None:
        top_idx = np.argsort(global_shap)[::-1][:10]
        logger.info("Top 10 features by SHAP:")
        for rank, idx in enumerate(top_idx, 1):
            logger.info("  %2d. %-30s %.4f", rank, feature_names[idx], global_shap[idx])

    return global_shap
