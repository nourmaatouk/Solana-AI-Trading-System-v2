import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
import shap

logger = logging.getLogger(__name__)

class TENGNNWrapper(nn.Module):
    """
    Wraps TEN-GNN to accept a single input tensor so that SHAP can compute
    gradients cleanly while holding the adjacency matrix constant.
    """
    def __init__(self, model: nn.Module, adj: torch.Tensor):
        super().__init__()
        self.model = model
        self.adj = adj
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x, self.adj)
        # return probability of spoofing (class 1)
        # DeepExplainer/GradientExplainer often expect a specific output format
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1].unsqueeze(-1)


def explain_single_alert(
    model: nn.Module,
    sequence: torch.Tensor,
    adjacency: torch.Tensor,
    feature_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Computes SHAP feature importance for a single spoofing alert.
    
    Parameters
    ----------
    model : nn.Module
        The TENGNNModel instance.
    sequence : torch.Tensor
        The sequence tensor of shape (1, n_assets, seq_len, d_input).
    adjacency : torch.Tensor
        The (n_assets, n_assets) detached adjacency tensor.
    feature_names : List[str]
        List of feature names (length equal to d_input).
        
    Returns
    -------
    dict
        A mapping from feature_name -> shap_value for this specific sequence.
    """
    # Fallback to generic names if none provided
    d_input = sequence.size(-1)
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(d_input)]
        
    wrapper = TENGNNWrapper(model, adjacency).eval().to(sequence.device)
    
    # GradientExplainer uses a background distribution. For a single alert,
    # we can use a zero-baseline or a small random noise baseline (requires >1 sample)
    background = torch.zeros_like(sequence)
    # We add sequence itself as background with low weight to satisfy shap's shape reqs easily
    explainer = shap.GradientExplainer(wrapper, background)
    
    # We must require grad on the sequence for GradientExplainer
    seq_input = sequence.clone().detach().requires_grad_(True)
    
    # Shap values returned as list of arrays per output or a single array
    # output shape (1, n_assets, seq_len, d_input)
    shap_values = explainer.shap_values(seq_input)
    
    # If shap returned a list, grab the first element for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    # mean absolute SHAP value per feature over the sequence and assets (for target asset or overall)
    # shap_values is a numpy array (1, n_assets, seq_len, d_input)
    shap_abs = np.abs(shap_values).mean(axis=(0, 1, 2))  # shape: (d_input,)
    
    importance_dict = {name: float(val) for name, val in zipping(feature_names, shap_abs)}
    return importance_dict

def zipping(l1, l2):
    return zip(l1, l2)

def get_global_importance(
    model: nn.Module,
    sequences: torch.Tensor,
    adjacency: torch.Tensor,
    out_path: Path
) -> np.ndarray:
    """
    Computes and aggregates SHAP values for a given batch of sequences.
    
    Logs the top features and saves the values to out_path.
    """
    logger.info("Computing global feature importance using SHAP...")
    
    wrapper = TENGNNWrapper(model, adjacency).eval().to(sequences.device)
    
    # Use zero tensors as a background baseline
    background = torch.zeros_like(sequences[:min(10, sequences.size(0))])
    explainer = shap.GradientExplainer(wrapper, background)
    
    # Requires gradient
    x_input = sequences.clone().detach().requires_grad_(True)
    shap_values = explainer.shap_values(x_input)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    # Aggregate over batch, target, seq_len -> (d_input,)
    global_shap = np.abs(shap_values).mean(axis=(0, 1, 2))
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), global_shap)
    logger.info(f"Saved global SHAP values to {out_path}")
    
    return global_shap
