import torch
import torch.nn as nn
from typing import List

def compute_integrated_gradients(
    model: nn.Module,
    sequence: torch.Tensor,
    adjacency: torch.Tensor,
    steps: int = 50
) -> torch.Tensor:
    """
    Computes Integrated Gradients from scratch via straight-line Riemann summation.
    IG_i = (x_i - x'_i) * integral_0^1 (d f / d x_i)(x' + alpha*(x-x')) d_alpha
    
    Parameters
    ----------
    model : nn.Module
        TENGNNModel or similar returning (batch, 2) logits.
    sequence : torch.Tensor
        Sequence data shape (1, n_assets, seq_len, d_input).
    adjacency : torch.Tensor
        Adjacency matrix.
    steps : int
        Interpolation steps.
        
    Returns
    -------
    torch.Tensor
        The computed Integrated Gradients of shape (1, n_assets, seq_len, d_input).
    """
    model.eval()
    baseline = torch.zeros_like(sequence)
    
    alphas = torch.linspace(0.0, 1.0, steps=steps).to(sequence.device)
    
    gradients = []
    
    for alpha in alphas:
        x_step = baseline + alpha * (sequence - baseline)
        x_step = x_step.clone().detach().requires_grad_(True)
        
        model.zero_grad()
        
        logits = model(x_step, adjacency)
        probs = torch.softmax(logits, dim=-1)
        
        # Spoofing class score
        score = probs[0, 1]
        
        score.backward()
        gradients.append(x_step.grad.clone().detach())
        
    grad_tensor = torch.stack(gradients, dim=0)
    
    # integral ~ mean across steps
    integral = torch.mean(grad_tensor, dim=0)
    
    ig = (sequence - baseline).detach() * integral
    return ig

def get_confidence_trace(
    model: nn.Module,
    sequence: torch.Tensor,
    adjacency: torch.Tensor
) -> List[float]:
    """
    Extracts the timestep-by-timestep progressive confidence traces
    replicating causal masking.
    
    Parameters
    ----------
    model : nn.Module
        TENGNNModel instance.
    sequence : torch.Tensor
        Standard input shape (1, n_assets, seq_len, d_input).
    adjacency : torch.Tensor
        Graph connectivity.
        
    Returns
    -------
    List[float]
        The detection confidence probabilities progressively at each event step.
    """
    model.eval()
    trace = []
    B, N_assets, seq_len, d_input = sequence.shape
    
    with torch.no_grad():
        for t in range(1, seq_len + 1):
            # Causal mask trick: we zero out the future events
            x_masked = sequence.clone()
            if t < seq_len:
                x_masked[:, :, t:, :] = 0.0
                
            logits = model(x_masked, adjacency)
            probs = torch.softmax(logits, dim=-1)
            spoof_prob = float(probs[0, 1].item())
            trace.append(spoof_prob)
            
    return trace
