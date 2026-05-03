"""
gnn.py — Graph Attention Network Component
==========================================

Implements the cross-asset topological feature extraction using Graph 
Attention Networks (GAT). This module consumes the embeddings produced 
by the TEN encoders out of individual LOB streams, along with the 
directional Hawkes Process adjacency matrix, to surface coordinated
spoofing behavior across multiple assets.

IMPORTANT DECOUPLING NOTE:
    The adjacency matrix input passed to `GNNEncoder.forward()` must be a 
    standalone tensor detached from the Hawkes parameter optimization loop.
    We do NOT backpropagate through the Hawkes estimation process.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GATLayer(nn.Module):
    """Single Graph Attention Network layer.
    
    Computes attention weights masked by the explicitly provided directional 
    adjacency matrix, then aggregates neighbor features.
    
    Formula:
        e_ij = LeakyReLU(a^T [W h_i || W h_j]) * A_ij
        α_ij = softmax_j(e_ij)
        h_i' = ELU(sum_j α_ij * W_v h_j)

    Parameters
    ----------
    in_dim : int
        Input feature dimension per node.
    out_dim : int
        Output feature dimension per attention head.
    n_heads : int
        Number of attention heads.
    dropout : float, default 0.1
        Dropout probability.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        
        # W transformation matrices (one per head)
        # Using nn.Linear to handle batching natively
        self.W = nn.Linear(in_dim, n_heads * out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, n_heads * out_dim, bias=False)
        
        # Attention parameter 'a' (one per head)
        # Dimensions: 2 * out_dim
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * out_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor shape (batch_size, n_assets, in_dim)
            Node embeddings from the previous layer or TEN encoders.
        adj : torch.Tensor shape (batch_size, n_assets, n_assets) or (n_assets, n_assets)
            The explicit adjacency matrix (e.g., from Hawkes Process estimator),
            acting as a structural mask and weight bias. Must be detached!

        Returns
        -------
        output : torch.Tensor shape (batch_size, n_assets, n_heads * out_dim)
        """
        B, N, _ = h.size()
        
        # Linear projection
        # h_proj: (B, N, n_heads * out_dim) -> (B, N, n_heads, out_dim)
        h_proj = self.W(h).view(B, N, self.n_heads, self.out_dim)
        v_proj = self.W_v(h).view(B, N, self.n_heads, self.out_dim)
        
        # Compute attention coefficients a^T [Wh_i || Wh_j]
        # h_proj_i: (B, N, 1, n_heads, out_dim) -> expand to (B, N, N, n_heads, out_dim)
        h_proj_i = h_proj.unsqueeze(2).expand(B, N, N, self.n_heads, self.out_dim)
        h_proj_j = h_proj.unsqueeze(1).expand(B, N, N, self.n_heads, self.out_dim)
        
        # a_input: (B, N, N, n_heads, 2 * out_dim)
        a_input = torch.cat([h_proj_i, h_proj_j], dim=-1)
        
        # Element-wise multiplication with attention vector 'a', then sum over last dim
        # a is (n_heads, 2 * out_dim). Reshape for broadcasting -> (1, 1, 1, n_heads, 2 * out_dim)
        e = self.leaky_relu((a_input * self.a.view(1, 1, 1, self.n_heads, -1)).sum(dim=-1))
        # e shape: (B, N, N, n_heads)

        # Apply structural mask and weight from adjacency matrix
        # adj shape is (B, N, N) or (N, N). We need to align it to (..., N, N, 1)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, N, N)
            
        adj = adj.unsqueeze(-1)  # (B, N, N, 1)

        # We multiply by A_ij, meaning if A_ij == 0, e_ij becomes large negative for softmax
        # We ensure zero connections result in 0 attention weight by masking before softmax
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e * adj, zero_vec)
        
        # Softmax over neighborhood (j dimension)
        attention = F.softmax(attention, dim=2)
        attention = self.dropout(attention) # (B, N, N, n_heads)
        
        # Aggregate features
        # v_proj: (B, N, n_heads, out_dim) -> reshape to match attention
        v_proj = v_proj.unsqueeze(1).expand(B, N, N, self.n_heads, self.out_dim)
        
        # Multiply attention by values and sum over j
        # attention: (B, N, N, n_heads) -> (B, N, N, n_heads, 1)
        h_prime = (attention.unsqueeze(-1) * v_proj).sum(dim=2)  # (B, N, n_heads, out_dim)
        
        # Concatenate heads -> (B, N, n_heads * out_dim)
        h_prime = h_prime.view(B, N, self.n_heads * self.out_dim)
        
        return self.elu(h_prime)


class GNNEncoder(nn.Module):
    """Multi-layer Graph Attention Network for modeling asset interaction topology.
    
    Parameters
    ----------
    in_dim : int, default 256
        Output dimension from the TEN temporal encoder.
    hidden_dim : int, default 128
        Dimension of the final graph embeddings.
    n_layers : int, default 3
        Number of GAT layers.
    n_heads : int, default 4
        Number of parallel attention heads per layer.
    dropout : float, default 0.1
        Dropout probability.
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, 
                 n_layers: int = 3, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Assumes intermediate layers preserve dimension using n_heads
        # with internal dim = hidden_dim // n_heads, but for simplicity
        # we configure head_dim so n_heads * head_dim = hidden_dim natively.
        head_dim = hidden_dim // n_heads
        assert head_dim * n_heads == hidden_dim, "hidden_dim must be divisible by n_heads"
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GATLayer(in_dim, head_dim, n_heads, dropout))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(GATLayer(hidden_dim, head_dim, n_heads, dropout))
            
        # Final layer (optionally averages heads, but here we concatenate to match hidden_dim)
        if n_layers > 1:
            self.layers.append(GATLayer(hidden_dim, head_dim, n_heads, dropout))

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        node_features : torch.Tensor shape (batch, n_assets, 256)
            Per-asset embeddings generated by TEN.
        adj : torch.Tensor shape (n_assets, n_assets) or (batch, n_assets, n_assets)
            Hawkes Process direction adjacency matrix. **Must be statically detached
            from gradient computation graphs.**

        Returns
        -------
        output : torch.Tensor shape (batch, n_assets, hidden_dim)
            Spatially aware graph embeddings capturing cross-asset dynamics.
        """
        h = node_features
        for layer in self.layers:
            h = layer(h, adj)
        return h


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("GRAPH ATTENTION NETWORK (GNN) — SHAPE VERIFICATION")
    print("=" * 70)
    
    batch_size = 8
    n_assets = 5
    in_dim = 256
    hidden_dim = 128
    
    print(f"Instantiating GNNEncoder (in_dim={in_dim}, hidden_dim={hidden_dim}, 3 layers, 4 heads)")
    model = GNNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, n_layers=3, n_heads=4)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} trainable")
    
    # Dummy node features (from TEN)
    h_in = torch.randn(batch_size, n_assets, in_dim)
    
    # Dummy asymmetric adjacency matrix (from Hawkes process)
    # Ensuring values are in [0, 1] and diagonal is 1
    A = torch.rand(n_assets, n_assets)
    A = (A + torch.eye(n_assets)).clip(0, 1)
    # Ensure some zeros to verify masking
    A[A < 0.3] = 0.0
    
    print(f"\nInputs:")
    print(f"  Node Features: {h_in.shape}")
    print(f"  Adjacency:     {A.shape} (detached, requires_grad=False)")
    
    assert not A.requires_grad, "Adjacency matrix must be detached!"
    
    h_out = model(h_in, A)
    
    print(f"\nOutputs:")
    print(f"  GNN Embedding: {h_out.shape}")
    
    # Assertions
    assert h_out.shape == (batch_size, n_assets, hidden_dim), "Output shape mismatch!"
    
    print("\n" + "=" * 70)
    print("GNN ENCODER SHAPE CHECKS PASSED ✓")
    print("=" * 70)
