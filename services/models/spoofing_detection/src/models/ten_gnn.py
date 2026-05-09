"""
ten_gnn.py — TEN-GNN Hybrid Fusion
===================================

Combines single-asset temporal reasoning (Transformer-Encoder Network) with
multi-asset relational reasoning (Graph Attention Network).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .gnn import GNNEncoder
from .ten import TENSpoofingDetector

logger = logging.getLogger(__name__)


class TENGNNModel(nn.Module):
    """Transformer-Encoder + Graph Neural Network Hybrid Fusion.

    Parameters
    ----------
    n_assets : int
        Number of correlated assets being monitored.
    d_input : int, default 47
        Number of raw features per timestep.
    d_model : int, default 256
        Output dimension of the TEN temporal embeddings.
    n_heads : int, default 8
        Number of attention heads for TEN.
    n_layers : int, default 6
        Number of transformer layers for TEN.
    d_ff : int, default 1024
        Feed-forward dimension for TEN.
    dropout : float, default 0.1
        Dropout probability.
    """

    def __init__(
        self,
        n_assets: int,
        d_input: int = 49,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_input = d_input
        self.n_assets = n_assets
        self.d_model = d_model
        # We assume GNN hidden dim is 128 as specified
        self.d_model_gnn = 128
        self.n_classes = 2
        
        # ── 1. Temporal Encoder (Shared across assets) ──────────────────────
        self.ten_encoder = TENSpoofingDetector(
            d_input=d_input,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            n_classes=self.n_classes
        )
        
        # ── 2. Spatial Graph Encoder ─────────────────────────────────────────
        self.gnn_encoder = GNNEncoder(
            in_dim=d_model,
            hidden_dim=self.d_model_gnn,
            n_layers=3, 
            n_heads=4,
            dropout=dropout
        )
        
        # ── 3. Hybrid Classifier Head ────────────────────────────────────────
        # Concatenates TEN (d_model) + GNN (d_model_gnn)
        fusion_dim = d_model + self.d_model_gnn
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_classes)
        )
        
        logger.info(
            "TENGNNModel initialized: %d assets, fusion_dim=%d",
            n_assets, fusion_dim
        )
        
    def forward(
        self,
        sequences: torch.Tensor,
        adjacency: torch.Tensor,
        target_asset_idx: int = 0,
        delta_t: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sequences : torch.Tensor shape (batch, n_assets, seq_len, d_input)
            LOB feature sequences for all assets in the graph.
        adjacency : torch.Tensor shape (n_assets, n_assets) or (batch, n_assets, n_assets)
            The explicit detached Hawkes adjacency matrix.
        target_asset_idx : int, default 0
            The index of the asset we are classifying for spoofing.

        Returns
        -------
        logits : torch.Tensor shape (batch, 2)
            Target asset spoofing logits.
        """
        # map variables to old logic names
        x = sequences
        adj = adjacency
        
        B, N, T, D = x.shape
        assert N == self.n_assets, f"Expected {self.n_assets} assets, got {N}"
        
        # Filter masks
        if delta_t is not None:
            delta_t = delta_t.reshape(B * N, T)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.reshape(B * N, T)
            
        # ── 1. Temporal Encoding ─────────────────────────────────────────────
        # Flatten batch and asset dims to reuse the shared TEN encoder efficiently
        x_flat = x.reshape(B * N, T, D)
        
        # Extract features (B*N, d_model)
        ten_embeddings_flat = self.ten_encoder.extract_features(
            x=x_flat,
            delta_t=delta_t,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Reshape back to (B, N, d_model)
        ten_embeddings = ten_embeddings_flat.reshape(B, N, self.d_model)
        
        # ── 2. Spatial Encoding ──────────────────────────────────────────────
        gnn_embeddings = self.gnn_encoder(ten_embeddings, adj)  # (B, N, 128)
        
        # ── 3. Hybrid Fusion ─────────────────────────────────────────────────
        # Extract the node embeddings specifically for the target asset
        target_ten = ten_embeddings[:, target_asset_idx, :]  # (B, 256)
        target_gnn = gnn_embeddings[:, target_asset_idx, :]  # (B, 128)
        
        fusion = torch.cat([target_ten, target_gnn], dim=-1) # (B, 384)
        
        # ── 4. Classification ────────────────────────────────────────────────
        logits = self.classifier(fusion)  # (B, 2)
        
        return logits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("TEN-GNN HYBRID FUSION — SHAPE VERIFICATION")
    print("=" * 70)
    
    B, N_assets, T, D_input = 4, 5, 100, 49
    
    print(f"Instantiating TENGNNModel (Assets={N_assets}, D={D_input})")
    model = TENGNNModel(n_assets=N_assets, d_input=D_input)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    
    # Dummy tensors
    x_dummy = torch.randn(B, N_assets, T, D_input)
    # Using index 43 as delta_t mimicking real data
    x_dummy[:, :, :, 43] = torch.abs(torch.randn(B, N_assets, T)) * 10
    
    adj_dummy = torch.rand(N_assets, N_assets)
    
    print(f"\nInputs:")
    print(f"  x:   {x_dummy.shape}")
    print(f"  adj: {adj_dummy.shape}")
    
    logits = model(x_dummy, adj_dummy, target_asset_idx=0)
    
    print(f"\nOutputs:")
    print(f"  Logits: {logits.shape} → {logits[0].tolist()}")
    
    assert logits.shape == (B, 2), "Output shape mismatch!"
    
    print("\n" + "=" * 70)
    print("TEN-GNN HYBRID SHAPE CHECKS PASSED ✓")
    print("=" * 70)
