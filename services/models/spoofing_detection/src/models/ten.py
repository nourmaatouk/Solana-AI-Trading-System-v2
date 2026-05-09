"""
ten.py — Transformer Encoder Network for LOB Spoofing Detection
================================================================

Architecture (paper-aligned):
    Input:  (batch=32, seq_len=100, d_input=47)
      │
      ├─ Linear Embedding: 47 → d_model=256
      │
      ├─ Adaptive Temporal Positional Encoding:
      │     PE(pos, 2i) = sin(pos / (10000^(2i/d_model) × Δt_norm))
      │     PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model) × Δt_norm))
      │     Added as learnable-scaled residual: x + α · PE(x, Δt)
      │
      ├─ 6× Transformer Encoder Layers:
      │     ├─ Pre-LayerNorm
      │     ├─ Multi-Head Self-Attention (8 heads, d_k=32)
      │     ├─ Feed-Forward: 256 → 1024 → 256 (ReLU)
      │     └─ Dropout (p=0.1)
      │
      ├─ Mean Pooling: (batch, 100, 256) → (batch, 256)
      │
      └─ Classifier Head:
            Linear(256, 64) → ReLU → Linear(64, 2)

    Output: logits ∈ ℝ² (class 0 = legitimate, class 1 = spoofing)
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Adaptive Temporal Positional Encoding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AdaptiveTemporalPE(nn.Module):
    """Adaptive Temporal Positional Encoding modulated by Δt.

    Unlike standard sinusoidal PE that uses integer positions [0, 1, 2, ...],
    this encoding modulates the frequency by the actual inter-event time gap
    (delta_t), capturing the irregular timing of HFT events.

    Formula:
        PE(pos, 2i)   = sin(pos / (10000^(2i/d_model) × Δt_norm))
        PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model) × Δt_norm))

    Added as a learnable-scaled residual:
        output = x + α · PE(x, Δt)

    where α is a learnable scalar initialized to 1.0.

    Parameters
    ----------
    d_model : embedding dimension (256).
    max_len : maximum sequence length (default 512).
    dropout : dropout rate (default 0.1).
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # Learnable scale factor for the PE residual
        self.alpha = nn.Parameter(torch.ones(1))

        # Precompute the base divisor terms: 10000^(2i/d_model) for i=0..d_model/2-1
        # Shape: (1, 1, d_model//2)
        i = torch.arange(0, d_model, 2, dtype=torch.float32)
        div_term = torch.exp(i * (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor, delta_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply adaptive temporal positional encoding.

        Parameters
        ----------
        x : (batch, seq_len, d_model) — embedded input.
        delta_t : (batch, seq_len) — time-since-last-event in ms.
                  If None, falls back to standard positional encoding.

        Returns
        -------
        (batch, seq_len, d_model) — input + scaled PE.
        """
        batch_size, seq_len, d_model = x.shape

        # Position indices: (1, seq_len, 1)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        positions = positions.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

        if delta_t is not None:
            # Normalize delta_t: avoid division by zero, clamp to [0.01, ∞)
            # delta_t: (batch, seq_len) → (batch, seq_len, 1)
            dt_norm = delta_t.unsqueeze(-1).clamp(min=0.01)

            # Normalize to mean=1 scale to prevent extreme frequency modulation
            dt_mean = dt_norm.mean(dim=1, keepdim=True).clamp(min=1e-6)
            dt_normalized = dt_norm / dt_mean  # relative timing
            dt_normalized = dt_normalized.clamp(min=0.1, max=10.0)

            # Modulated divisor: div_term × dt_normalized
            # div_term: (1, 1, d_model//2), dt_normalized: (batch, seq_len, 1)
            modulated_div = self.div_term * dt_normalized  # (batch, seq_len, d_model//2)

            # Compute PE: positions / modulated_div
            # positions: (1, seq_len, 1), modulated_div: (batch, seq_len, d_model//2)
            angles = positions * modulated_div  # (batch, seq_len, d_model//2)
        else:
            # Fallback to standard sinusoidal PE (no delta_t modulation)
            angles = positions * self.div_term  # (1, seq_len, d_model//2)

        # Interleave sin and cos
        pe = torch.zeros(
            angles.shape[0], seq_len, d_model,
            device=x.device, dtype=x.dtype,
        )
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)

        # Broadcast if needed (standard PE case: batch dim = 1)
        if pe.shape[0] == 1 and batch_size > 1:
            pe = pe.expand(batch_size, -1, -1)

        # Learnable-scaled residual addition
        output = x + self.alpha * pe

        return self.dropout(output)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEN Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TENSpoofingDetector(nn.Module):
    """Transformer Encoder Network for LOB spoofing detection.

    Matches the paper specification:
        - d_model=256, n_heads=8 (d_k=32), d_ff=1024
        - 6 encoder layers with pre-LayerNorm
        - Adaptive Temporal PE modulated by delta_t
        - Mean pooling → 2-class classifier

    Parameters
    ----------
    d_input : input feature dimension (default 47).
    d_model : transformer embedding dimension (default 256).
    n_heads : number of attention heads (default 8).
    d_ff : feed-forward intermediate dimension (default 1024).
    n_layers : number of transformer encoder layers (default 6).
    dropout : dropout rate (default 0.1).
    max_len : maximum sequence length for PE (default 512).
    n_classes : output classes (default 2: legitimate vs spoofing).
    delta_t_feature_idx : index of the delta_t feature in the input
                          (default 43, matching feature_engineering.py).
    """

    def __init__(
        self,
        d_input: int = 49,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 512,
        n_classes: int = 2,
        delta_t_feature_idx: int = 43,
    ):
        super().__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.delta_t_feature_idx = delta_t_feature_idx

        # ── Input Embedding ──────────────────────────────────────────────────
        self.input_projection = nn.Linear(d_input, d_model)

        # ── Adaptive Temporal Positional Encoding ────────────────────────────
        self.positional_encoding = AdaptiveTemporalPE(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        # ── Transformer Encoder ──────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm (applied before attention and FFN)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,  # safer for variable-length inputs
        )

        # ── Final LayerNorm (post-encoder) ───────────────────────────────────
        self.final_norm = nn.LayerNorm(d_model)

        # ── Classification Head ──────────────────────────────────────────────
        # Mean-pool → Linear(256, 64) → ReLU → Linear(64, 2)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

        # Initialize weights
        self._init_weights()

        # Log architecture
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "TEN initialized: d_input=%d, d_model=%d, %d heads, %d layers, "
            "%d classes | %.2fM params (%.2fM trainable)",
            d_input, d_model, n_heads, n_layers, n_classes,
            n_params / 1e6, n_trainable / 1e6,
        )

    def _init_weights(self) -> None:
        """Xavier uniform initialization for linear layers."""
        for name, param in self.named_parameters():
            if param.dim() > 1:  # Weight matrices
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def extract_features(
        self,
        x: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process inputs through the encoder, returning the pooled embedding."""
        if delta_t is None and self.delta_t_feature_idx < x.shape[-1]:
            delta_t = x[:, :, self.delta_t_feature_idx]
            
        embedded = self.input_projection(x)
        encoded = self.positional_encoding(embedded, delta_t)
        transformer_out = self.transformer_encoder(encoded, src_key_padding_mask=src_key_padding_mask)
        transformer_out = self.final_norm(transformer_out)

        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()
            pooled = (transformer_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = transformer_out.mean(dim=1)
            
        return pooled

    def forward(
        self,
        x: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (batch, seq_len, d_input) — raw feature sequences.
        delta_t : (batch, seq_len) — time-since-last-event in ms.
        src_key_padding_mask : (batch, seq_len) — True for padded positions.

        Returns
        -------
        logits : (batch, n_classes) — raw class logits.
        """
        pooled = self.extract_features(x, delta_t, src_key_padding_mask)
        logits = self.classifier(pooled)  # (batch, n_classes)

        return logits

    def predict_proba(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return class probabilities (softmax over logits)."""
        logits = self.forward(x, **kwargs)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return predicted class indices."""
        logits = self.forward(x, **kwargs)
        return logits.argmax(dim=-1)

    @property
    @torch.jit.unused
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    @torch.jit.unused
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Smoke test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 70)
    print("TEN MODEL — SHAPE VERIFICATION (2 epochs on dummy data)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = TENSpoofingDetector(
        d_input=47, d_model=256, n_heads=8,
        d_ff=1024, n_layers=6, dropout=0.1, n_classes=2,
    ).to(device)

    print(f"Parameters: {model.num_parameters:,} ({model.num_trainable_parameters:,} trainable)")

    # ── Dummy data ───────────────────────────────────────────────────────────
    batch_size = 32
    seq_len = 100
    d_input = 47

    X_dummy = torch.randn(batch_size * 4, seq_len, d_input)  # 4 batches
    y_dummy = torch.randint(0, 2, (batch_size * 4,))

    # ── Training config ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    criterion = nn.CrossEntropyLoss()

    # ── Train 2 epochs ───────────────────────────────────────────────────────
    model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_dummy), batch_size):
            xb = X_dummy[i:i + batch_size].to(device)
            yb = y_dummy[i:i + batch_size].to(device)

            optimizer.zero_grad()
            logits = model(xb)

            # Validate shapes
            assert logits.shape == (len(xb), 2), f"Logits shape mismatch: {logits.shape}"

            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch + 1}/2 — loss: {avg_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")

    # ── Inference test ───────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, seq_len, d_input).to(device)
        logits = model(test_input)
        probs = model.predict_proba(test_input)
        pred = model.predict(test_input)

    print(f"\n  Inference shapes:")
    print(f"    Input:  {test_input.shape}")
    print(f"    Logits: {logits.shape}  → {logits.cpu().numpy()}")
    print(f"    Probs:  {probs.shape}   → {probs.cpu().numpy()}")
    print(f"    Pred:   {pred.shape}    → {pred.item()}")

    # ── Test with explicit delta_t ───────────────────────────────────────────
    with torch.no_grad():
        delta_t = torch.rand(1, seq_len).to(device) * 100  # ms
        logits_dt = model(test_input, delta_t=delta_t)
        print(f"    Logits (with Δt): {logits_dt.shape} → {logits_dt.cpu().numpy()}")

    # ── Test with padding mask ───────────────────────────────────────────────
    with torch.no_grad():
        mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        mask[0, 80:] = True  # last 20 positions are padding
        logits_masked = model(test_input, src_key_padding_mask=mask)
        print(f"    Logits (masked):  {logits_masked.shape} → {logits_masked.cpu().numpy()}")

    print("\n" + "=" * 70)
    print("ALL SHAPE CHECKS PASSED ✓")
    print("=" * 70)
