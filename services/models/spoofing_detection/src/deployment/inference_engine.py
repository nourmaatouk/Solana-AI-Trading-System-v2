"""
inference_engine.py — Production inference for TEN-GNN spoofing detection
=========================================================================

Provides:
    - ``SpoofingInferenceEngine``: load checkpoint, infer with optional live
      adjacency override from ``HawkesUpdater``.
    - ``export_torchscript``: benchmark and export to TorchScript for serving.
    - ``benchmark_inference``: latency measurement helper.

Typical usage
-------------
    engine = SpoofingInferenceEngine.from_checkpoint(
        "checkpoints/tengnn_btc_eth_sol_1min/best_model.pt"
    )
    engine.attach_hawkes_updater(updater)   # optional — feeds live adjacency

    probs = engine.infer(sequences_tensor)  # (B,) spoofing probabilities
"""

import time
import threading
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from src.models.ten_gnn import TENGNNModel

logger = logging.getLogger(__name__)


class SpoofingInferenceEngine:
    """Thread-safe inference wrapper around a trained TENGNNModel.

    Parameters
    ----------
    model : TENGNNModel
        The trained model (already moved to ``device`` and set to eval).
    adj_tensor : torch.Tensor
        Default adjacency matrix (n_assets × n_assets) stored in the checkpoint.
    device : torch.device
    """

    def __init__(
        self,
        model: TENGNNModel,
        adj_tensor: torch.Tensor,
        device: torch.device,
    ) -> None:
        self._model = model
        self._default_adj = adj_tensor.to(device)
        self._device = device
        self._hawkes_updater = None
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
    ) -> "SpoofingInferenceEngine":
        """Load a trained checkpoint and return a ready-to-use engine.

        Falls back to random weights if the checkpoint does not exist (useful
        for testing the pipeline without a trained model).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            cfg = ckpt.get("config", {})
            n_assets = cfg.get("n_assets", 3)
            d_input  = cfg.get("d_input", 33)
            d_model  = cfg.get("d_model", 256)
            n_heads  = cfg.get("n_heads", min(8, d_model // 32))
            d_ff     = cfg.get("d_ff", d_model * 4)
            sd = ckpt["model_state_dict"]
            n_layers = cfg.get("n_layers", max(
                int(k.split(".")[3]) + 1
                for k in sd if k.startswith("ten_encoder.transformer_encoder.layers.")
            ))
            model = TENGNNModel(
                n_assets=n_assets, d_input=d_input, d_model=d_model,
                n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
            )
            model.load_state_dict(sd)
            adj_np = ckpt.get("adj", np.eye(n_assets))
            logger.info(
                "Loaded checkpoint: n_assets=%d  d_input=%d  d_model=%d  n_layers=%d",
                n_assets, d_input, d_model, n_layers,
            )
        else:
            logger.warning("Checkpoint not found at %s — using random weights", checkpoint_path)
            n_assets, d_input, d_model, n_layers, d_ff = 3, 34, 128, 3, 512
            model = TENGNNModel(
                n_assets=n_assets, d_input=d_input, d_model=d_model,
                n_layers=n_layers, d_ff=d_ff,
            )
            adj_np = np.eye(n_assets)

        model = model.to(device).eval()
        adj_tensor = torch.tensor(adj_np, dtype=torch.float32)
        return cls(model, adj_tensor, device)

    def attach_hawkes_updater(self, updater) -> None:
        """Attach a ``HawkesUpdater`` so inference uses its live adjacency."""
        self._hawkes_updater = updater
        logger.info("HawkesUpdater attached — adjacency will be refreshed dynamically")

    def _current_adj(self) -> torch.Tensor:
        if self._hawkes_updater is not None:
            live = self._hawkes_updater.adjacency
            return torch.tensor(live, dtype=torch.float32, device=self._device)
        return self._default_adj

    def infer(self, sequences: torch.Tensor) -> np.ndarray:
        """Run inference and return spoofing probability for each window.

        Parameters
        ----------
        sequences : torch.Tensor of shape (B, N, T, D)
            B = batch size, N = n_assets, T = window_length, D = n_features.

        Returns
        -------
        np.ndarray of shape (B,) — P(spoofing) in [0, 1] for each window.
        """
        with self._lock:
            adj = self._current_adj()
            sequences = sequences.to(self._device)
            with torch.no_grad():
                logits = self._model(sequences, adj)
                probs = torch.softmax(logits, dim=-1)[:, 1]
            return probs.cpu().numpy()

    def predict(self, sequences: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions (0=legit, 1=spoofing)."""
        return (self.infer(sequences) >= threshold).astype(np.int64)

def benchmark_inference(model, input_tensor, adj_tensor, name="Model", iterations=100):
    model.eval()
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(input_tensor, adj_tensor)
            
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            model(input_tensor, adj_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    
    avg_latency = (t1 - t0) * 1000 / iterations
    logger.info(f"{name} Average Latency: {avg_latency:.4f} ms")
    return avg_latency


def compile_torchscript_model(
    model: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, torch.Tensor],
) -> torch.jit.ScriptModule:
    """Compile model for deployment, falling back to trace if scripting fails."""
    try:
        return torch.jit.script(model)
    except Exception as exc:
        logger.warning("torch.jit.script failed; falling back to torch.jit.trace: %s", exc)
        return torch.jit.trace(model, example_inputs, strict=False)

def export_torchscript(model_path: Path, output_path: Path):
    """Exports the TEN-GNN model to TorchScript."""
    device = torch.device("cpu")

    # Load checkpoint and reconstruct model with saved config
    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        n_assets = cfg.get("n_assets", 3)
        d_input  = cfg.get("d_input", 33)
        d_model  = cfg.get("d_model", 256)
        model = TENGNNModel(n_assets=n_assets, d_input=d_input, d_model=d_model).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        adj_np = ckpt.get("adj", None)
    else:
        logger.warning("Checkpoint not found at %s — using random weights (n_assets=3, d_input=33)", model_path)
        n_assets, d_input = 3, 33
        model = TENGNNModel(n_assets=n_assets, d_input=d_input).to(device)
        adj_np = None
    model.eval()

    adj_tensor = (
        torch.tensor(adj_np, dtype=torch.float32).to(device)
        if adj_np is not None
        else torch.eye(n_assets).to(device)
    )
    input_tensor = torch.randn(1, n_assets, 100, d_input).to(device)
    
    logger.info("Comparing Eager vs TorchScript...")
    eager_lat = benchmark_inference(model, input_tensor, adj_tensor, name="Eager Mode")
    
    # Compile. Script is preferred, trace fallback supports the fixed-shape
    # low-latency deployment path used by this benchmark command.
    scripted_model = compile_torchscript_model(model, (input_tensor, adj_tensor))
    scripted_lat = benchmark_inference(scripted_model, input_tensor, adj_tensor, name="TorchScript Mode")
    
    speedup = eager_lat / scripted_lat
    print(f"\nTorchScript speedup: {eager_lat:.2f}ms -> {scripted_lat:.2f}ms")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted_model.save(output_path)
    logger.info(f"Saved TorchScript model to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parent.parent.parent
    # Default: BTC+ETH+SOL 1min checkpoint — override with --checkpoint arg if needed
    checkpoint = root / "checkpoints" / "tengnn_btc_eth_sol_1min" / "best_model.pt"
    export_path = root / "checkpoints" / "ten_gnn_scripted.pt"
    export_torchscript(checkpoint, export_path)
