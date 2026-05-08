import time
import torch
import logging
from pathlib import Path
from typing import Tuple
from src.models.ten_gnn import TENGNNModel

logger = logging.getLogger(__name__)

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
    device = torch.device("cpu") # Benchmarking on CPU as requested
    
    # Load model
    model = TENGNNModel(n_assets=5).to(device)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Mock input
    input_tensor = torch.randn(1, 5, 100, 49).to(device)
    adj_tensor = torch.eye(5).to(device)
    
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
    checkpoint = root / "checkpoints" / "ten_gnn_latest.pt"
    export_path = root / "checkpoints" / "ten_gnn_scripted.pt"
    export_torchscript(checkpoint, export_path)
