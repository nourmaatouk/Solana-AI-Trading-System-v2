import torch
import torch.nn as nn
import logging
import time
import numpy as np
import copy
from pathlib import Path
from src.models.ten_gnn import TENGNNModel
from src.evaluation.metrics import compute_point_metrics

logger = logging.getLogger(__name__)

def evaluate_model(model, data_loader, adj_tensor, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X, adj_tensor.to(device))
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    latency = (time.perf_counter() - t0) * 1000 / max(len(all_labels), 1)

    metrics = compute_point_metrics(np.asarray(all_labels), np.asarray(all_preds))
    return metrics, latency


def quantize_deployment_safe_modules(model: TENGNNModel) -> TENGNNModel:
    """Apply dynamic INT8 quantization only where PyTorch supports it safely.

    Quantizing ``nn.TransformerEncoder`` internals currently breaks the
    transformer fast path in this environment. The deployment-safe path keeps
    the temporal transformer in FP32 and quantizes the surrounding linear
    projections, graph encoder linears, and final classifier head.
    """
    quantized_model = copy.deepcopy(model).cpu().eval()

    quantized_model.ten_encoder.input_projection = torch.ao.quantization.quantize_dynamic(
        quantized_model.ten_encoder.input_projection,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized_model.ten_encoder.classifier = torch.ao.quantization.quantize_dynamic(
        quantized_model.ten_encoder.classifier,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized_model.gnn_encoder = torch.ao.quantization.quantize_dynamic(
        quantized_model.gnn_encoder,
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized_model.classifier = torch.ao.quantization.quantize_dynamic(
        quantized_model.classifier,
        {nn.Linear},
        dtype=torch.qint8,
    )

    return quantized_model

def run_quantization(checkpoint_path: Path, output_path: Path):
    from src.data.dataset import prepare_data, create_dataloaders
    
    device = torch.device("cpu")
    logger.info("Starting Dynamic Quantization (INT8)...")
    
    # 1. Load Original Model — reconstruct shape from saved config
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        n_assets = cfg.get("n_assets", 3)
        d_input  = cfg.get("d_input", 33)
        d_model  = cfg.get("d_model", 256)
        model = TENGNNModel(n_assets=n_assets, d_input=d_input, d_model=d_model).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        adj_stored = ckpt.get("adj", None)
    else:
        logger.warning("Checkpoint not found — using random weights (n_assets=3, d_input=33)")
        n_assets, d_input, adj_stored = 3, 33, None
        model = TENGNNModel(n_assets=n_assets, d_input=d_input).to(device)
    model.eval()
    
    # 2. Build the adjacency tensor from saved checkpoint or identity fallback
    adj_tensor = (
        torch.tensor(adj_stored, dtype=torch.float32)
        if adj_stored is not None
        else torch.eye(n_assets)
    )

    # 3. Prepare Validation Data using real multi-asset tensors from checkpoint
    # We generate a minimal synthetic loader shaped to match (B, N, T, D)
    from src.data.adversarial_injection import AdversarialInjector
    injector = AdversarialInjector(seed=0)
    X_syn, y_syn = injector.generate_synthetic_spoofing_sequences(
        n_sequences=500, window_length=100, n_features=d_input
    )
    # Expand to (B, N, T, D) by replicating across assets
    X_syn_gnn = np.stack([X_syn] * n_assets, axis=1)  # (500, N, 100, d_input)
    val_tensor_x = torch.tensor(X_syn_gnn, dtype=torch.float32)
    val_tensor_y = torch.tensor(y_syn, dtype=torch.long)
    from torch.utils.data import TensorDataset, DataLoader as DL
    val_loader = DL(TensorDataset(val_tensor_x, val_tensor_y), batch_size=32)

    # 4. Baseline Metrics
    logger.info("Measuring Baseline (FP32) performance...")
    metrics_fp32, lat_fp32 = evaluate_model(model, val_loader, adj_tensor, device)
    f1_fp32 = metrics_fp32["f1"]
    logger.info(f"FP32 F1: {f1_fp32:.4f}, Latency: {lat_fp32:.4f} ms/seq")

    # 5. Apply Quantization
    # Quantize only deployment-safe modules. Full Transformer dynamic
    # quantization is not stable in this PyTorch/runtime combination.
    logger.info("Applying torch.ao.quantization.quantize_dynamic...")
    quantized_model = quantize_deployment_safe_modules(model)

    # 6. Quantized Metrics
    logger.info("Measuring Quantized (INT8) performance...")
    metrics_int8, lat_int8 = evaluate_model(quantized_model, val_loader, adj_tensor, device)
    f1_int8 = metrics_int8["f1"]

    speedup = lat_fp32 / max(lat_int8, 1e-9)
    logger.info(f"INT8 F1: {f1_int8:.4f} (Drop: {f1_fp32 - f1_int8:.4f}), Latency: {lat_int8:.4f} ms/seq")
    logger.info(f"Quantization Speedup: {speedup:.2f}x")

    # 7. Safety Assert
    assert f1_int8 >= f1_fp32 - 0.02, f"F1 drop too high: {f1_fp32 - f1_int8:.4f}"

    # 8. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), output_path)
    logger.info(f"Saved INT8 model to {output_path}")

    print("\nQuantization Report:")
    print(f"| Setup | F1 | Latency (ms) | Speedup |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| CPU FP32 | {f1_fp32:.4f} | {lat_fp32:.4f} | 1.0x |")
    print(f"| CPU INT8 | {f1_int8:.4f} | {lat_int8:.4f} | {speedup:.2f}x |")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parent.parent.parent
    # Default: BTC+ETH+SOL 1min checkpoint
    checkpoint = root / "checkpoints" / "tengnn_btc_eth_sol_1min" / "best_model.pt"
    q_output = root / "checkpoints" / "ten_gnn_int8.pt"
    run_quantization(checkpoint, q_output)
