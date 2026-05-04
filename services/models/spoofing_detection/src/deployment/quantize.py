import torch
import torch.nn as nn
import logging
import time
from pathlib import Path
from src.models.ten_gnn import TENGNNModel
from src.evaluation.metrics import compute_point_metrics

logger = logging.getLogger(__name__)

def evaluate_model(model, data_loader, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []
    
    t0 = time.perf_counter()
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            # TENGNN expects (B, N, T, D), data_loader gives (B, T, D)
            # Mocking n_assets=5 as in benchmarks
            X_gnn = X.unsqueeze(1).expand(-1, 5, -1, -1)
            adj = torch.eye(5).to(device)
            
            logits = model(X_gnn, adj)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    latency = (time.perf_counter() - t0) * 1000 / len(all_labels)
    
    metrics = compute_point_metrics(all_labels, all_preds)
    return metrics, latency

def run_quantization(checkpoint_path: Path, output_path: Path):
    from src.data.dataset import prepare_data, create_dataloaders
    
    device = torch.device("cpu")
    logger.info("Starting Dynamic Quantization (INT8)...")
    
    # 1. Load Original Model
    model = TENGNNModel(n_assets=5).to(device)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 2. Prepare Validation Data
    data = prepare_data(use_real_data=False, n_sequences=500)
    loaders = create_dataloaders(data, batch_size=32)
    val_loader = loaders["val"]
    
    # 3. Baseline Metrics
    logger.info("Measuring Baseline (FP32) performance...")
    metrics_fp32, lat_fp32 = evaluate_model(model, val_loader, device)
    f1_fp32 = metrics_fp32["f1"]
    logger.info(f"FP32 F1: {f1_fp32:.4f}, Latency: {lat_fp32:.4f} ms/seq")
    
    # 4. Apply Quantization
    # We quantize the linear layers in the whole hybrid model for maximum gain
    logger.info("Applying torch.quantization.quantize_dynamic...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    # 5. Quantized Metrics
    logger.info("Measuring Quantized (INT8) performance...")
    metrics_int8, lat_int8 = evaluate_model(quantized_model, val_loader, device)
    f1_int8 = metrics_int8["f1"]
    
    speedup = lat_fp32 / lat_int8
    logger.info(f"INT8 F1: {f1_int8:.4f} (Drop: {f1_fp32 - f1_int8:.4f}), Latency: {lat_int8:.4f} ms/seq")
    logger.info(f"Quantization Speedup: {speedup:.2f}x")
    
    # 6. Safety Assert
    assert f1_int8 >= f1_fp32 - 0.02, f"F1 drop too high: {f1_fp32 - f1_int8:.4f}"
    
    # 7. Save
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
    checkpoint = root / "checkpoints" / "ten_gnn_latest.pt"
    q_output = root / "checkpoints" / "ten_gnn_int8.pt"
    run_quantization(checkpoint, q_output)
