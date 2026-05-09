# How to Run the LOB Spoofing Detection Pipeline

The core single-asset LOB spoofing detection logic (the data pipeline, TEN model, and trainer loop) is completely functional and can be tested immediately using built-in synthetic data.

Run the following commands from the spoofing module directory:

```bash
cd /home/nourmaatouk/Solana-AI-Trading-System/services/models/spoofing_detection
```

If you run these commands from the repository root, Python will not find the local `src` package unless you set `PYTHONPATH`.

### 1. Run the Data Pipeline Smoke Test
This validates the entire data pipeline. It generates 5,000 synthetic LOBSTER events, extracts 47 features, applies the L(t) temporal labels, and injects 100 synthetic spoofing sequences (layering and flipping).
```bash
python -m src.data.smoke_test
```

### 2. Verify the TEN Architecture
This verifies the Transformer-Encoder Network dimensions, including the Adaptive Temporal Positional Encoding, by running a 2-epoch test on dummy data.
```bash
python -m src.models.ten
```

### 3. Run the Trainer Smoke Test
This validates the training loop by running 2 epochs on dummy data. It verifies gradient clipping, class-weight balancing, learning rate scheduling (cosine annealing), checkpointing (`best_model.pt`), and CSV logging.
```bash
python -m src.training.trainer
```

---

## Moving to Production Data
Currently, these tests fall back to a dynamic generation of synthetic LOBSTER data structure so that everything runs out-of-the-box. 

Once you are ready to evaluate real data:
1. Place your actual LOBSTER CSV files inside `data/raw/`.
2. The pipeline will automatically route through `load_lobster_data()` strictly adapting to your actual orderbook data.

### 4. Deployment & Optimization
To prepare benchmark/export artifacts:

**TorchScript Export (Latency Benchmark)**
```bash
python3 -m src.deployment.inference_engine
```

Output:

```text
checkpoints/ten_gnn_scripted.pt
```

The exporter prefers `torch.jit.script` and falls back to `torch.jit.trace` if the runtime cannot script the model. Always compare the printed latency numbers on your machine; TorchScript is an export format here, not a guaranteed speedup.

**Dynamic Quantization (INT8 Optimization)**
```bash
python3 -m src.deployment.quantize
```

Output:

```text
checkpoints/ten_gnn_int8.pt
```

The quantizer keeps the Transformer encoder in FP32 and applies dynamic INT8 quantization only to deployment-safe linear blocks around the temporal encoder, graph encoder, and classifier head. This avoids the PyTorch Transformer runtime failure caused by quantizing Transformer internals directly.
