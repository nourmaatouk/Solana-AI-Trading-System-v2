# How to Run the LOB Spoofing Detection Pipeline

The core single-asset LOB spoofing detection logic (the data pipeline, TEN model, and trainer loop) is completely functional and can be tested immediately using built-in synthetic data.

Run the following commands from the project root (`/home/nourmaatouk/Solana-AI-Trading-System/services/models/spoofing_detection`):

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
To prepare the model for production HFT environments:

**TorchScript Export (Latency Benchmark)**
```bash
python3 -m src.deployment.inference_engine
```

**Dynamic Quantization (INT8 Optimization)**
```bash
python3 -m src.deployment.quantize
```
