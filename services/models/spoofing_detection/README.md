# Spoofing Detection

This module detects potential limit-order-book spoofing patterns using a Temporal Encoder Network (TEN) and an optional multi-asset graph layer (TEN-GNN).

The current implementation is useful for research, smoke testing, synthetic spoofing experiments, Kaggle pseudo-label training, and deployment export experiments. It is not yet a production HFT enforcement system.

## Current Status

| Area | Status |
| :--- | :--- |
| Synthetic data smoke pipeline | Working |
| TEN model shape/training smoke tests | Working |
| Kaggle adapter | Working |
| Kaggle training script | Available |
| TorchScript export | Working, with trace fallback |
| Dynamic INT8 quantization | Working for deployment-safe linear blocks |
| Real ground-truth spoofing validation | Not available |
| Sub-millisecond HFT deployment | Not implemented |

Important: Kaggle crypto data does not include confirmed spoofing labels. The loader creates heuristic pseudo-labels from cancel ratio and imbalance behavior. Treat Kaggle results as research signals, not regulatory-grade labels.

## Run Location

The `src` package is local to this module. Run module commands from:

```bash
cd /home/nourmaatouk/Solana-AI-Trading-System/services/models/spoofing_detection
```

If you run `python -m src...` from the repository root, Python will not find `src` unless you set `PYTHONPATH`.

## Quick Verification

From `services/models/spoofing_detection`:

```bash
python -m src.data.smoke_test
python -m src.models.ten
python -m src.training.trainer
```

From the repository root:

```bash
python -m pytest services/models/spoofing_detection/tests
```

Latest verified result:

```text
6 passed
```

## Deployment Export

From `services/models/spoofing_detection`:

```bash
python -m src.deployment.inference_engine
```

Output:

```text
checkpoints/ten_gnn_scripted.pt
```

The exporter tries `torch.jit.script` first and falls back to `torch.jit.trace` when the local PyTorch runtime cannot script the full model. Benchmark the printed latency on your machine; TorchScript is not guaranteed to be faster for this architecture.

Run deployment-safe INT8 quantization:

```bash
python -m src.deployment.quantize
```

Output:

```text
checkpoints/ten_gnn_int8.pt
```

The quantizer keeps the Transformer encoder in FP32 and dynamically quantizes only stable linear blocks around the temporal encoder, graph encoder, and classifier head. This avoids the PyTorch Transformer runtime failure caused by quantizing Transformer internals directly.

## Architecture

```text
LOB events / Kaggle LOB rows
        |
        v
Feature extraction and windowing
        |
        v
TEN temporal encoder
        |
        v
Optional cross-asset GNN encoder
        |
        v
Spoofing / legitimate logits
```

Core components:

| File | Purpose |
| :--- | :--- |
| `src/data/smoke_test.py` | Synthetic data pipeline smoke test |
| `src/data/lobster_loader.py` | Synthetic LOBSTER-like event generation/loading |
| `src/data/feature_engineering.py` | LOB feature extraction |
| `src/data/labeling.py` | Temporal labeling/window creation |
| `src/data/adversarial_injection.py` | Synthetic spoofing/layering sequence generation |
| `src/data/kaggle_loader.py` | Kaggle crypto LOB loading and pseudo-labeling |
| `src/models/ten.py` | Single-asset Temporal Encoder Network |
| `src/models/gnn.py` | Graph Attention Network |
| `src/models/ten_gnn.py` | TEN + GNN hybrid model |
| `src/training/trainer.py` | Training loop and smoke test |
| `src/deployment/inference_engine.py` | TorchScript/trace export and latency benchmark |
| `src/deployment/quantize.py` | Deployment-safe dynamic INT8 quantization |

## Feature Dimensions

Current code paths use different feature dimensions depending on the experiment:

| Path | Feature Count | Notes |
| :--- | :--- | :--- |
| Synthetic smoke pipeline | 47 | Verified by `src.data.smoke_test` |
| TEN trainer dummy smoke | 49 | Uses generated dummy training tensors |
| TEN-GNN deployment default | 49 | Used by export/quantization scripts |
| Kaggle loader with 15 levels | 33 | `2 + 15*2 + 1`: midpoint, spread, bid/ask volumes, imbalance |

This is acceptable for isolated experiments because model constructors accept `d_input`, but production training/export should use one explicit feature contract end to end.

## Synthetic Benchmark Snapshot

These metrics come from synthetic or simulated evaluation reports. They are useful for comparing architecture variants, not for claiming live-market performance.

| Model | Precision | Recall | F1 | Data Source |
| :--- | :--- | :--- | :--- | :--- |
| Rule-Based | 1.000 | 0.214 | 0.353 | Synthetic |
| Logistic Regression | 1.000 | 0.143 | 0.250 | Synthetic |
| Random Forest | 1.000 | 0.500 | 0.667 | Synthetic |
| LSTM-Attention | 0.610 | 0.700 | 0.650 | Synthetic |
| TEN | 0.860 | 0.900 | 0.880 | Synthetic |
| TEN-GNN | 0.930 | 0.950 | 0.940 | Synthetic |

OOD numbers in this module are simulation/proxy results, not exchange-certified real spoofing validation.

## Kaggle Training

Quick training example:

```bash
cd /home/nourmaatouk/Solana-AI-Trading-System/services/models/spoofing_detection
python train_kaggle_full.py --asset BTC --frequency 1min --max-rows 5000 --epochs 5
```

Medium run:

```bash
python train_kaggle_full.py --asset BTC --frequency 1min --max-rows 50000 --epochs 30
```

Full run:

```bash
python train_kaggle_full.py --asset BTC --frequency 1min --epochs 50
```

Expected outputs:

```text
checkpoints/kaggle_btc_1min/best_model.pt
checkpoints/kaggle_btc_1min/training_log_kaggle_btc_1min.csv
```

Because Kaggle labels are heuristic pseudo-labels, validate carefully before using a checkpoint downstream.

## Known Limitations

1. Synthetic benchmarks do not prove real exchange performance.
2. Kaggle labels are pseudo-labels, not confirmed spoofing ground truth.
3. Feature dimensions are not yet unified across all code paths.
4. TorchScript and INT8 speedups are environment-dependent.
5. The current CPU path is not suitable for sub-millisecond HFT controls.
6. Production deployment would need a stable live LOB feed, strict feature contract, monitoring, calibration, and human review of alert quality.

## Recommended Next Steps

1. Unify the feature contract across synthetic, Kaggle, TEN, TEN-GNN, and deployment.
2. Add a real validation report for Kaggle pseudo-label runs using held-out dates/assets.
3. Add a small HTML or Markdown report for spoofing model outputs, similar to the classification/NLP dashboards.
4. Move production-grade alerting behind calibrated thresholds instead of raw class predictions.
5. Replace deprecated PyTorch quantization APIs with the newer `torchao` path when the project updates PyTorch tooling.
