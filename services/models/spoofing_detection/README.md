# TEN-GNN Spoofing Detection

Detects limit-order-book (LOB) spoofing using a multi-asset **Temporal Encoder Network (TEN) + Graph Attention Network (GNN)** trained on real Kaggle high-frequency crypto data (BTC, ETH, SOL).

---

## Project objective

Build a machine-learning detector for LOB spoofing — the manipulation strategy where a trader places large visible orders to move prices, then cancels them before execution. The system learns cross-asset excitation patterns (BTC influences ETH, etc.) using a Hawkes process to build a dynamic adjacency matrix, feeds it into a GNN that aggregates per-asset transformer embeddings, and outputs a per-window spoofing probability.

**Honest scope statement:**
- Labels are heuristic pseudo-labels (cancel ratio + imbalance), not confirmed spoofing ground truth.
- The system is a research prototype, not a production HFT enforcement tool.
- Results are not directly comparable to the paper — the paper uses private exchange LOBSTER data with confirmed L(t) labels; this uses public Kaggle pseudo-labels.

---

## How the system works — step by step

### 1. Data enters the project

**Source:** Kaggle "High Frequency Crypto Limit Order Book Data" — three assets:
- `data/raw/kaggle_crypto/BTC_1min.csv` — real Coinbase BTC order book, 1-min bars
- `data/raw/kaggle_crypto/ETH_1min.csv` — real Coinbase ETH order book
- `data/raw/kaggle_crypto/SOL_1min.csv` — synthesized from 15-min OHLCV (cancel columns = 0)

Each CSV row represents one 1-minute bar with up to 15 depth levels: bid/ask limit notionals, cancel notionals, midpoint, spread, buys/sells counts, and a `system_time` wall-clock timestamp.

### 2. Data is cleaned and transformed (`src/data/kaggle_adapters.py`)

The Kaggle CSVs use 0-based depth columns (`bids_notional_0`). The adapter maps them to 1-based aliases (`bids_limit_notional_1`) so the feature extractor stays simple. It also parses timestamps and coerces numeric types.

### 3. Features are extracted (`src/data/kaggle_loader.py → extract_lob_features`)

For each row, 33 features are extracted:
```
2  base features:  midpoint, spread
30 depth features: bid_notional_1..15 and ask_notional_1..15 (one pair per level)
1  imbalance:      (total_bid - total_ask) / (total_bid + total_ask)
──────────────────────────────────────
33 total  →  n_features = 33 (N_FEATURES_KAGGLE in config.py)
```

After extraction, a `StandardScaler` is fit on the entire asset's data and applied — all 33 features are z-scored before windowing. This is critical because raw midpoint values span very different scales (BTC ~50,000, ETH ~2,000).

### 4. Labels are generated (`src/data/kaggle_loader.py → label_real_spoofing`)

Each row gets a spoofing score:
```
score = cancel_ratio + 0.5 × |order_imbalance|

cancel_ratio = cancel_vol / (cancel_vol + limit_vol)
```

A **rolling 500-row 70th-percentile threshold** is computed. A row is marked spoofing when its score exceeds the local rolling threshold. This keeps the positive rate stable (~30%) across time — a global threshold would concentrate positives in early high-volatility periods and leave the test split near-empty.

This is NOT the paper's L(t) labeling. L(t) requires: (1) V_cancel/V_placed > 90%, (2) order duration < 500ms, (3) mid-price impact > 1.5 ticks. The Kaggle data lacks order-level cancel tracking and sub-second timestamps, so these conditions cannot be implemented.

### 5. Sliding windows are created

A 100-event sliding window with 50% stride creates the final sequences:
- Window label = 1 if ≥40% of events in the window are spoofing
- Output shape: `(n_windows, 100, 33)`

### 6. Real Hawkes timestamps are parsed

The `system_time` column from the Kaggle CSV is parsed to wall-clock seconds. All assets are normalized relative to the earliest observation across all assets, converted to minutes. This gives Hawkes real temporal separation between assets (BTC ticks at :41s, ETH at :49s, SOL at :23s within each minute).

### 7. Hawkes adjacency is estimated (`src/models/hawkes.py`)

A multivariate Hawkes process with double-exponential kernel is fit via MLE (scipy L-BFGS-B) on the real timestamps:
```
λᵢ(t) = μᵢ + Σⱼ αᵢⱼ₁·exp(-β₁(t-tⱼ)) + αᵢⱼ₂·exp(-β₂(t-tⱼ))
```
- β₁=10, β₂=0.5 (fast/slow excitation windows: ~6s and ~2min in minute units)
- Self-excitation is zeroed (`fill_diagonal(alpha, 0)`) — only cross-asset influence
- Adjacency = normalized branching ratio A_ij = α_ij1/β₁ + α_ij2/β₂

Typical result with BTC+ETH+SOL:
```
BTC→ETH: 1.0  BTC→SOL: 0.43
ETH→BTC: 1.0  ETH→SOL: 0.00
SOL→BTC: 0.00 SOL→ETH: 1.0
```

### 8. The TEN encoder processes each asset (`src/models/ten.py`)

The TEN is a Transformer Encoder Network matching the paper spec:
- Input projection: d_input (33) → d_model (256)
- **Adaptive Temporal PE**: sinusoidal PE with frequency modulated by the actual inter-event time Δt (extracted from feature index 43 in the 49-feature LOBSTER schema; the Kaggle path uses a fixed fallback). Learnable scale α.
- 6× Transformer encoder layers: Pre-LayerNorm, 8 attention heads (d_k=32), FFN 256→1024→256, dropout=0.1
- Mean pooling → embedding of shape (batch, 256)

Note: the Kaggle path does not have a Δt feature (feature index 43 is out of range for d_input=33), so the TEN falls back to standard sinusoidal PE. The adaptive PE is only active on the 49-feature LOBSTER path.

### 9. The GNN aggregates cross-asset context (`src/models/gnn.py`, `src/models/ten_gnn.py`)

In `TENGNNModel`:
1. The shared TEN processes all N assets in one batch (B×N flattened to B·N, then reshaped)
2. TEN embeddings (B, N, 256) become node features in a 3-layer GAT
3. GAT hidden size = 128. Adjacency = Hawkes branching ratios from step 7
4. Target asset TEN embedding (256) + target asset GNN embedding (128) → concat (384)
5. Linear(384→2) → 2 class logits

### 10. Training (`train_tengnn_full.py`)

- AdamW optimizer, lr=5e-4, weight_decay=1e-4
- CosineAnnealingLR scheduler
- CrossEntropyLoss with inverse-frequency class weights (handles ~30/70 imbalance)
- Gradient clipping (global norm=1.0)
- Chronological 65/21/14 split (no shuffle on val/test — preserves time order)
- Early stopping on val F1 (patience=8)
- Best checkpoint saved to `checkpoints/tengnn_btc_eth_sol_1min/best_model.pt`

### 11. Inference (`src/deployment/inference_engine.py`)

`SpoofingInferenceEngine.from_checkpoint(path)` reads model config from the checkpoint's `config` dict (n_assets, d_input, d_model) and reconstructs the correct model shape. It returns P(spoofing) for each window.

### 12. Live Hawkes updating (`src/deployment/hawkes_updater.py`)

A daemon thread accumulates incoming event timestamps per asset (sliding window of 2000 events) and re-fits the Hawkes MLE every 500ms. The latest adjacency is exposed via a thread-safe `.adjacency` property. When attached to `SpoofingInferenceEngine`, every inference call reads the live adjacency instead of the frozen training-time estimate.

### 13. SHAP explainability (`src/explainability/shap_explainer.py`)

`GradientExplainer` with a zero baseline computes per-feature attribution scores. The adjacency is held constant (registered as a buffer in `TENGNNWrapper`). Returns mean absolute SHAP per feature, averaged over assets and sequence length.

### 14. Quantization/export (`src/deployment/quantize.py`, `src/deployment/inference_engine.py`)

- `export_torchscript`: reads checkpoint config, reconstructs model, tries `torch.jit.script`, falls back to `torch.jit.trace`
- `run_quantization`: applies `torch.ao.quantization.quantize_dynamic` to linear blocks only (projection, GAT linears, classifier). Transformer encoder stays FP32 (quantizing it breaks the fast path). **Warning:** `torch.ao.quantization` is deprecated in PyTorch 2.10 — migrate to `torchao` before the next PyTorch upgrade.

---

## Folder structure

```
spoofing_detection/
│
├── train_tengnn_full.py          ← main training script (multi-asset TEN-GNN)
├── train_kaggle_full.py          ← single-asset TEN-only training
│
├── data/raw/kaggle_crypto/       ← Kaggle LOB CSVs (BTC, ETH, SOL)
│
├── checkpoints/
│   ├── tengnn_btc_eth_sol_1min/
│   │   └── best_model.pt         ← current best checkpoint
│   ├── ten_gnn_int8.pt           ← quantized model
│   └── ten_gnn_scripted.pt       ← TorchScript export
│
├── src/
│   ├── config.py                 ← centralized constants (n_features, paths, hyperparams)
│   │
│   ├── data/
│   │   ├── kaggle_adapters.py    ← Kaggle CSV column normalization
│   │   ├── kaggle_loader.py      ← load → extract 33 features → label → window
│   │   ├── dataset.py            ← SpoofingDataset, chronological split, DataLoaders
│   │   ├── adversarial_injection.py  ← synthetic spoofing sequences (smoke testing)
│   │   ├── feature_engineering.py   ← LOBFeatureEngineer (LOBSTER path, 49 features)
│   │   ├── labeling.py           ← L(t) labeler (LOBSTER path, not used in Kaggle training)
│   │   ├── lobster_loader.py     ← synthetic LOBSTER data generator (smoke testing)
│   │   └── smoke_test.py         ← end-to-end pipeline smoke test (LOBSTER path)
│   │
│   ├── models/
│   │   ├── ten.py                ← TENSpoofingDetector (paper-aligned Transformer)
│   │   ├── gnn.py                ← GATLayer, GraphEncoder (3-layer GAT)
│   │   ├── ten_gnn.py            ← TENGNNModel (TEN + GNN fusion)
│   │   └── hawkes.py             ← HawkesEstimator (MLE, double-exponential kernel)
│   │
│   ├── training/
│   │   └── trainer.py            ← TENTrainer (used by train_kaggle_full.py)
│   │
│   ├── deployment/
│   │   ├── inference_engine.py   ← SpoofingInferenceEngine + TorchScript export
│   │   ├── hawkes_updater.py     ← live Hawkes background thread (500ms refresh)
│   │   └── quantize.py           ← dynamic INT8 quantization
│   │
│   ├── evaluation/
│   │   ├── metrics.py            ← compute_point_metrics, bootstrap_metrics
│   │   ├── benchmarks.py         ← classical ML vs deep model latency/F1 (synthetic data)
│   │   ├── ablation.py           ← architectural variants latency comparison (random weights)
│   │   ├── flash_crash_sim.py    ← NAVSTRAT layering detection delay simulation
│   │   └── ood_validation.py     ← synthetic OOD microstructure robustness test
│   │
│   └── explainability/
│       ├── shap_explainer.py     ← SHAP feature attribution (GradientExplainer)
│       ├── integrated_gradients.py  ← IG implementation + confidence trace
│       └── report_generator.py   ← alert report combining SHAP + IG
│
├── tests/
│   ├── test_deployment_compat.py ← TorchScript + quantization shape tests
│   ├── test_kaggle_adapters.py   ← adapter normalization tests
│   └── test_metrics.py           ← metrics function tests
│
└── paper.md                      ← reference (empty — paper notes go here)
```

---

## Working files — what each does

| File | Role | Input | Output | Pipeline stage |
|:---|:---|:---|:---|:---|
| `src/config.py` | Centralized constants | — | constants | All stages |
| `src/data/kaggle_adapters.py` | Column normalization | Raw Kaggle DataFrame | Normalized DataFrame | Data |
| `src/data/kaggle_loader.py` | Load → features → labels → windows | CSV path | (X, y) arrays | Data |
| `src/data/dataset.py` | PyTorch Dataset + splits | (X, y) arrays | DataLoaders | Data |
| `src/data/adversarial_injection.py` | Synthetic sequences for smoke testing | seed, n_sequences | (X, y) arrays | Testing |
| `src/data/feature_engineering.py` | 49-feature LOBFeatureEngineer | LOBSTER DataFrame | feature matrix | LOBSTER path |
| `src/data/labeling.py` | L(t) event labels + windowing | LOBSTER DataFrame + features | (X, y) | LOBSTER path |
| `src/data/lobster_loader.py` | Synthetic LOBSTER data generator | n_events, seed | DataFrame | Testing |
| `src/data/smoke_test.py` | End-to-end LOBSTER pipeline test | — | pass/fail | Testing |
| `src/models/ten.py` | Transformer Encoder (per-asset) | (B, T, D) | (B, 256) embedding | Model |
| `src/models/gnn.py` | 3-layer GAT | node features + adjacency | aggregated embeddings | Model |
| `src/models/ten_gnn.py` | TEN + GNN fusion | (B, N, T, D) + adj | (B, 2) logits | Model |
| `src/models/hawkes.py` | Hawkes MLE estimator | event timestamps dict | (N, N) adjacency | Model |
| `src/training/trainer.py` | TENTrainer loop | DataLoaders + model | checkpoint + CSV log | Training |
| `src/deployment/hawkes_updater.py` | Live Hawkes background thread | streaming events | (N, N) adjacency | Inference |
| `src/deployment/inference_engine.py` | Production inference engine | checkpoint path | SpoofingInferenceEngine | Inference |
| `src/deployment/quantize.py` | Dynamic INT8 quantization | checkpoint path | quantized model | Deployment |
| `src/evaluation/metrics.py` | Precision/recall/F1 | y_true, y_pred | metrics dict | Evaluation |
| `src/evaluation/benchmarks.py` | Classical ML vs deep model comparison | synthetic data | latency + F1 table | Evaluation |
| `src/evaluation/ablation.py` | Architecture variant comparison | random weights | latency + param count | Evaluation |
| `src/evaluation/flash_crash_sim.py` | NAVSTRAT detection delay test | trained model (optional) | detection delay ms | Evaluation |
| `src/evaluation/ood_validation.py` | Synthetic OOD robustness test | checkpoint + synthetic data | F1 per microstructure | Evaluation |
| `src/explainability/shap_explainer.py` | SHAP feature attribution | model + sequences + adj | importance dict | Explainability |
| `src/explainability/integrated_gradients.py` | IG implementation + confidence trace | model + sequences + adj | IG tensor / trace list | Explainability |
| `src/explainability/report_generator.py` | Alert report (SHAP + IG) | model + sequences | text report file | Explainability |
| `train_tengnn_full.py` | Multi-asset TEN-GNN training | Kaggle CSVs | checkpoint | Training |
| `train_kaggle_full.py` | Single-asset TEN training | Kaggle CSVs | checkpoint | Training |
| `tests/` | Automated correctness tests | — | pass/fail | Testing |

---

## Training

### Multi-asset TEN-GNN (primary model)

```bash
cd services/models/spoofing_detection

# Full run (all Kaggle data, 50 epochs)
python train_tengnn_full.py --assets BTC ETH SOL --frequency 1min --epochs 50

# Quick smoke run (5000 rows per asset, 5 epochs)
python train_tengnn_full.py --assets BTC ETH SOL --frequency 1min \
    --max-rows 5000 --epochs 5
```

Output: `checkpoints/tengnn_btc_eth_sol_1min/best_model.pt`

The checkpoint stores: `model_state_dict`, `optimizer_state_dict`, `val_f1`, `epoch`, `adj` (Hawkes adjacency matrix), `config` (n_assets, d_input, d_model).

### Single-asset TEN (baseline)

```bash
python train_kaggle_full.py --asset BTC --frequency 1min --epochs 50
```

Output: `checkpoints/kaggle_btc_1min/best_model.pt`

---

## Inference

```python
from src.deployment.inference_engine import SpoofingInferenceEngine
import torch

engine = SpoofingInferenceEngine.from_checkpoint(
    "checkpoints/tengnn_btc_eth_sol_1min/best_model.pt"
)

# sequences: (B, N, T, D) = (batch, 3 assets, 100 events, 33 features)
sequences = torch.randn(8, 3, 100, 33)

probs = engine.infer(sequences)    # (B,) → P(spoofing) per window
preds = engine.predict(sequences)  # (B,) → binary labels at threshold 0.5
```

---

## Live Hawkes updating

```python
from src.deployment.hawkes_updater import HawkesUpdater

updater = HawkesUpdater(n_assets=3, refresh_interval_ms=500)
updater.start()

# Push events as they arrive (asset_idx: 0=BTC, 1=ETH, 2=SOL)
updater.push_event(asset_idx=0, timestamp_minutes=1234.5)
updater.push_events_batch(asset_idx=1, timestamps_minutes=np.array([1234.6, 1234.7]))

# Attach to inference engine — all subsequent engine.infer() calls use live adjacency
engine.attach_hawkes_updater(updater)

# Read current adjacency directly
adj = updater.adjacency  # (3, 3) numpy array

updater.stop()
```

The updater needs ≥50 events per asset before it runs its first fit; it returns the identity matrix until then.

---

## Explainability

```python
from src.explainability.shap_explainer import get_global_importance
from src.explainability.report_generator import generate_alert_report
from pathlib import Path
import torch

# Feature names (33 for Kaggle path)
feature_names = ["midpoint", "spread"] + \
    [f"bid_notional_{i}" for i in range(1, 16)] + \
    [f"ask_notional_{i}" for i in range(1, 16)] + \
    ["imbalance"]

# Global importance across a batch
sequences = torch.randn(20, 3, 100, 33)
global_shap = get_global_importance(
    model=engine._model,
    sequences=sequences,
    adjacency=engine._default_adj,
    out_path=Path("results/global_shap.npy"),
    feature_names=feature_names,
)

# Per-alert report (SHAP + confidence trace)
single_seq = sequences[:1]  # (1, 3, 100, 33)
generate_alert_report(
    model=engine._model,
    sequence=single_seq,
    adjacency=engine._default_adj,
    feature_names=feature_names,
    out_path=Path("results/alert_report.txt"),
)
```

---

## Evaluation

```bash
# Run all unit tests
python -m pytest tests/ -v

# Smoke test (synthetic LOBSTER pipeline)
python -m src.data.smoke_test

# Architecture shape verification (TEN)
python -m src.models.ten

# Trainer smoke test (2 epochs on dummy data)
python -m src.training.trainer

# Baseline benchmarks (classical ML vs deep model latency — synthetic data only)
python -m src.evaluation.benchmarks

# Ablation study (random-weight architecture comparison — latency and param count only)
python -m src.evaluation.ablation

# OOD validation with real checkpoint
python -m src.evaluation.ood_validation

# Flash crash simulation
python -m src.evaluation.flash_crash_sim
```

---

## Deployment (export)

```bash
# TorchScript export
python -m src.deployment.inference_engine  # → checkpoints/ten_gnn_scripted.pt

# Dynamic INT8 quantization (deprecated API — will break in PyTorch 2.10)
python -m src.deployment.quantize          # → checkpoints/ten_gnn_int8.pt
```

---

## Limitations — what is not working or not paper-aligned

| Area | Status |
|:---|:---|
| Labels | **Heuristic pseudo-labels** (cancel ratio + imbalance rolling threshold), not L(t) |
| L(t) conditions | Cannot implement: Kaggle data lacks order-level cancel tracking and sub-second timestamps |
| Adaptive temporal PE | Falls back to standard sinusoidal PE on the Kaggle path (Δt not available at d_input=33) |
| SOL data | Synthesized from 15-min OHLCV — cancel columns are all zero, weaker Hawkes signal |
| OOD validation | Uses synthetic microstructure simulations, not real exchange data |
| Flash crash sim | Synthetic injection only — not a real real-world detection benchmark |
| Ablation study | Uses random weights — only latency and parameter counts are meaningful |
| Quantization API | `torch.ao.quantization` deprecated in PyTorch 2.10 — migrate to `torchao` |
| TorchScript | `torch.jit.script` deprecated — migrate to `torch.compile` or `torch.export` |
| Sub-millisecond HFT | Not implemented — CPU inference target is 1–30ms/window, not tick speed |
| Real ground truth | No confirmed spoofing labels — results cannot be compared to paper F1 numbers |

---

## Files removed and why

| File | Reason |
|:---|:---|
| `src/deployment/deployment_tiers.py` | Pure documentation dataclass — no code connected to any pipeline component |
| `results/evaluation_report.json` | Contained fake F1 metrics from random-weight models presented without that caveat |
| `high-frequency-crypto-limit-order-book-data.zip` | 1 GB raw zip — data already extracted to `data/raw/kaggle_crypto/` |
| `save_dummy_weights.py` | Script that saved random weights to a checkpoint — served no valid purpose |
| All `__pycache__/`, `.pyc` | Build artifacts |
| 3 Jupyter notebooks | Stale GPU-cloud training notebooks, non-reproducible locally |
| 4 redundant `.md` guides | Content merged into this README |
| 9 old checkpoint directories | Superseded by `tengnn_btc_eth_sol_1min/` |
| `results/` directory | Was empty after removing fake report |
| `training_output.log`, `kaggle_ten_smoke_local.csv` | Temporary logs |

---

## Next steps

1. **Real labels**: Obtain LOBSTER exchange data with confirmed order-level cancel events to implement the paper's L(t) labeling (V_cancel/V_placed > 90%, duration < 500ms, MIM > 1.5 ticks).
2. **SOL data**: Replace synthesized LOB with real Solana on-chain order book data.
3. **Adaptive PE on Kaggle path**: Add a computed Δt feature to `extract_lob_features` (time difference between consecutive `system_time` entries in seconds), enabling the full adaptive PE path.
4. **Quantization migration**: Replace `torch.ao.quantization.quantize_dynamic` with `torchao` eager-mode `quantize_()` API before PyTorch 2.10.
5. **TorchScript migration**: Replace `torch.jit.script` with `torch.compile` or `torch.export` for the export path.
6. **Calibration**: Add Platt scaling or temperature scaling so the raw softmax output is calibrated to true P(spoofing).
7. **Streaming data connector**: Wire `HawkesUpdater.push_event` to a live WebSocket feed (Coinbase Advanced Trade API or similar).
