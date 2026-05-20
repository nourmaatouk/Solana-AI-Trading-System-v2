# Solana AI Trading System

AI-powered cryptocurrency trading system targeting SOL/USDT on Solana. The signal generation pipeline (data ingestion → feature engineering → ML models → fusion → paper trading) is fully operational. The execution layer and agent orchestration are planned for future phases.

## Table of Contents

- [What Works Today](#what-works-today)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Folder Structure](#folder-structure)
- [Production Pipeline Status](#production-pipeline-status)

---

## What Works Today

| Component | Status | Notes |
|-----------|--------|-------|
| XGBoost classification pipeline | ✅ Working | 47 features, H1/H4/H8/H16 horizons |
| TEN-GNN spoofing detection | ✅ Working | Trained checkpoint (8.3MB), test F1=0.522 |
| PPO reinforcement learning agent | ✅ Working | Trained, wired into fusion |
| FinBERT sentiment (NLP) | ✅ Working | GDELT + HuggingFace, 15-min buckets |
| Fusion aggregator | ✅ Working | XGB 40% / RL 30% / Sentiment 20% / Spoofing 10% |
| Paper trader | ✅ Working | Reads `last_fused_decision.json` |
| FastAPI backend | ✅ Working | Read-only monitoring endpoints |
| Node.js trading dashboard | ✅ Working | Live Binance WS, candlestick chart, order book (apps/trading-dashboard) |
| Binance WebSocket orderbook | ✅ Working | SOL/USDT L2, writes orderbook_features.csv |
| CI pipeline | ✅ Working | `.github/workflows/ci.yml` |
| Hawkes process realtime gate | ✅ Working | Branching-ratio spoofing filter |
| ElizaOS agent orchestration | ⏳ Planned | Phase 5 — no orchestrator yet (apps/agent/ is empty stub) |
| Solana execution layer | ⏳ Planned | Phase 6 — Jupiter/Raydium/Jito |
| On-chain connectors | ⏳ Planned | OpenBook DEX, Helius RPC |
| Solana smart contracts | ⏳ Planned | Phase 6 — Anchor/Rust |
| SHAP explainability dashboard | ⏳ Planned | Next sprint |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 6 (API/UI)       apps/api · apps/trading-dashboard         │
│ FastAPI monitoring endpoints                                      │
│ Node.js live trading dashboard (TSX + Vite + Express proxy)      │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────┐
│ Layer 4-5 (Agent/Exec)  [PLANNED]                                │
│ ElizaOS orchestrator · Jupiter/Raydium/Jito routing              │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────┐
│ Layer 3 (Models)        services/models/                         │
│ XGBoost · TEN-GNN · PPO · FinBERT · Fusion aggregator           │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────┐
│ Layer 1-2 (Ingest/Process)  services/ingest/ · services/models/  │
│ Binance WebSocket (live) · Feature engineering · Sentiment fetch  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone and create venv
git clone <repo> ~/Solana-AI-Trading-System
cd ~/Solana-AI-Trading-System
python -m venv .venv && source .venv/bin/activate
pip install -r services/models/classification/requirements.txt

# Run the full pipeline once (fetches data, runs all models, writes last_fused_decision.json)
python services/models/classification/run_realtime_pipeline.py --once

# Start the FastAPI backend
uvicorn apps.api.main:app --port 8000

# Start the Node.js live trading dashboard (separate terminal pair)
cd apps/trading-dashboard/backend && npm install && npm run dev   # port 3001
cd apps/trading-dashboard/frontend && npm install && npm run dev  # port 5174
```

### Running individual models

```bash
# Orderbook collector (continuous)
python services/ingest/connectors/binance_orderbook.py

# Spoofing gate
python services/models/classification/realtime_spoofing_detector.py

# PPO RL signal
python services/models/rl/realtime_rl_predictor.py

# Train TEN-GNN from scratch
cd services/models/spoofing_detection
python train_tengnn_full.py --d-model 128 --n-layers 3 --dropout 0.4 --epochs 30

# Train PPO agent
python services/models/rl/train_ppo.py
```

### Tests

```bash
python -m pytest services/models/classification/tests/ -v
python -m pytest services/models/nlp/tests/ -v
python -m pytest services/models/fusion/tests/ -v
python -m pytest services/models/spoofing_detection/ -v
```

---

## Folder Structure

### `apps/` — User-Facing Applications

#### `apps/api/` ✅ Implemented
FastAPI backend — read-only monitoring endpoints (signal, paper-trading, health, status, preferences).

```
main.py          — FastAPI app, all endpoints
requirements.txt — Dependencies
tests/           — API tests
```

#### `apps/trading-dashboard/` ✅ Implemented
Separate Node.js/Express backend + React/Vite frontend. Streams live Binance WebSocket data, overlays AI signals from FastAPI.

```
backend/   — Express :3001, proxies /api/ai/* to FastAPI, broadcasts WS to frontend
frontend/  — Vite :5174, candlestick chart, order book, watchlist, signal panel
```

#### `apps/agent/` ⏳ Empty stub
Reserved for the ElizaOS orchestration layer (Phase 5). No code yet — orchestrator,
plugins, memory, and reasoning are all planned.

```
memory/                — [PLANNED] ChromaDB episodic memory (Phase 5)
plugins/               — [PLANNED] Solana Agent Kit (Phase 6)
reasoning/             — [PLANNED] Ollama Llama-3 reasoning (Phase 5)
```

---

### `services/` — Background Workers

#### `services/ingest/` ✅ Partially implemented
```
connectors/
  binance_orderbook.py   — ✅ Live WebSocket SOL/USDT L2 orderbook → CSV
  openbook_orderbook.py  — [PLANNED] OpenBook DEX on-chain L2
  helius_rpc.py          — [PLANNED] Helius enriched blockchain events
```

#### `services/models/` ✅ Core pipeline implemented
```
classification/          — ✅ XGBoost 4-hour trend classifier
  config.py              — All thresholds, paths, feature counts
  run_realtime_pipeline.py — Orchestrates all pipeline steps
  realtime_spoofing_detector.py — Hawkes gate + TEN-GNN inference
  data/realtime/         — Live data files (orderbook, sentiment, signals)
  checkpoints/           — Trained XGBoost model binary

nlp/                     — ✅ FinBERT sentiment pipeline
  src/models/finbert_predictor.py
  src/aggregation/feature_aggregator.py

spoofing_detection/      — ✅ TEN-GNN trained model
  src/models/ten_gnn.py  — TEN-GNN architecture
  src/models/ten.py      — Transformer encoder with ATPE
  src/models/gnn.py      — Graph attention network
  src/models/hawkes.py   — Double-exponential Hawkes estimator
  src/deployment/inference_engine.py
  checkpoints/tengnn_btc_eth_sol_1min/best_model.pt  — Trained (8.3MB)
  train_tengnn_full.py   — Training script

rl/                      — ✅ PPO agent fully trained
  config.py              — All RL hyperparameters and paths
  environment.py         — TradingEnv (gymnasium.Env)
  train_ppo.py           — PPO training script
  evaluate.py            — Backtest + equity curve
  realtime_rl_predictor.py — Realtime inference → last_rl_signal.json
  (model file: services/models/models/ppo_trading_agent.zip)

fusion/                  — ✅ Multi-signal aggregation
  src/fusion_aggregator.py  — Weighted ensemble
  src/fusion_contracts.py   — Signal dataclasses
  config.py                 — Weights: XGB 40%, RL 30%, Sentiment 20%, Spoofing 10%

spatial/                 — [PLANNED] Standalone GNN module (Phase 5)
temporal/                — [PLANNED] Standalone LSTM/Transformer (Phase 5)
```

#### `services/execution/` ⏳ Not built
```
[PLANNED Phase 6]
standard/   — Jupiter/Raydium/Orca swap routing
hft/        — Jito MEV bundle submission
rpc/        — Solana RPC connection pool
risk/       — Pre-trade checks, circuit breaker, position sizing
```

#### `services/processing/` ⏳ Not built
Feature engineering is currently embedded in each model service. A centralized processing layer is planned.

---

### `packages/` ⏳ All stub
Shared library stubs — no working code exists yet.
```
core/    — [PLANNED] Type definitions
db/      — [PLANNED] TimescaleDB, Redis, ChromaDB clients
utils/   — [PLANNED] Shared utilities
```

### `contracts/` ⏳ Not built
No Rust or Anchor code. On-chain programs are planned after the execution layer is stable on devnet (Phase 6).

### `infra/`
```
systemd/  — ✅ Orderbook collector systemd service template
docker/   — [PLANNED] Dockerfiles and Compose files
k8s/      — [PLANNED] Kubernetes manifests
terraform/ — [PLANNED] Cloud provisioning IaC
scripts/  — [PLANNED] Bootstrap and maintenance scripts
```

### `.github/workflows/ci.yml` ✅ CI pipeline
Runs classification, NLP, fusion, and spoofing tests on every push.

### `data/`
```
raw/              — Raw CSV downloads (Binance, Kaggle LOB)
processed/        — Engineered feature CSVs (47-feature training data)
model_artifacts/
  xgboost/        — [empty] XGBoost checkpoint lives in services/models/classification/checkpoints/
  ppo/            — [empty] PPO checkpoint lives in services/models/models/
  gnn/            — [empty] TEN-GNN checkpoint lives in services/models/spoofing_detection/checkpoints/
```

### `notebooks/`
```
eda/             — Exploratory data analysis
model_training/  — Training experiment notebooks
backtesting/     — Backtest notebooks
```

---

## Production Pipeline Status

### Signal Pipeline (May 2026)

The full multi-branch signal pipeline is validated end-to-end:

```
Binance WebSocket (SOL/USDT L2)
        │
        ▼
Orderbook Features CSV ──────────────┐
        │                            │
        ▼                            ▼
XGBoost Classifier         Hawkes Gate + TEN-GNN
(4-hour trend, 47 features) (spoofing probability)
        │                            │
        ▼                            │
FinBERT Sentiment ───────────────────┤
(GDELT + HuggingFace)                │
        │                            │
        ▼                            │
PPO RL Agent ────────────────────────┘
(gymnasium TradingEnv)               │
                                     ▼
                          Fusion Aggregator
                    XGB 40% + RL 30% + Sentiment 20% + Spoofing 10%
                                     │
                                     ▼
                         last_fused_decision.json
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                     Paper Trader          FastAPI / Dashboard
```

### Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| XGBoost H4 | Val AUC | 0.71 |
| XGBoost confidence gate | Sharpe-optimal threshold | 0.69 |
| TEN-GNN | Test F1 | 0.522 (P=0.603, R=0.461) |
| TEN-GNN | Checkpoint | epoch=4, val_f1=0.369, threshold=0.51 |
| PPO | Training timesteps | 500,000 |

### Layer 2/3 Contracts

| Contract Layer | Tests | Status |
|---------------|-------|--------|
| NLP Layer 2 contracts | 2/2 | ✅ |
| Fusion contracts | 9/9 | ✅ |
| Fusion aggregator | 9/9 | ✅ |
| Fusion batch | 3/3 | ✅ |
| Fusion integration | 2/2 (1 skipped) | ✅ |
| End-to-end pipeline | 13/13 | ✅ |

---

## Key Principles

- **Signal pipeline only**: No real money is at risk. All trading is paper-simulated.
- **Chronological splits**: All train/val/test splits are time-ordered to prevent leakage.
- **Honest status**: Components are clearly marked as working or planned — nothing is described as built until it is.
- **Confidence gate**: Signals below the 0.69 threshold emit NEUTRAL rather than force a trade.
- **Spoofing override**: Spoofing probability > 0.7 forces HOLD regardless of other signals.

---

**Maintainer**: Nour Maatouk  
**Last Updated**: May 2026
