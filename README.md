# Solana AI Trading System

> Production-grade AI-powered cryptocurrency trading system on Solana with real-time data ingestion, advanced ML models, reinforcement learning, autonomous agents, and execution layer.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture Layers](#architecture-layers)
- [Folder Structure](#folder-structure)
- [Quick Start](#quick-start)

---

## Project Overview

This is an enterprise-grade trading system that:
- Ingests real-time market data from multiple sources (Binance, OpenBook DEX, Helius, Reddit)
- Processes and engineers features at sub-millisecond latency
- Runs multiple AI models (TEN, GNN, XGBoost, FinBERT) in parallel
- Uses reinforcement learning (PPO) for autonomous decision-making
- Executes trades through Jupiter, Raydium, and Jito bundles on Solana
- Provides explainability via SHAP heatmaps and dashboards
- Implements risk management and circuit breakers

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 6: API & Dashboard (apps/)                            │
│ FastAPI Backend + React Frontend + SHAP Explainability      │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 4-5: Agent Layer + Execution (apps/agent, services/)  │
│ ElizaOS Orchestrator + Jupiter/Raydium/Jito Routing         │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: AI Models (services/models/)                       │
│ TEN, GNN, XGBoost, FinBERT, PPO, Fusion Layer               │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│ Infrastructure (packages/db/, infra/)                       │
│ TimescaleDB, Redis, ChromaDB, Docker, K8s, Terraform        │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Data Processing (services/processing/)             │
│ Features, Pipelines, Normalization, Sequencing              │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Data Ingestion (services/ingest/)                  │
│ Binance, OpenBook, Helius, Reddit, Cross-Asset Feeds        │
└─────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

### 📁 `apps/` — Deployable Services

User-facing and orchestration services that run as independent processes.

#### `apps/api/`
**FastAPI Backend Server (Layer 6)**
- **routers/** → HTTP endpoints for signals, positions, explanations
- **middleware/** → Authentication, rate limiting, CORS, request logging
- **schemas/** → Pydantic models for request/response validation
- **main.py** → FastAPI app initialization, startup/shutdown hooks
- **Dockerfile** → Container definition for production deployment

*Purpose: REST API for dashboard, external integrations, and client queries*

#### `apps/dashboard/`
**React Frontend (Layer 6)**
- **src/components/**
  - **Heatmaps/** → SHAP force plots, waterfall charts, dependency plots
  - **ShapExplainer/** → Model interpretation UI components
  - **LiveSignals/** → Real-time trading signal display, WebSocket updates
- **src/hooks/** → Custom React hooks for WebSocket, API calls, state management
- **src/store/** → Zustand/Redux state management for app-wide state
- **src/pages/** → Dashboard, signals, positions, settings pages
- **public/** → Static assets (logos, favicons, etc.)
- **package.json** → Dependencies and build scripts
- **Dockerfile** → Node.js production image

*Purpose: Real-time monitoring dashboard with model interpretability visualizations*

#### `apps/agent/`
**ElizaOS Orchestrator Service (Layer 4)**
- **orchestrator.py** → Agent lifecycle, plugin coordination, decision routing
- **plugins/solana_agent_kit/** → Solana-specific actions (swap, transfer, query)
- **memory/** → ChromaDB integration for vector embeddings and semantic search
- **reasoning/** → Ollama/Llama-3 integration for autonomous reasoning
- **Dockerfile** → Python runtime for agent container

*Purpose: Autonomous decision-making layer; coordinates AI models, executes decisions through plugins*

---

### 📁 `services/` — Internal Microservices (Workers)

Non-HTTP background services that process data, train models, and execute trades.

#### `services/ingest/` — Layer 1: Data Sources

Real-time market data ingestion from multiple sources.

- **connectors/**
  - `openbook_dex.py` → WebSocket L2/L3 order book from OpenBook DEX
  - `helius_ws.py` → Helius RPC WebSocket for on-chain events
  - `binance.py` → Binance REST API + WebSocket for spot/futures
  - `cross_asset.py` → Multi-pair correlation and cross-asset feeds
  - `reddit.py` → Reddit PRAW API for sentiment monitoring
- **normalizer.py** → Converts all feeds to unified tick format
- **dispatcher.py** → Redis Streams publisher for downstream consumers
- **Dockerfile** → Container for ingest service

*Purpose: Continuously streams normalized market data into the pipeline*

#### `services/processing/` — Layer 2: Data Processing

Feature engineering and pipeline orchestration.

- **features/**
  - `micro.py` → Microstructure features (bid-ask imbalance, spread, depth)
  - `macro.py` → Technical indicators (RSI, MACD, VWAP, Bollinger Bands)
  - `hawkes.py` → Hawkes process MLE for self-exciting arrival intensity
  - `nlp_sentiment.py` → Sentiment scoring from NLP on news/social feeds
- **pipeline/**
  - `normalizer.py` → Data cleaning, missing value handling, outlier removal
  - `sequencer.py` → Windowing and sliding windows for time-series models
- **writer.py** → TimescaleDB writer for features and raw ticks
- **Dockerfile** → Processing service container

*Purpose: Transforms raw market data into engineered features for ML models*

#### `services/models/` — Layer 3: AI Models

Model inference, training coordination, and fusion.

- **temporal/**
  - `ten_encoder.py` → TEN (Transformer Encoder Network) for sub-millisecond encoding
- **spatial/**
  - `gnn_encoder.py` → GNN (Graph Neural Network) for market structure relationships
- **classification/**
  - `xgboost_updown.py` → XGBoost binary classifier for up/down price movement
- **nlp/**
  - `finbert.py` → FinBERT for financial sentiment and document classification
- **fusion/**
  - `aggregator.py` → Ensemble aggregation (voting, stacking, weighted average)
- **rl/**
  - `ppo_agent.py` → PPO reinforcement learning agent for trade sizing/timing
  - `env.py` → Custom Gym environment for RL training
  - `reward.py` → Reward function definition (Sharpe, profit, risk-adjusted)
- **registry.py** → Model versioning, loading, and artifact management
- **serving.py** → Inference server (Triton or custom ONNX runtime)
- **Dockerfile** → GPU-enabled container for model inference

*Purpose: Runs inference on engineered features; produces trading signals and confidence scores*

#### `services/execution/` — Layer 5: Trade Execution

Execution routing, risk management, and blockchain interaction.

- **router.py** → Routes signals to standard or HFT execution paths
- **standard/**
  - `jupiter.py` → Jupiter aggregator for best-price swaps
  - `raydium.py` → Raydium AMM direct interactions
- **hft/**
  - `jito_bundler.py` → Jito MEV bundle submission for atomic execution
- **rpc/**
  - `helius_node.py` → Helius RPC client for fast on-chain queries
- **risk/**
  - `position_sizer.py` → Kelly Criterion, volatility-based position sizing
  - `circuit_breaker.py` → Drawdown limits, trade frequency caps, emergency stops
- **Dockerfile** → Execution service container

*Purpose: Executes approved trades with risk management, handles MEV, manages positions*

---

### 📁 `contracts/` — Solana On-Chain Programs

Anchor/Rust smart contracts for protocol-level trading logic.

- **programs/trader/src/**
  - **instructions/**
    - `open_position.rs` → Create new position (account initialization)
    - `close_position.rs` → Close/liquidate position
  - **state/** → PDA (Program-Derived Account) definitions and state schemas
  - `lib.rs` → Program entry point
- **tests/** → TypeScript/Anchor tests (trader.ts)
- **migrations/** → Deployment scripts and state migrations
- **Anchor.toml** → Anchor framework configuration
- **Cargo.toml** → Rust dependencies

*Purpose: On-chain settlement, position tracking, and atomic trade execution*

---

### 📁 `packages/` — Shared Libraries

Importable internal packages used across all services.

#### `packages/core/`
Shared types and constants.

- **types/**
  - `signal.py` → Signal dataclass (timestamp, symbol, confidence, action)
  - `market.py` → Market data types (OHLCV, orderbook snapshot, tick)
  - `position.py` → Position types (entry price, size, P&L, status)
- `constants.py` → Global constants (symbols, thresholds, timeouts)

*Purpose: Ensures consistent type definitions across all services*

#### `packages/db/`
Database clients and schemas.

- **timescale/**
  - `client.py` → TimescaleDB connection pool, query helpers
  - `migrations/` → SQL migration files for schema versioning
- **redis/**
  - `client.py` → Redis streams, pub/sub helpers, caching utilities
- **chroma/**
  - `client.py` → ChromaDB vector store client for semantic search

*Purpose: Centralized database access layer*

#### `packages/utils/`
Utility functions and helpers.

- `logging.py` → Structured logging with structlog
- `metrics.py` → Prometheus metrics helpers
- `retry.py` → Exponential backoff and retry decorators

*Purpose: Shared utilities for logging, monitoring, and resilience*

---

### 📁 `infra/` — Infrastructure as Code

Deployment, orchestration, and cloud provisioning.

#### `infra/docker/`
Docker Compose configurations.

- `docker-compose.yml` → Production-like full stack (all services)
- `docker-compose.dev.yml` → Development overrides (hot reload, debug)
- `docker-compose.prod.yml` → Production overrides (resource limits, healthchecks)

*Purpose: Local and production environment setup*

#### `infra/k8s/`
Kubernetes manifests (for scaling).

- **base/** → Core K8s resources (deployments, services, ConfigMaps)
- **overlays/dev/** → Development overrides (1 replica, dev resources)
- **overlays/prod/** → Production overrides (replicas, PVs, HPA)

*Purpose: Multi-environment Kubernetes deployment*

#### `infra/terraform/`
Cloud provisioning (AWS/GCP/Azure).

*Purpose: IaC for managed databases, networking, VMs*

#### `infra/scripts/`
Bootstrapping and maintenance scripts.

- `setup_wsl.sh` → WSL2 Ubuntu initial setup (Solana CLI, Anchor, Rust, Docker)
- `seed_db.sh` → Populate test data into TimescaleDB
- `migrate.sh` → Run database migrations

*Purpose: One-shot setup and maintenance*

---

### 📁 `.ci/` — Continuous Integration/Deployment

GitHub Actions or GitLab CI workflows.

- **workflows/**
  - `test.yml` → Unit, integration, and contract tests
  - `build.yml` → Build Docker images, push to registry
  - `deploy.yml` → Deploy to K8s/cloud on merge to main
- **scripts/**
  - `lint_contracts.sh` → Rust contract linting

*Purpose: Automated testing, building, and deployment*

---

### 📁 `notebooks/` — Research & Experimentation

Jupyter notebooks for exploration (NOT in production path).

- **eda/** → Exploratory data analysis (market patterns, correlations)
- **model_training/**
  - `train_gnn.ipynb` → GNN training and hyperparameter tuning
  - `train_ppo.ipynb` → PPO agent training and policy optimization
  - `train_xgboost.ipynb` → XGBoost training and feature importance
- **backtesting/** → Backtest notebooks with historical data

*Purpose: Development and research isolated from production*

---

### 📁 `tests/` — Test Suites

Automated tests at multiple levels.

- **unit/**
  - `test_features.py` → Feature engineering unit tests
  - `test_models.py` → Model inference unit tests
  - `test_execution.py` → Execution logic unit tests
- **integration/**
  - `test_pipeline.py` → End-to-end pipeline tests (ingest → models → execution)
  - `test_agent.py` → Agent orchestration tests
- **e2e/**
  - `test_full_cycle.py` → Full cycle tests (data → signal → execution)

*Purpose: Quality assurance across all layers*

---

### 📁 `data/` — Local Development Data

Local-only data (gitignored in production).

- **raw/** → Raw downloaded market data (Binance, OpenBook snapshots)
- **processed/** → Processed features and engineered data
- **model_artifacts/**
  - **gnn/** → Trained GNN model weights and checkpoints
  - **ppo/** → PPO agent policy and value function
  - **xgboost/** → XGBoost model binary files

*Purpose: Development data and model checkpoints (not committed to git)*

---

### 📁 `docs/` — Documentation

- **architecture.md** → System design, layer interactions, data flow
- **runbooks/**
  - `deploy.md` → Deployment procedures
  - `incident_response.md` → Troubleshooting and runbooks
- **adr/** → Architecture Decision Records (design choices)

*Purpose: Team documentation and operational guides*

---

### 📄 Root Configuration Files

- **`.env.example`** → Template for environment variables (never commit `.env`)
- **`.gitignore`** → Git ignore rules (data/, model_artifacts/, .env)
- **`Makefile`** → Build targets (`make dev`, `make test`, `make deploy`)
- **`pyproject.toml`** → Python workspace root (dependencies, build system)
- **`pnpm-workspace.yaml`** → Node monorepo configuration (apps/api, apps/dashboard)
- **`README.md`** → This file

---

## Quick Start

### WSL2 Setup
```bash
git clone <repo> ~/projects/solana-ai-trader
cd ~/projects/solana-ai-trader
make setup        # Bootstrap WSL2 (Solana, Anchor, Docker)
make dev          # Start full dev stack
```

### Services
```bash
make test         # Run all tests
make lint         # Lint Python and Rust
make deploy       # Build and push Docker images
```

---

## Key Principles

- **Microservices**: Each service is independently deployable
- **Separation of Concerns**: Clear layer boundaries (ingest → processing → models → execution)
- **Scalability**: Docker + K8s ready; stateless services; distributed data stores
- **Testability**: Unit, integration, and E2E test coverage
- **Production-Ready**: Logging, metrics, error handling, circuit breakers, risk management
- **WSL2 Optimized**: All paths use `~/` (native WSL2 filesystem, not `/mnt/c/`)

---

## Architecture at a Glance

```
🔴 Layer 1 (Ingest)   → Real-time market data from Binance, DEX, RPC, Reddit
    ↓
🟠 Layer 2 (Process)  → Feature engineering, normalization, windowing
    ↓
🟡 Layer 3 (Models)   → TEN, GNN, XGBoost, FinBERT, PPO inference
    ↓
🟢 Layer 4 (Agent)    → ElizaOS orchestrator, autonomous decision-making
    ↓
🔵 Layer 5 (Exec)     → Trade routing, risk management, Solana execution
    ↓
🟣 Layer 6 (API/UI)   → FastAPI backend, React dashboard, SHAP explainability
```

---

**Status**: Production-ready architecture  
**Last Updated**: April 2026  
**Maintainers**: Nour Maatouk 
