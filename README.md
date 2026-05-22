# Solana AI Trading System

AI-powered cryptocurrency trading system targeting SOL/USDT on Solana. The full chain — data ingestion → feature engineering → ML models → fusion → paper trading **and** Guardian-validated execution + Phantom signing — runs end-to-end. Mainnet execution and an on-chain Anchor Guardian are still planned.

## Table of Contents

- [What Works Today](#what-works-today)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Folder Structure](#folder-structure)
- [Production Pipeline Status](#production-pipeline-status)
- [Key Principles](#key-principles)

---

## What Works Today

| Component | Status | Notes |
| --- | --- | --- |
| XGBoost classification pipeline | ✅ Working | 47 features, H1/H4/H8/H16 horizons; H4 active in production |
| TEN-GNN spoofing detection | ✅ Working | Trained checkpoint (8.3 MB); Hawkes gate runs every cycle |
| FinBERT sentiment (NLP) | ✅ Working | GDELT + RSS + Reddit → 15-min buckets; timeout-prone live |
| Fusion aggregator | ✅ Working | XGB 50% / Sentiment 25% / Spoofing 20% / GNN 5% / RL 0% |
| PPO reinforcement-learning agent | ✅ Trained, ❌ Disabled | 2M+ training steps; `WEIGHT_RL=0.0` in fusion (val Sharpe 1.926, test –0.308) |
| Paper trader (Python) | ✅ Working | Reads `last_fused_decision.json`; gated by Python Guardian |
| FastAPI backend (`apps/api`) | ✅ Working | Read-mostly + scoped write endpoints (`/preferences`, `/scheduler/run`, `/retrain/trigger`) |
| Node trading dashboard | ✅ Working | Live Binance WS, candlestick chart, order book, execution panel |
| TypeScript Guardian | ✅ Working | 14 rules, first-failure-wins, 47 unit + 16 integration tests |
| `AutoExecutor` (Node) | ✅ Working | Event-driven from SignalPoller; persistent dedup; reads user_risk_level/max_trade_size/auto_execute from disk |
| Phantom wallet signing | ✅ Working | Frontend `useAutoExecution` hook; backend never signs in phantom mode |
| Paper / devnet execution | ✅ Working | Paper simulates; devnet uses Solana RPC + optional Jupiter swap |
| `ElizaOS` narrator | ✅ Working | Post-decision narration only — never on critical path |
| CI pipeline | ⚠️ Python only | `.github/workflows/ci.yml` does not run TypeScript / Guardian tests yet |
| Mainnet execution | ⏳ Gated | `ALLOW_MAINNET_EXECUTION=false` by default; explicit env opt-in required |
| On-chain Anchor Guardian | ⏳ Planned | Off-chain Guardian feeds the same contract once deployed |
| Solana Agent Kit plugins | ⏳ Planned | Phase 6 |
| SHAP explainability dashboard | ⏳ Planned | Next sprint |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 6  (UI/API)         apps/api · apps/trading-dashboard             │
│   • FastAPI :8000  (signal + paper-trading + control endpoints)         │
│   • Node :3001     (Express + WebSocket, proxies /api/ai/* to FastAPI)  │
│   • Vite :5174     (React + Zustand + Phantom wallet adapter)           │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│ Layer 5  (Execution)      services/execution + apps/.../autoExecutor    │
│   • TypeScript Guardian (14 rules) · ExecutionWorkflow · PaperExecutor  │
│   • DevnetExecutor + MainnetExecutor (Solana RPC; Jupiter v6 swap)      │
│   • AutoExecutor (Node)   — event-driven, persistent dedup              │
│   • useAutoExecution      — Phantom signing on the browser; arm toggle  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│ Layer 4  (Agent)                                                        │
│   • ElizaOS narrator (post-decision text only; never decides trades)    │
│   • Python Guardian gate (paper trader internal pre-trade check)        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│ Layer 3  (Models)         services/models/                              │
│   XGBoost · TEN-GNN · FinBERT · Isolation-forest GNN · PPO (disabled)   │
│   Fusion aggregator   →   last_fused_decision.json                      │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│ Layer 1-2  (Ingest + features)                                          │
│   services/ingest · services/models/classification/realtime_*.py         │
│   Binance WebSocket + REST · GDELT/RSS sentiment · orderbook collector  │
└─────────────────────────────────────────────────────────────────────────┘
```

The Guardian sits on the critical path twice, at two different layers:
1. **Python Guardian** inside the paper trader (`services/models/classification/guardian_agent.py`) gates simulated trades.
2. **TypeScript Guardian** inside `services/execution/` validates every Solana transaction — paper, devnet, or mainnet — before it touches an executor. The LLM is never on this path.

---

## Quick Start

> A verified launch sequence with smoke-test checklist also lives in `.claude/skills/run-app/SKILL.md`.

```bash
# Clone + Python env
git clone <repo> ~/Solana-AI-Trading-System
cd ~/Solana-AI-Trading-System
python -m venv .venv && source .venv/bin/activate
pip install -r services/models/classification/requirements.txt
pip install -r apps/api/requirements.txt

# Build the TypeScript execution package once. Re-run whenever services/execution/src changes.
cd services/execution && npm install && npm run build && cd -

# Backend Node dependencies
cd apps/trading-dashboard/backend && npm install && cd -
cd apps/trading-dashboard/frontend && npm install && cd -
```

Run the full stack in three terminals:

```bash
# Terminal 1 — FastAPI on :8000 (serves model outputs)
.venv/bin/uvicorn apps.api.main:app --port 8000 --host 127.0.0.1

# Terminal 2 — Node backend on :3001 (Express + WebSocket, mounts the execution layer)
cd apps/trading-dashboard/backend && PORT=3001 npx ts-node-dev -r tsconfig-paths/register src/server.ts

# Terminal 3 — Vite frontend on :5174 (React dashboard)
cd apps/trading-dashboard/frontend && npx vite
```

Then open **http://127.0.0.1:5174/** and the dashboard should render. Smoke-test the API chain:

```bash
curl -s http://localhost:3001/api/execution/auto-execute/status
# {"enabled":false,"user_risk_level":null,"max_trade_size_sol":null}

curl -s http://localhost:3001/api/guardian/policy | python3 -m json.tool
# Must include allowHighRiskTrades and defaultUserRiskLevel.
# If missing → services/execution/dist/ is stale; rerun `npm run build` there.
```

Run a one-shot ML pipeline pass to write a fresh `last_fused_decision.json`:

```bash
python services/models/classification/run_realtime_pipeline.py --once --fusion-aggregate --paper-trade
```

### Tests

```bash
# 63 tests across the execution package (Guardian rules + workflow integration)
cd services/execution && npm test

# 9 tests on the SignalPoller → AutoExecutor wiring
cd apps/trading-dashboard/backend && npm test

# Python ML test suites
python -m pytest services/models/classification/tests/ -v
python -m pytest services/models/nlp/tests/ -v
python -m pytest services/models/fusion/tests/ -v
python -m pytest services/models/spoofing_detection/ -v
```

### Stopping the stack

```bash
pkill -f "ts-node-dev.*server.ts"
pkill -f "vite"
pkill -f "uvicorn.*apps.api"
```

---

## Folder Structure

### `apps/` — User-facing applications

#### `apps/api/` ✅ Implemented
FastAPI service. ~20 endpoints across health, signal, paper-trading, scheduler, accuracy, drift, backtest, walk-forward, retrain. See [`apps/api/README.md`](apps/api/README.md) for the full inventory.

#### `apps/trading-dashboard/` ✅ Implemented
- `backend/` — Express + WebSocket on `:3001`. Proxies `/api/ai/*` to FastAPI, mounts `/api/execution/*`, `/api/guardian/*`, `/api/wallet/*`, streams Binance market data on `/ws`.
- `frontend/` — Vite + React + Zustand on `:5174`. Pages: Dashboard, LiveMarket, WhySignal, BacktestPage, AlertsPage, Execution, SystemStatus, HowItWorks, Settings.

#### `apps/agent/` ⏳ Stub
ElizaOS character JSON exists but no orchestrator code yet. The narration runtime is embedded in `services/execution/src/orchestrator/elizaRuntime.ts` for post-decision text generation only.

### `services/` — Background workers and libraries

#### `services/ingest/` ✅ Partial
- `connectors/binance_orderbook.py` — Live SOL/USDT L2 → CSV.
- `connectors/openbook_orderbook.py` — Planned (Phase 6).
- `connectors/helius_rpc.py` — Planned (Phase 6).

#### `services/models/` ✅ Core pipeline implemented
- `classification/` — XGBoost pipeline + realtime orchestrator + Python Guardian + paper trader.
- `nlp/` — FinBERT + GDELT/RSS/Reddit fetchers.
- `spoofing_detection/` — TEN-GNN + Hawkes; trained checkpoint at `checkpoints/tengnn_btc_eth_sol_1min/`.
- `fusion/` — Weighted aggregator; weights `XGB 0.50 / Sentiment 0.25 / Spoofing 0.20 / GNN 0.05 / RL 0.00`.
- `rl/` — PPO; trained, currently disabled in fusion (`WEIGHT_RL=0.0`).
- `gnn/` — Isolation Forest on microstructure features (not a graph network despite the name).

#### `services/execution/` ✅ Implemented
TypeScript Guardian + ExecutionWorkflow + per-mode executors. 63 tests (47 unit + 16 integration). See [`services/execution/README.md`](services/execution/README.md).

```
src/
  config/env.ts            — Zod-validated env loader (single source of truth for execution policy)
  guardian/                — 14 deterministic safety rules, first-failure-wins
  orchestrator/            — ExecutionWorkflow + ElizaNarrator (post-decision text)
  signalClient.ts          — Fetches signals from FastAPI /signal
  solana/                  — WalletService + PaperExecutor + DevnetExecutor + MainnetExecutor
                             (Jupiter v6 swap optional via JUPITER_SWAP_ENABLED)
  __tests__/integration/   — End-to-end risk-preference scenarios against the paper executor
```

#### `services/checkpoints/` — Saved model artifacts (gitignored payloads).

### `docs/`
- `docs/agentic_execution_architecture.md` — Execution-layer design.
- `docs/onchain_guardian_audit.md` — Notes towards the Phase 6 Anchor Guardian.
- `docs/adr/` — Architecture decision records.
- `docs/runbooks/` — Operational notes.

### `infra/`
- `infra/systemd/` — Orderbook-collector service template (only systemd file in the repo).
- `infra/docker`, `k8s`, `terraform`, `scripts` — Planned.

### `data/`
- `raw/` and `processed/` — Training / engineered CSVs.
- `model_artifacts/` — Stub directories; actual artifacts live next to each model in `services/models/`.

### `.claude/`
- `skills/run-app/SKILL.md` — Verified launch + smoke-test sequence; used when the user asks an assistant to "run the app."

### `.github/workflows/ci.yml`
Runs the Python test suites (classification, NLP, fusion, spoofing) and ruff linting. **Does not currently run the TypeScript execution tests, backend tests, or the FastAPI tests** — those have to be run locally.

---

## Production Pipeline Status

### Signal pipeline

```
Binance WS (SOL/USDT L2) + REST (OHLCV)
        │
        ▼
realtime_market_fetcher.py ─► latest_sol_15m.csv ─┐
realtime_btc_fetcher.py    ─► latest_btc_15m.csv ─┤
                                                  ▼
realtime_feature_builder.py  →  47 technical indicators
        │
        ├─ sentiment_status.json   (FinBERT daemon or inline)
        ├─ orderbook_status.json   (Binance WS daemon)
        │
        ▼
realtime_merge_features.py  →  latest_features_with_sentiment.csv
        │
        ▼
┌──────────────────────────────────────────────┐
│ XGBoost            ─► last_signal.json       │
│ Hawkes + TEN-GNN   ─► spoofing_score.json    │
│ Isolation-forest   ─► last_gnn_signal.json   │
│ PPO (weight 0.0)   ─► last_rl_signal.json    │
│              ↓                               │
│  realtime_fusion_aggregator.py               │
│              ↓                               │
│  last_fused_decision.json                    │
└──────────────────────────────────────────────┘
        │
        ▼
realtime_paper_trader.py + Python Guardian
        │
        ▼
paper_trading_log_fusion.json + paper_trading_history.json
        │
        ▼   (read by FastAPI /signal, /paper-trading, /history)
   Node backend (proxy + WS broadcast)
        │
        ▼
   React dashboard + AutoExecutor + TypeScript Guardian
```

### Execution pipeline (added in May 2026)

```
last_fused_decision.json
        │
        ▼  (SignalPoller polls every 10 s and emits 'signal_updated' on key change)
AutoExecutor (Node)
   ├─ reads auto_execute.json / user_risk_level.json / max_trade_size_sol.json
   ├─ reads emergency_stop.json / daily_volume.json
   ├─ checks SignalDedup (in-memory + executed_signal_keys.json on disk)
   └─ if WALLET_MODE=paper|system → workflow.execute()
   └─ if WALLET_MODE=phantom    → skip (frontend hook handles signing)
        │
        ▼
ExecutionWorkflow → Guardian → PaperExecutor / DevnetExecutor / MainnetExecutor
        │
        ▼
last_execution.json + execution_history.jsonl + WS broadcast → dashboard
```

### Model performance (latest run)

| Model | Metric | Value |
| --- | --- | --- |
| XGBoost H4 | Val AUC | 0.71 |
| XGBoost confidence gate | Sharpe-optimal threshold | 0.69 |
| TEN-GNN | Test F1 | 0.522 (P 0.603 / R 0.461) |
| TEN-GNN | Checkpoint | epoch 4, val_f1 0.369, threshold 0.51 |
| PPO | Training timesteps | 2 000 000 (3 model variants, disabled in fusion) |

### Test coverage

| Suite | Count | Status |
| --- | --- | --- |
| `services/execution` Guardian unit tests | 47 | ✅ |
| `services/execution` end-to-end integration tests | 16 | ✅ |
| `apps/trading-dashboard/backend` AutoExecutor flow | 9 | ✅ |
| Python classification + NLP + fusion + spoofing | (run locally / in CI) | ✅ |
| `apps/api/tests/` (FastAPI) | (not in CI) | local only |

---

## Key Principles

- **Paper-mode default.** `EXECUTION_MODE=paper`, `WALLET_MODE=phantom`, `ALLOW_MAINNET_EXECUTION=false`, `ALLOW_HIGH_RISK_TRADES=false`.
- **Guardian is the single safety boundary.** The LLM cannot trigger trades. The narrator only describes what the Guardian + executor already decided.
- **Chronological splits.** All train/val/test splits are time-ordered; no future leakage.
- **Honest status.** Components are marked working, partially-working, or planned. RL is explicitly trained-but-disabled, not falsely advertised as live.
- **Confidence gate.** Signals below `CONFIDENCE_THRESHOLD=0.69` emit NEUTRAL — never forced trades.
- **Spoofing override.** `spoofing_confidence > 0.7` forces HOLD regardless of other signals.
- **Phantom safety.** Backend cannot sign in `WALLET_MODE=phantom`. Auto-execution requires an explicit arm toggle in the browser and auto-disarms after each successful trade.

---

**Maintainer:** Nour Maatouk
**Last updated:** May 2026
