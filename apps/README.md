# Apps Layer

User-facing applications for the Solana AI Trading System.

## Structure

| Directory | Status | Description |
|-----------|--------|-------------|
| `api/` | ✅ Implemented | FastAPI backend — read-only signal and status endpoints |
| `dashboard/` | ✅ Implemented | React/Vite SPA — signal panel, paper trading metrics |
| `trading-dashboard/` | ✅ Implemented | Node.js + React — live Binance WS, candlestick chart, order book |
| `agent/` | ⏳ Partial | Only `fusion_consumer.py` exists; ElizaOS orchestration not built |

## What is not implemented

- **SHAP visualizations** — the `Heatmaps/`, `ShapExplainer/`, and `LiveSignals/` component folders are stubs with no code.
- **Agent autonomy** — `apps/agent/` has no orchestrator, no plugin system, no memory layer, and no reasoning engine. The only working file is `fusion_consumer.py`.
- **WebSocket push** — dashboards use polling (15–30 s), not server-push WebSocket.
