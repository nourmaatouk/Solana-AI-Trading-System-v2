# Dashboard

**Status: Implemented — Paper Trading Mode.**

React + Vite single-page application served directly from the FastAPI backend.

## Running

### Development (hot reload, Vite dev server)

```bash
cd apps/dashboard
npm install
npm run dev          # http://localhost:5173
```

API calls are proxied to `http://localhost:8000` automatically.

### Production (served from FastAPI)

```bash
# 1. Build
cd apps/dashboard
npm run build        # outputs to dist/

# 2. Start the API (serves dashboard at /app/)
cd ../..
uvicorn apps.api.main:app --port 8000
```

Dashboard URL: **http://localhost:8000/app/**

## Sections

| Section | Endpoint | Description |
|---------|----------|-------------|
| Header | `GET /health` | Backend status, Paper Trading badge |
| System Health | `GET /system`, `GET /status` | Orderbook, sentiment, pipeline freshness |
| Market Analysis | `GET /signal` | Price, XGBoost signal, prob bars, spoofing risk |
| Trading Decision | `GET /signal` | BUY/SELL/HOLD card, fused score, risk level |
| Paper Trading | `GET /paper-trading` | Equity, win rate, Sharpe, open position |
| Preferences | `GET/POST /preferences` | Asset, risk level, confidence threshold |
| Future Work | — | Disabled cards for wallet/execution/agent layer |

## What is not implemented

- Real wallet connection (planned Phase 6)
- On-chain Solana execution (planned Phase 6)
- Signal history chart / equity curve plot (next sprint)
- WebSocket live push (polling every 15–30 s currently)
- SHAP explainability heatmaps
