# Solvanta AI — Trading Dashboard

Real-time crypto trading dashboard for the Solana AI Trading System. Streams live market data from Binance via WebSocket, overlays AI fusion signals from the XGBoost/FinBERT/TEN-GNN pipeline, and provides a paper-trading control panel.

---

## Architecture

```
Binance WS ──►  backend (Node/Express :3001)  ──WS──►  frontend (React/Vite :5174)
                        │
              Python FastAPI (:8000)  ──proxy──►  /api/ai/*
                        │
              AI JSON files (data/realtime/)
```

---

## Prerequisites

- Node.js >= 18
- Python FastAPI server running on port 8000 (`uvicorn apps.api.main:app --port 8000`)
- AI pipeline has run at least once (produces `last_fused_decision.json`)

---

## Setup

### Backend

```bash
cd apps/trading-dashboard/backend
npm install
npm run dev          # ts-node-dev hot reload
# or
npm run build && npm start
```

Listens on **port 3001**.

### Frontend

```bash
cd apps/trading-dashboard/frontend
npm install
npm run dev          # Vite dev server with HMR
# or
npm run build        # outputs to dist/
```

Listens on **port 5174** (proxies `/api` to `:3001`).

---

## Features

| Feature | Description |
|---------|-------------|
| Live price chart | Candlestick + volume + MA7/MA25/MA99 via `lightweight-charts` v4 |
| Order book | Real-time bid/ask depth with depth-bar visualization |
| Recent trades | Live trade tape from Binance WS |
| Watchlist | 6 pairs (SOL, BTC, ETH, BNB, ADA, XRP) |
| AI Signal panel | XGBoost direction + FinBERT sentiment + TEN-GNN spoofing |
| Spoofing panel | Order book imbalance gauge + wall detection |
| Trading controls | Risk level, mode selector, Generate Signal, Execute Paper Trade |
| OKX fallback | Automatic fallback if Binance WS fails after 3 retries |

---

## WebSocket Message Types (backend → frontend)

| Type | Description |
|------|-------------|
| `ticker` | 24h ticker snapshot |
| `kline` | Single candlestick update |
| `klines_snapshot` | Full kline history on connect |
| `trade` | Individual trade |
| `trades_snapshot` | Trade history on connect |
| `depth` | Order book depth (20 levels) |
| `watchlist` | All watched pairs aggregated |
| `connection` | Connection status + exchange |
| `signal` | AI fusion signal (every 10s) |

---

## API Endpoints (backend)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ai/health` | Python API health (proxy) |
| GET | `/api/ai/signal` | Raw signal (proxy) |
| GET | `/api/ai/fusion` | Full fusion JSON |
| GET | `/api/ai/prediction` | Prediction summary |
| GET | `/api/ai/spoofing` | TEN-GNN spoofing result |
| GET | `/api/ai/sentiment` | FinBERT sentiment |
| GET | `/api/ai/history` | Paper trading history |
| POST | `/api/ai/trigger` | Trigger pipeline run |
| GET | `/api/paper-trading` | Paper trading log |
| POST | `/api/paper-trading/execute` | Execute simulated trade |
| GET | `/api/market/status` | Binance WS connection status |

---

## Disclaimer

This dashboard is for **simulation and research purposes only**. No real trades are executed. All "paper trades" are simulated signals with no financial consequence.
