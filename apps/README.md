# Apps Layer

User-facing applications for the Solana AI Trading System.

## Structure

| Directory | Status | Description |
|-----------|--------|-------------|
| `api/` | ✅ Implemented | FastAPI backend — monitoring + scoped control endpoints for the realtime pipeline |
| `trading-dashboard/` | ✅ Implemented | Node.js + React — live Binance WS, candlestick chart, order book, AI signals, paper trading |
| `agent/` | ✅ Implemented | ElizaOS "Sentinel" narration character. The runtime is embedded in `services/execution` (post-decision narration only). |

## What is not implemented

- **Agent autonomy** — the agent narrates decisions only. It has no plugins, no tool access, and cannot decide, approve, or execute trades (the Guardian + execution layer own that). There is no standalone memory or reasoning engine.
- **On-chain data ingestion** — no OpenBook/Helius connectors yet; market data comes from Binance (with OKX fallback) and the order-book WebSocket.
