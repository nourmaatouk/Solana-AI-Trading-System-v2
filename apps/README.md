# Apps Layer

User-facing applications for the Solana AI Trading System.

## Structure

| Directory | Status | Description |
|-----------|--------|-------------|
| `api/` | ✅ Implemented | FastAPI backend — read-only signal and status endpoints |
| `trading-dashboard/` | ✅ Implemented | Node.js + React — live Binance WS, candlestick chart, order book |
| `agent/` | ⏳ Empty stub | Reserved for the ElizaOS orchestration layer (Phase 5); no code yet |

## What is not implemented

- **Agent autonomy** — `apps/agent/` has no orchestrator, no plugin system, no memory layer, and no reasoning engine. Planned for Phase 5.
- **WebSocket push for signals** — the trading-dashboard has live market WS from Binance/OKX, but the AI signal layer still polls (10–30 s).
