import 'dotenv/config';
import http from 'http';
import path from 'path';
import fs from 'fs';
import express from 'express';
import cors from 'cors';
import { WebSocketServer } from 'ws';

import { aiRouter } from './routes/ai.routes';
import { marketRouter } from './routes/market.routes';
import { paperTradingRouter } from './routes/paperTrading.routes';
import { executionRouter, registerAutoExecutor } from './routes/execution.routes';
import { guardianRouter } from './routes/guardian.routes';
import { walletRouter } from './routes/wallet.routes';
import { BinanceWebSocketService } from './services/binanceWebSocket.service';
import { SignalPoller } from './services/aiSignal.service';
import { AutoExecutor } from './services/autoExecutor.service';
import { executionEvents } from './services/executionEvents';
import { handleClientConnection } from './sockets/market.socket';
import { ACTIVE_SYMBOL, ACTIVE_TIMEFRAME, KLINES_LIMIT } from './config';

const PORT = process.env.PORT ? parseInt(process.env.PORT) : 3001;

const app = express();
app.use(cors());
app.use(express.json());

// ── HTTP server ───────────────────────────────────────────────────────────────
const server = http.createServer(app);

// ── WebSocket server ──────────────────────────────────────────────────────────
const wss = new WebSocketServer({ server, path: '/ws' });

// ── Binance WebSocket service ─────────────────────────────────────────────────
const binanceService = new BinanceWebSocketService(wss);

// Pre-load candles so the chart has history on first connect
binanceService.fetchHistoricalKlines(ACTIVE_SYMBOL, ACTIVE_TIMEFRAME, KLINES_LIMIT).then(() => {
  binanceService.connect();
}).catch(() => {
  binanceService.connect();
});

// ── Signal poller ─────────────────────────────────────────────────────────────
const signalPoller = new SignalPoller(wss);
signalPoller.start();

// ── Auto-executor ──────────────────────────────────────────────────────────────
const autoExecutor = new AutoExecutor(undefined, signalPoller);
autoExecutor.start();
// Expose the live AutoExecutor runtime to the /auto-execute/status route.
registerAutoExecutor(() => autoExecutor.getStatus());

// ── Execution event → WS broadcast ────────────────────────────────────────────
function broadcastToAll(payload: unknown): void {
  const msg = JSON.stringify(payload);
  wss.clients.forEach((client) => {
    if (client.readyState === 1 /* OPEN */) client.send(msg);
  });
}

executionEvents.on('execution_update', (entry) => {
  broadcastToAll({ type: 'execution_update', data: entry });
});

// ── Routes ─────────────────────────────────────────────────────────────────────
app.use('/api/ai', aiRouter);
app.use('/api/market', marketRouter(binanceService));
app.use('/api/paper-trading', paperTradingRouter);
app.use('/api/execution', executionRouter);
app.use('/api/guardian', guardianRouter);
app.use('/api/wallet', walletRouter);

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'trading-dashboard-backend', port: PORT });
});

// ── WS client handling ─────────────────────────────────────────────────────────
wss.on('connection', (ws) => {
  handleClientConnection(ws, binanceService, signalPoller);
});

// ── Static frontend (production build) ────────────────────────────────────────
// Resolved from this file's location, not process.cwd(), so it works whether
// the backend is launched from its own folder or the repo root. __dirname is
// backend/{src,dist}, so two levels up reaches the trading-dashboard folder.
const FRONTEND_DIST = path.resolve(__dirname, '..', '..', 'frontend', 'dist');
if (fs.existsSync(FRONTEND_DIST)) {
  app.use(express.static(FRONTEND_DIST));
  // SPA fallback: any non-API GET returns index.html
  app.get('*', (_req, res) => {
    res.sendFile(path.join(FRONTEND_DIST, 'index.html'));
  });
  console.log(`[server] Serving frontend from ${FRONTEND_DIST}`);
} else {
  console.log('[server] No frontend build found — run "npm run build:all" first');
}

// ── Start ──────────────────────────────────────────────────────────────────────
server.listen(PORT, '0.0.0.0', () => {
  console.log(`[server] Trading dashboard backend listening on port ${PORT}`);
  console.log(`[server] WebSocket at ws://localhost:${PORT}/ws`);
  if (fs.existsSync(FRONTEND_DIST)) {
    console.log(`[server] Dashboard → http://localhost:${PORT}/`);
  }
});
