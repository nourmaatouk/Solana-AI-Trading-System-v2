import WebSocket from 'ws';
import { BinanceWebSocketService } from '../services/binanceWebSocket.service';
import { SignalPoller } from '../services/aiSignal.service';

export function handleClientConnection(
  ws: WebSocket,
  binanceService: BinanceWebSocketService,
  signalPoller: SignalPoller,
): void {
  console.log('[ws] Client connected');

  // Send current cached state immediately on connect
  const ticker = binanceService.getLatestTicker();
  if (ticker) {
    ws.send(JSON.stringify({ type: 'ticker', symbol: 'SOLUSDT', data: ticker }));
  }

  const depth = binanceService.getLatestDepth();
  if (depth) {
    ws.send(JSON.stringify({ type: 'depth', symbol: 'SOLUSDT', data: depth }));
  }

  const klines = binanceService.getLatestKlines();
  if (klines.length > 0) {
    ws.send(JSON.stringify({ type: 'klines_snapshot', symbol: 'SOLUSDT', data: klines }));
  }

  const trades = binanceService.getLatestTrades();
  if (trades.length > 0) {
    ws.send(JSON.stringify({ type: 'trades_snapshot', symbol: 'SOLUSDT', data: trades }));
  }

  const signal = signalPoller.getLatestSignal();
  if (signal) {
    ws.send(JSON.stringify({ type: 'signal', data: signal }));
  }

  // Send connection status
  ws.send(JSON.stringify({
    type: 'connection',
    status: binanceService.getConnectionStatus(),
    exchange: binanceService.getExchange(),
  }));

  // Handle incoming client messages
  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw.toString()) as { type: string; symbol?: string; interval?: string };

      if (msg.type === 'subscribe') {
        // Client wants to change symbol/interval — acknowledged
        ws.send(JSON.stringify({
          type: 'subscribed',
          symbol: msg.symbol ?? 'SOLUSDT',
          interval: msg.interval ?? '15m',
        }));
      } else if (msg.type === 'changeInterval') {
        const interval = msg.interval ?? '15m';
        // Fetch REST history for the new interval, then reconnect WS stream to get live updates
        binanceService.fetchHistoricalKlines('SOLUSDT', interval, 500).then(() => {
          const snapshot = binanceService.getLatestKlines();
          ws.send(JSON.stringify({ type: 'klines_snapshot', symbol: 'SOLUSDT', data: snapshot }));
          binanceService.setKlineInterval(interval); // reconnect WS with new kline stream
        }).catch(() => {
          ws.send(JSON.stringify({ type: 'klines_snapshot', symbol: 'SOLUSDT', data: [] }));
        });
      }
    } catch { /* ignore malformed */ }
  });

  ws.on('close', () => {
    console.log('[ws] Client disconnected');
  });

  ws.on('error', (err) => {
    console.error('[ws] Client error:', err.message);
  });
}
