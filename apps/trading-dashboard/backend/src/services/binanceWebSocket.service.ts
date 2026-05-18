import WebSocket, { WebSocketServer } from 'ws';
import {
  ACTIVE_SYMBOL, ACTIVE_TIMEFRAME, KLINES_LIMIT, TIMEOUT_KLINES_MS,
  MAX_TRADES, MAX_BINANCE_RETRIES, WS_RECONNECT_BASE_MS, WS_RECONNECT_MAX_MS,
  WS_KLINE_RETRY_BASE_MS,
} from '../config';

// ── Interfaces ────────────────────────────────────────────────────────────────

export interface TickerData {
  symbol: string;
  price: string;
  change: string;
  changePct: string;
  high: string;
  low: string;
  volume: string;
  quoteVolume: string;
}

export interface KlineData {
  time: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  isClosed: boolean;
}

export interface TradeData {
  id: number;
  price: string;
  qty: string;
  time: number;
  isBuyerMaker: boolean;
}

export interface DepthData {
  bids: [string, string][];
  asks: [string, string][];
}

export interface WatchlistItem {
  symbol: string;
  price: string;
  change: string;
  changePct: string;
  volume: string;
}

// ── Binance message shape types ────────────────────────────────────────────────

interface BinanceCombinedMsg {
  stream: string;
  data: BinanceTickerMsg | BinanceKlineMsg | BinanceTradeMsg | BinanceDepthMsg;
}

interface BinanceTickerMsg {
  e: '24hrTicker';
  s: string;
  c: string;
  P: string;
  p: string;
  h: string;
  l: string;
  v: string;
  q: string;
}

interface BinanceKlineMsg {
  e: 'kline';
  s: string;
  k: {
    t: number; o: string; h: string; l: string; c: string; v: string; x: boolean;
  };
}

interface BinanceTradeMsg {
  e: 'trade';
  s: string;
  t: number;
  p: string;
  q: string;
  T: number;
  m: boolean;
}

// depth20 partial-book stream uses bids/asks; diff stream uses b/a
interface BinanceDepthMsg {
  e?: 'depthUpdate';
  b?: [string, string][];   // diff stream
  a?: [string, string][];   // diff stream
  bids?: [string, string][]; // partial-book depth20 stream
  asks?: [string, string][]; // partial-book depth20 stream
}

// ── Service ───────────────────────────────────────────────────────────────────

const BINANCE_WS_HOST   = process.env.BINANCE_WS_HOST   ?? 'wss://stream.binance.com:9443';
const BINANCE_REST_HOST = process.env.BINANCE_REST_HOST ?? 'https://api.binance.com';
const OKX_WS_URL        = process.env.OKX_WS_URL        ?? 'wss://ws.okx.com:8443/ws/v5/public';

// Active trading symbol (no slash, lowercase) — e.g. "solusdt"
const ACTIVE_STREAM_SYMBOL = (process.env.ACTIVE_SYMBOL ?? 'SOLUSDT').replace('/', '').toLowerCase();

// Watchlist symbols that get a ticker stream (comma-separated, no slash, lowercase)
const WATCHLIST_STREAM_SYMBOLS = (process.env.WATCHLIST_SYMBOLS ?? 'btcusdt,ethusdt,bnbusdt,adausdt,xrpusdt')
  .split(',')
  .map(s => s.trim().toLowerCase())
  .filter(s => s && s !== ACTIVE_STREAM_SYMBOL);

function buildBinanceUrl(klineInterval: string): string {
  const streams = [
    `${ACTIVE_STREAM_SYMBOL}@ticker`,
    `${ACTIVE_STREAM_SYMBOL}@kline_${klineInterval}`,
    `${ACTIVE_STREAM_SYMBOL}@trade`,
    `${ACTIVE_STREAM_SYMBOL}@depth20@100ms`,
    ...WATCHLIST_STREAM_SYMBOLS.map(s => `${s}@ticker`),
  ].join('/');
  return `${BINANCE_WS_HOST}/stream?streams=${streams}`;
}


export class BinanceWebSocketService {
  private wss: WebSocketServer;
  private ws: WebSocket | null = null;

  private latestTicker: TickerData | null = null;
  private latestDepth: DepthData | null = null;
  private klines: KlineData[] = [];
  private trades: TradeData[] = [];
  private watchlist: Map<string, WatchlistItem> = new Map();

  private connectionStatus: 'connected' | 'reconnecting' | 'disconnected' = 'disconnected';
  private exchange = 'binance';
  private reconnectDelay = WS_RECONNECT_BASE_MS;
  private binanceFailCount = 0;
  private destroyed = false;
  private selectedInterval = ACTIVE_TIMEFRAME;

  constructor(wss: WebSocketServer) {
    this.wss = wss;
  }

  connect(): void {
    if (this.connectionStatus === 'disconnected') {
      this.connectionStatus = 'reconnecting';
    }
    if (this.binanceFailCount < MAX_BINANCE_RETRIES) {
      this.connectBinance();
    } else {
      this.connectOkx();
    }
  }

  async fetchHistoricalKlines(symbol: string, interval: string, limit = 500): Promise<boolean> {
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        const url = `${BINANCE_REST_HOST}/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
        const res = await fetch(url, { signal: AbortSignal.timeout(TIMEOUT_KLINES_MS) });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const raw = await res.json() as [number, string, string, string, string, string, ...unknown[]][];
        this.klines = raw.map((k) => ({
          time: Math.floor(k[0] / 1000),
          open: k[1],
          high: k[2],
          low: k[3],
          close: k[4],
          volume: k[5],
          isClosed: true,
        }));
        console.log(`[binance] Loaded ${this.klines.length} historical ${interval} klines for ${symbol}`);
        return true;
      } catch (err) {
        console.warn(`[binance] Kline fetch attempt ${attempt}/3 failed: ${(err as Error).message}`);
        if (attempt < 3) await new Promise((r) => setTimeout(r, WS_KLINE_RETRY_BASE_MS * attempt));
      }
    }
    console.error('[binance] All kline fetch attempts failed');
    return false;
  }

  private connectBinance(): void {
    const url = buildBinanceUrl(this.selectedInterval);
    console.log(`[binance] Connecting with kline interval: ${this.selectedInterval}`);
    this.exchange = 'binance';
    this.ws = new WebSocket(url);
    this.attachHandlers();
  }

  setKlineInterval(interval: string): void {
    if (this.selectedInterval === interval) return;
    console.log(`[binance] Switching kline interval: ${this.selectedInterval} → ${interval}`);
    this.selectedInterval = interval;
    this.binanceFailCount = 0; // reset so reconnect goes to Binance, not OKX
    this.reconnectDelay = WS_RECONNECT_BASE_MS;
    this.ws?.close();            // triggers scheduleReconnect → connectBinance with new interval
  }

  private connectOkx(): void {
    console.log('[okx] Falling back to OKX WebSocket...');
    this.exchange = 'okx';
    this.ws = new WebSocket(OKX_WS_URL);

    this.ws.once('open', () => {
      const okxInstId = ACTIVE_STREAM_SYMBOL.toUpperCase().replace('USDT', '-USDT');
      console.log(`[okx] Connected — subscribing to ${okxInstId} ticker`);
      this.ws?.send(JSON.stringify({
        op: 'subscribe',
        args: [{ channel: 'tickers', instId: okxInstId }],
      }));
      this.connectionStatus = 'connected';
      this.reconnectDelay = WS_RECONNECT_BASE_MS;
      this.broadcast({ type: 'connection', status: 'connected', exchange: 'okx' });
      // OKX has no kline stream — preserve chart history from the last Binance fetch
      if (this.klines.length > 0) {
        this.broadcast({ type: 'klines_snapshot', symbol: ACTIVE_SYMBOL, data: this.klines });
      }
    });

    this.ws.on('message', (raw) => {
      try {
        const msg = JSON.parse(raw.toString()) as { event?: string; arg?: { channel: string }; data?: Array<{ instId: string; last: string; open24h: string; high24h: string; low24h: string; vol24h: string; volCcy24h: string; change24h: string }> };
        if (msg.event === 'subscribe') return;
        if (msg.arg?.channel === 'tickers' && msg.data) {
          const d = msg.data[0];
          if (!d) return;
          const changePct = parseFloat(d.change24h ?? '0') * 100;
          const ticker: TickerData = {
            symbol: ACTIVE_SYMBOL,
            price: d.last,
            change: String((parseFloat(d.last) - parseFloat(d.open24h)).toFixed(4)),
            changePct: changePct.toFixed(2),
            high: d.high24h,
            low: d.low24h,
            volume: d.vol24h,
            quoteVolume: d.volCcy24h,
          };
          this.latestTicker = ticker;
          this.broadcast({ type: 'ticker', symbol: ACTIVE_SYMBOL, data: ticker });
          this.updateWatchlist(ACTIVE_SYMBOL, ticker);
          this.broadcastWatchlist();
        }
      } catch (err) { console.debug('[okx] message parse error:', err); }
    });

    this.ws.on('error', (err) => console.error('[okx] WS error:', err.message));
    this.ws.on('close', () => {
      if (!this.destroyed) this.scheduleReconnect();
    });
  }

  private attachHandlers(): void {
    if (!this.ws) return;

    this.ws.on('open', () => {
      console.log('[binance] Connected');
      this.connectionStatus = 'connected';
      this.reconnectDelay = WS_RECONNECT_BASE_MS;
      this.broadcast({ type: 'connection', status: 'connected', exchange: 'binance' });

      if (this.klines.length === 0) {
        // Startup REST pre-fetch failed — backfill now via the live connection
        this.fetchHistoricalKlines(ACTIVE_SYMBOL, this.selectedInterval, KLINES_LIMIT).then((ok) => {
          if (ok) {
            this.broadcast({ type: 'klines_snapshot', symbol: ACTIVE_SYMBOL, data: this.klines });
          }
        });
      } else {
        // Klines already loaded (pre-fetch succeeded or interval change).
        // Broadcast to any client that connected before the WS handshake finished.
        this.broadcast({ type: 'klines_snapshot', symbol: ACTIVE_SYMBOL, data: this.klines });
      }
    });

    this.ws.on('message', (raw) => {
      try {
        const msg = JSON.parse(raw.toString()) as BinanceCombinedMsg;
        this.handleBinanceMessage(msg);
      } catch (err) { console.debug('[binance] message parse error:', err); }
    });

    this.ws.on('error', (err) => {
      console.error('[binance] WS error:', err.message);
    });

    this.ws.on('close', (code) => {
      console.warn(`[binance] WS closed (code=${code})`);
      if (!this.destroyed) {
        this.binanceFailCount++;
        this.scheduleReconnect();
      }
    });
  }

  private handleBinanceMessage(msg: BinanceCombinedMsg): void {
    const { stream, data } = msg;
    if (!stream || !data) return;

    if (stream.endsWith('@ticker')) {
      this.handleTicker(data as BinanceTickerMsg);
    } else if (stream.includes('@kline_')) {
      this.handleKline(data as BinanceKlineMsg);
    } else if (stream.endsWith('@trade')) {
      this.handleTrade(data as BinanceTradeMsg);
    } else if (stream.includes('@depth')) {
      this.handleDepth(data as BinanceDepthMsg);
    }
  }

  private handleTicker(d: BinanceTickerMsg): void {
    const ticker: TickerData = {
      symbol: d.s,
      price: d.c,
      change: d.p,
      changePct: d.P,
      high: d.h,
      low: d.l,
      volume: d.v,
      quoteVolume: d.q,
    };

    if (d.s === ACTIVE_SYMBOL) {
      this.latestTicker = ticker;
      this.broadcast({ type: 'ticker', symbol: d.s, data: ticker });
    }

    this.updateWatchlist(d.s, ticker);
    this.broadcastWatchlist();
  }

  private handleKline(d: BinanceKlineMsg): void {
    const k = d.k;
    const kline: KlineData = {
      time: Math.floor(k.t / 1000),
      open: k.o,
      high: k.h,
      low: k.l,
      close: k.c,
      volume: k.v,
      isClosed: k.x,
    };

    // Update or append kline
    const last = this.klines[this.klines.length - 1];
    if (last && last.time === kline.time) {
      this.klines[this.klines.length - 1] = kline;
    } else {
      this.klines.push(kline);
      if (this.klines.length > KLINES_LIMIT) {
        this.klines.shift();
      }
    }

    this.broadcast({ type: 'kline', symbol: d.s, interval: this.selectedInterval, data: kline });
  }

  private handleTrade(d: BinanceTradeMsg): void {
    const trade: TradeData = {
      id: d.t,
      price: d.p,
      qty: d.q,
      time: d.T,
      isBuyerMaker: d.m,
    };

    this.trades.unshift(trade);
    if (this.trades.length > MAX_TRADES) {
      this.trades.pop();
    }

    this.broadcast({ type: 'trade', symbol: d.s, data: trade });
  }

  private handleDepth(d: BinanceDepthMsg): void {
    const bids = d.bids ?? d.b;
    const asks = d.asks ?? d.a;
    if (!bids || !asks) return;
    this.latestDepth = { bids, asks };
    this.broadcast({ type: 'depth', symbol: ACTIVE_SYMBOL, data: this.latestDepth });
  }

  private updateWatchlist(symbol: string, ticker: TickerData): void {
    this.watchlist.set(symbol, {
      symbol,
      price: ticker.price,
      change: ticker.change,
      changePct: ticker.changePct,
      volume: ticker.volume,
    });
  }

  private broadcastWatchlist(): void {
    const items = Array.from(this.watchlist.values());
    this.broadcast({ type: 'watchlist', data: items });
  }

  private scheduleReconnect(): void {
    this.connectionStatus = 'reconnecting';
    this.broadcast({ type: 'connection', status: 'reconnecting', exchange: this.exchange });

    console.log(`[ws] Reconnecting in ${this.reconnectDelay}ms...`);
    setTimeout(() => {
      if (!this.destroyed) {
        this.connect();
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, WS_RECONNECT_MAX_MS);
      }
    }, this.reconnectDelay);
  }

  broadcast(payload: unknown): void {
    const msg = JSON.stringify(payload);
    this.wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(msg);
      }
    });
  }

  destroy(): void {
    this.destroyed = true;
    this.ws?.close();
  }

  // ── Getters ─────────────────────────────────────────────────────────────────

  getLatestTicker(): TickerData | null { return this.latestTicker; }
  getLatestDepth(): DepthData | null { return this.latestDepth; }
  getLatestKlines(): KlineData[] { return this.klines; }
  getLatestTrades(): TradeData[] { return this.trades; }
  getConnectionStatus(): string { return this.connectionStatus; }
  getExchange(): string { return this.exchange; }
}
