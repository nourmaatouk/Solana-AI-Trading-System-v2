import { create } from 'zustand';
import type { TickerData, KlineData, TradeData, DepthData, WatchlistItem, AISignalData } from '../types';
import { ACTIVE_SYMBOL, ACTIVE_TIMEFRAME, CHART_KLINES_MAX, RECENT_TRADES_MAX } from '../config';

export interface ExecutionHistoryEntry {
  request_received_at_utc: string;
  decision: 'APPROVE' | 'BLOCK';
  blocked_by: string | null;
  reasons: string[];
  execution_status: string | null;
  mode: string | null;
  side: 'BUY' | 'SELL' | null;
  amount_sol: number | null;
  asset: string | null;
  signature: string | null;
  error: string | null;
}

const EXECUTION_HISTORY_MAX = 100 as const;

interface TradingStore {
  // Market data
  ticker: TickerData | null;
  klines: KlineData[];
  trades: TradeData[];
  depth: DepthData | null;
  watchlist: WatchlistItem[];
  connectionStatus: 'connected' | 'reconnecting' | 'disconnected';
  exchange: string;
  selectedSymbol: string;
  selectedInterval: string;

  // AI
  signal: AISignalData | null;

  // Execution history (newest first, max 100)
  executionHistory: ExecutionHistoryEntry[];

  // Theme
  theme: 'dark' | 'light';
  toggleTheme: () => void;

  // Alerts
  alertsEnabled: boolean;
  toggleAlerts: () => void;

  // Actions
  setTicker: (t: TickerData) => void;
  addKline: (k: KlineData) => void;
  setKlines: (ks: KlineData[]) => void;
  addTrade: (t: TradeData) => void;
  setTrades: (ts: TradeData[]) => void;
  setDepth: (d: DepthData) => void;
  setWatchlist: (w: WatchlistItem[]) => void;
  setConnectionStatus: (s: 'connected' | 'reconnecting' | 'disconnected', exchange: string) => void;
  setSignal: (s: AISignalData) => void;
  setSelectedSymbol: (s: string) => void;
  setSelectedInterval: (i: string) => void;
  setExecutionHistory: (entries: ExecutionHistoryEntry[]) => void;
  prependExecutionEntry: (entry: ExecutionHistoryEntry) => void;
}

export const useTradingStore = create<TradingStore>((set) => ({
  ticker: null,
  klines: [],
  trades: [],
  depth: null,
  watchlist: [],
  connectionStatus: 'reconnecting',
  exchange: 'binance',
  selectedSymbol: ACTIVE_SYMBOL,
  selectedInterval: ACTIVE_TIMEFRAME,
  signal: null,
  executionHistory: [],
  theme: localStorage.getItem('theme') === 'light' ? 'light' : 'dark',
  toggleTheme: () =>
    set((state) => {
      const next = state.theme === 'dark' ? 'light' : 'dark';
      localStorage.setItem('theme', next);
      document.documentElement.setAttribute('data-theme', next);
      return { theme: next };
    }),

  alertsEnabled: localStorage.getItem('alertsEnabled') !== 'false',
  toggleAlerts: () =>
    set((state) => {
      const next = !state.alertsEnabled;
      localStorage.setItem('alertsEnabled', String(next));
      return { alertsEnabled: next };
    }),

  setTicker: (t) => set({ ticker: t }),

  addKline: (k) =>
    set((state) => {
      const existing = state.klines;
      const last = existing[existing.length - 1];
      if (last && last.time === k.time) {
        const updated = [...existing];
        updated[updated.length - 1] = k;
        return { klines: updated };
      }
      const newKlines = [...existing, k];
      if (newKlines.length > CHART_KLINES_MAX) newKlines.shift();
      return { klines: newKlines };
    }),

  setKlines: (ks) => set({ klines: ks }),

  addTrade: (t) =>
    set((state) => {
      const updated = [t, ...state.trades];
      if (updated.length > RECENT_TRADES_MAX) updated.pop();
      return { trades: updated };
    }),

  setTrades: (ts) => set({ trades: ts }),

  setDepth: (d) => set({ depth: d }),

  setWatchlist: (w) => set({ watchlist: w }),

  setConnectionStatus: (s, exchange) => set({ connectionStatus: s, exchange }),

  setSignal: (s) => set({ signal: s }),

  setSelectedSymbol: (s) => set({ selectedSymbol: s }),

  setSelectedInterval: (i) => set({ selectedInterval: i }),

  setExecutionHistory: (entries) => set({ executionHistory: entries }),

  prependExecutionEntry: (entry) =>
    set((state) => {
      const updated = [entry, ...state.executionHistory];
      if (updated.length > EXECUTION_HISTORY_MAX) updated.pop();
      return { executionHistory: updated };
    }),
}));
