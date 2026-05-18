import { create } from 'zustand';
import type { TickerData, KlineData, TradeData, DepthData, WatchlistItem, AISignalData } from '../types';
import { ACTIVE_SYMBOL, ACTIVE_TIMEFRAME } from '../config';

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
  theme: (localStorage.getItem('theme') as 'dark' | 'light') ?? 'dark',
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
      if (newKlines.length > 500) newKlines.shift();
      return { klines: newKlines };
    }),

  setKlines: (ks) => set({ klines: ks }),

  addTrade: (t) =>
    set((state) => {
      const updated = [t, ...state.trades];
      if (updated.length > 100) updated.pop();
      return { trades: updated };
    }),

  setTrades: (ts) => set({ trades: ts }),

  setDepth: (d) => set({ depth: d }),

  setWatchlist: (w) => set({ watchlist: w }),

  setConnectionStatus: (s, exchange) => set({ connectionStatus: s, exchange }),

  setSignal: (s) => set({ signal: s }),

  setSelectedSymbol: (s) => set({ selectedSymbol: s }),

  setSelectedInterval: (i) => set({ selectedInterval: i }),
}));
