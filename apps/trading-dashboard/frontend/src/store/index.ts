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
  /** The fused model action the Guardian evaluated (BUY/SELL/HOLD) — present
   *  even on a blocked row, so the table can show the intended trade. */
  signal_action: 'BUY' | 'SELL' | 'HOLD' | null;
  side: 'BUY' | 'SELL' | null;
  amount_sol: number | null;
  asset: string | null;
  signature: string | null;
  error: string | null;
  signed_by: string | null;
}

const EXECUTION_HISTORY_MAX = 500 as const;

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

  // Trading mode — chosen on the Get Started screen, persisted across reloads.
  // 'paper'  = simulated trading, no wallet needed.
  // 'devnet' = real Solana test-net execution via a connected/backend wallet.
  mode: 'paper' | 'devnet';
  setMode: (m: 'paper' | 'devnet') => void;

  // Execution history (newest first, max 100)
  executionHistory: ExecutionHistoryEntry[];

  // Theme
  theme: 'dark' | 'light';
  toggleTheme: () => void;

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
  // Always start in paper mode. The wallet adapter clears its stored wallet name
  // on every page load (WalletProvider.tsx), so no wallet is ever auto-connected.
  // Any previously persisted 'devnet' value is therefore invalid on startup.
  // App.tsx upgrades back to devnet automatically once the wallet reconnects.
  mode: 'paper',
  setMode: (m) =>
    set(() => {
      localStorage.setItem('solvanta_mode', m);
      return { mode: m };
    }),
  executionHistory: [],
  theme: localStorage.getItem('theme') === 'light' ? 'light' : 'dark',
  toggleTheme: () =>
    set((state) => {
      const next = state.theme === 'dark' ? 'light' : 'dark';
      localStorage.setItem('theme', next);
      document.documentElement.setAttribute('data-theme', next);
      return { theme: next };
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
      // Idempotent: the same decision can be delivered more than once — via
      // both the execution_update event and a later execution_history_snapshot,
      // or over two WS connections. request_received_at_utc is ms-precise and
      // unique per decision, so use it to drop a row we already have.
      if (
        state.executionHistory.some(
          (e) => e.request_received_at_utc === entry.request_received_at_utc,
        )
      ) {
        return state;
      }
      const updated = [entry, ...state.executionHistory];
      if (updated.length > EXECUTION_HISTORY_MAX) updated.pop();
      return { executionHistory: updated };
    }),
}));
