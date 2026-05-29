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

export interface AISignalData {
  action?: string;
  signal?: string;
  /** abs(fused_score) — absent in the "no_signal" early-return */
  confidence?: number;
  /** Can be negative (bearish) or absent before pipeline runs */
  fused_score?: number;
  risk_level?: string;
  classification_signal?: string;
  /** Absent when aggregator has no classifier output yet */
  classification_probability_up?: number;
  classification_probability_down?: number;
  spoofing_confidence?: number;
  sentiment_relevance?: number;
  reasoning?: {
    classification?: string;
    sentiment?: string;
    spoofing?: string;
    rl?: string;
  };
  reason?: string;
  generated_at_utc?: string;
  /** Null when the pipeline has no close price available yet */
  latest_close?: number | null;
  signal_timestamp?: string;
  alignment_score?: number;
  classification_confidence?: number;
  confidence_gate?: number;
  rl_signal?: string;
  rl_confidence?: number;
  /** 'ppo' when PPO RL agent owns the action; 'fusion' for weighted ensemble;
   *  'fusion_rl_stale' when PPO signal was too old and fusion fell back. */
  decision_source?: string;
}
