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
  confidence: number;
  fused_score: number;
  risk_level: string;
  classification_signal?: string;
  classification_probability_up: number;
  classification_probability_down: number;
  spoofing_confidence: number;
  sentiment_relevance: number;
  reasoning?: {
    classification?: string;
    sentiment?: string;
    spoofing?: string;
  };
  reason: string;
  generated_at_utc: string;
  latest_close: number;
  signal_timestamp?: string;
  alignment_score?: number;
  classification_confidence?: number;
}
