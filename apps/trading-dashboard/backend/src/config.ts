export const ACTIVE_SYMBOL    = (process.env.ACTIVE_SYMBOL ?? 'SOLUSDT').replace('/', '');
export const ACTIVE_TIMEFRAME  = process.env.ACTIVE_TIMEFRAME  ?? '15m';
export const KLINES_LIMIT      = parseInt(process.env.KLINES_LIMIT ?? '500');

export const PYTHON_API        = process.env.PYTHON_API        ?? 'http://localhost:8000';
export const PYTHON_API_KEY    = (process.env.API_KEY ?? '').trim();
export const AUTH_HEADERS: Record<string, string> = PYTHON_API_KEY
  ? { 'X-API-Key': PYTHON_API_KEY }
  : {};

export const TIMEOUT_DEFAULT_MS  = 5_000;
export const TIMEOUT_PIPELINE_MS = 15_000;
export const TIMEOUT_KLINES_MS   = 20_000;
export const TIMEOUT_PREFS_MS    = 3_000;
export const TIMEOUT_LONG_MS     = 10_000;

export const MAX_TRADES              = 100;
export const MAX_BINANCE_RETRIES     = 3;
export const WS_RECONNECT_BASE_MS    = 1_000;
export const WS_RECONNECT_MAX_MS     = 30_000;
export const WS_KLINE_RETRY_BASE_MS  = 2_000;
export const SIGNAL_POLL_INTERVAL_MS = 10_000;
