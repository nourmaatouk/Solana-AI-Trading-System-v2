/**
 * Backend configuration — zod-validated environment loader.
 *
 * Replaces the earlier manual `numberEnv` helpers (which silently fell back to
 * defaults when a value was malformed, e.g. KLINES_LIMIT=abc → NaN → default).
 * With zod, a malformed env var fails loud at startup with a clear path/message
 * so misconfigurations don't become silent runtime surprises.
 *
 * All previously exported names and types are preserved. Default values match
 * the prior implementation exactly. Consumers do not need to change.
 */
import { z } from 'zod';

const positiveInt = z
  .string()
  .transform((v) => Number(v))
  .pipe(z.number().int().positive());

const nonNegInt = z
  .string()
  .transform((v) => Number(v))
  .pipe(z.number().int().nonnegative());

const csvList = z
  .string()
  .transform((s) =>
    s
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean),
  );

const ConfigSchema = z.object({
  ACTIVE_SYMBOL:     z.string().default('SOLUSDT'),
  ACTIVE_TIMEFRAME:  z.string().default('15m'),
  KLINES_LIMIT:      positiveInt.default('500'),
  WATCHLIST_SYMBOLS: csvList.default('BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,XRPUSDT'),
  QUOTE_ASSETS:      csvList.default('USDT,USDC,BTC,ETH,SOL'),

  PYTHON_API: z.string().url().default('http://localhost:8000'),
  API_KEY:    z.string().default(''),

  TIMEOUT_DEFAULT_MS:  positiveInt.default('5000'),
  TIMEOUT_PIPELINE_MS: positiveInt.default('15000'),
  TIMEOUT_KLINES_MS:   positiveInt.default('20000'),
  TIMEOUT_PREFS_MS:    positiveInt.default('3000'),
  TIMEOUT_LONG_MS:     positiveInt.default('10000'),

  MAX_TRADES:              positiveInt.default('100'),
  MAX_BINANCE_RETRIES:     nonNegInt.default('3'),
  KLINE_FETCH_MAX_RETRIES: nonNegInt.default('3'),
  WS_RECONNECT_BASE_MS:    positiveInt.default('1000'),
  WS_RECONNECT_MAX_MS:     positiveInt.default('30000'),
  WS_KLINE_RETRY_BASE_MS:  positiveInt.default('2000'),
  SIGNAL_POLL_INTERVAL_MS: positiveInt.default('10000'),

  // Solana / wallet
  SOLANA_CLUSTER: z.string().default('devnet'),

  // Devnet airdrop limits — validated here so a malformed env var fails loudly
  // at startup rather than silently falling back to a magic number.
  DEVNET_AIRDROP_AMOUNT_SOL: z
    .string()
    .transform((v) => parseFloat(v))
    .pipe(z.number().positive())
    .default('1'),
  MAX_DEVNET_AIRDROP_SOL: z
    .string()
    .transform((v) => parseFloat(v))
    .pipe(z.number().positive())
    .default('2'),
});

const parsed = ConfigSchema.safeParse(process.env);
if (!parsed.success) {
  const issues = parsed.error.issues
    .map((i) => `  - ${i.path.join('.')}: ${i.message}`)
    .join('\n');
  throw new Error(`Invalid backend environment:\n${issues}`);
}
const env = parsed.data;

const normaliseSymbol = (s: string): string => s.replace('/', '').toUpperCase();

export const ACTIVE_SYMBOL      = normaliseSymbol(env.ACTIVE_SYMBOL);
export const ACTIVE_TIMEFRAME   = env.ACTIVE_TIMEFRAME;
export const KLINES_LIMIT       = env.KLINES_LIMIT;
export const WATCHLIST_SYMBOLS  = env.WATCHLIST_SYMBOLS.map(normaliseSymbol);
export const QUOTE_ASSETS       = env.QUOTE_ASSETS.map((a) => a.toUpperCase());

export const PYTHON_API         = env.PYTHON_API;
export const PYTHON_API_KEY     = env.API_KEY.trim();
export const AUTH_HEADERS: Record<string, string> = PYTHON_API_KEY
  ? { 'X-API-Key': PYTHON_API_KEY }
  : {};

export const TIMEOUT_DEFAULT_MS  = env.TIMEOUT_DEFAULT_MS;
export const TIMEOUT_PIPELINE_MS = env.TIMEOUT_PIPELINE_MS;
export const TIMEOUT_KLINES_MS   = env.TIMEOUT_KLINES_MS;
export const TIMEOUT_PREFS_MS    = env.TIMEOUT_PREFS_MS;
export const TIMEOUT_LONG_MS     = env.TIMEOUT_LONG_MS;

export const MAX_TRADES              = env.MAX_TRADES;
export const MAX_BINANCE_RETRIES     = env.MAX_BINANCE_RETRIES;
export const KLINE_FETCH_MAX_RETRIES = env.KLINE_FETCH_MAX_RETRIES;
export const WS_RECONNECT_BASE_MS    = env.WS_RECONNECT_BASE_MS;
export const WS_RECONNECT_MAX_MS     = env.WS_RECONNECT_MAX_MS;
export const WS_KLINE_RETRY_BASE_MS  = env.WS_KLINE_RETRY_BASE_MS;
export const SIGNAL_POLL_INTERVAL_MS = env.SIGNAL_POLL_INTERVAL_MS;

export const SOLANA_CLUSTER              = env.SOLANA_CLUSTER;
export const DEVNET_AIRDROP_AMOUNT_SOL   = env.DEVNET_AIRDROP_AMOUNT_SOL;
export const MAX_DEVNET_AIRDROP_SOL      = env.MAX_DEVNET_AIRDROP_SOL;
