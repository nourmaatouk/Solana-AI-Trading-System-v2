const envString = (key: string, fallback: string): string => {
  const value = import.meta.env[key] as string | undefined;
  return value && value.trim() ? value.trim() : fallback;
};

const envNumber = (key: string, fallback: number): number => {
  const value = Number(import.meta.env[key]);
  return Number.isFinite(value) ? value : fallback;
};

const envList = (key: string, fallback: string[]): string[] => {
  const raw = import.meta.env[key] as string | undefined;
  if (!raw?.trim()) return fallback;
  const values = raw.split(',').map((item) => item.trim()).filter(Boolean);
  return values.length ? values : fallback;
};

const envNumberList = (key: string, fallback: number[]): number[] => {
  const raw = import.meta.env[key] as string | undefined;
  if (!raw?.trim()) return fallback;
  const values = raw
    .split(',')
    .map((item) => Number(item.trim()))
    .filter((value) => Number.isFinite(value) && value > 0);
  return values.length ? values : fallback;
};

const unique = <T,>(items: T[]): T[] => Array.from(new Set(items));

/** Active trading pair — must match SYMBOL in services/models/classification/config.py. */
export const ACTIVE_SYMBOL = envString('VITE_ACTIVE_SYMBOL', 'SOLUSDT').replace('/', '').toUpperCase();

/** Active chart timeframe — must match TIMEFRAME in classification/config.py. */
export const ACTIVE_TIMEFRAME = envString('VITE_ACTIVE_TIMEFRAME', '15m');

/** Secondary market used for correlation/co-feature display. */
export const CORRELATION_SYMBOL = envString('VITE_CORRELATION_SYMBOL', 'BTCUSDT').replace('/', '').toUpperCase();

/** User-selectable chart timeframes shown in the price chart toolbar. */
export const AVAILABLE_TIMEFRAMES = unique(envList('VITE_AVAILABLE_TIMEFRAMES', ['1m', '5m', ACTIVE_TIMEFRAME, '1h', '4h', '1D']));

/** How often the signal hook polls the backend for updates (ms) */
export const SIGNAL_POLL_INTERVAL_MS = envNumber('VITE_SIGNAL_POLL_INTERVAL_MS', 15_000);

/** How often the Dashboard polls paper-trading metrics and history (ms) */
export const DASHBOARD_POLL_INTERVAL_MS = envNumber('VITE_DASHBOARD_POLL_INTERVAL_MS', 30_000);

/** Number of recent pipeline runs shown in the Dashboard history table */
export const HISTORY_SHOW = envNumber('VITE_HISTORY_SHOW', 8);

/** Fallback confidence gate used when the API has not returned a value yet. */
export const DEFAULT_CONFIDENCE_GATE = envNumber('VITE_CONFIDENCE_GATE', 0.69);

/** Signal freshness fallback; backend remains the source of truth when present. */
export const SIGNAL_STALE_MINUTES = envNumber('VITE_SIGNAL_STALE_MINUTES', 20);

/** Spoofing severity thresholds; keep env values aligned with fusion/Guardian config. */
export const SPOOFING_HIGH = envNumber('VITE_SPOOFING_HIGH', 0.60);
export const SPOOFING_MEDIUM = envNumber('VITE_SPOOFING_MEDIUM', 0.30);

/** Guardian display/config values. */
export const GUARDIAN_SPOOFING_THRESHOLD = envNumber('VITE_GUARDIAN_SPOOFING_THRESHOLD', SPOOFING_HIGH);
export const GUARDIAN_SLIPPAGE_THRESHOLD = envNumber('VITE_GUARDIAN_SLIPPAGE_THRESHOLD', 0.005);
export const GUARDIAN_DAILY_LOSS_LIMIT = envNumber('VITE_GUARDIAN_DAILY_LOSS_LIMIT', 0.03);

/** Fusion display weights. RL is disabled unless explicitly changed upstream. */
export const FUSION_WEIGHT_CLASSIFICATION = envNumber('VITE_FUSION_WEIGHT_CLASSIFICATION', 0.50);
export const FUSION_WEIGHT_SENTIMENT = envNumber('VITE_FUSION_WEIGHT_SENTIMENT', 0.25);
export const FUSION_WEIGHT_SPOOFING = envNumber('VITE_FUSION_WEIGHT_SPOOFING', 0.20);
export const FUSION_WEIGHT_GNN = envNumber('VITE_FUSION_WEIGHT_GNN', 0.05);
export const FUSION_WEIGHT_RL = envNumber('VITE_FUSION_WEIGHT_RL', 0.00);
export const MODEL_FEATURE_COUNT = envNumber('VITE_MODEL_FEATURE_COUNT', 49);
export const BACKTEST_MODEL_LABEL = envString('VITE_BACKTEST_MODEL_LABEL', 'XGBoost H4');
export const RANDOM_BASELINE_AUC = envNumber('VITE_RANDOM_BASELINE_AUC', 0.5);
export const BACKTEST_ACCURACY_GOOD_THRESHOLD = envNumber('VITE_BACKTEST_ACCURACY_GOOD_THRESHOLD', 0.55);
export const BACKTEST_AUC_GOOD_THRESHOLD = envNumber('VITE_BACKTEST_AUC_GOOD_THRESHOLD', 0.55);
export const BACKTEST_MCC_GOOD_THRESHOLD = envNumber('VITE_BACKTEST_MCC_GOOD_THRESHOLD', 0.1);
export const BACKTEST_PRECISION_GOOD_THRESHOLD = envNumber('VITE_BACKTEST_PRECISION_GOOD_THRESHOLD', 0.65);
export const WALK_FORWARD_FOLDS = envNumber('VITE_WALK_FORWARD_FOLDS', 5);
export const PURGE_GAP_BARS = envNumber('VITE_PURGE_GAP_BARS', 4);
export const PURGE_GAP_LABEL = envString('VITE_PURGE_GAP_LABEL', '1 hour');
export const PAPER_TRADING_FEE_RATE = envNumber('VITE_PAPER_TRADING_FEE_RATE', 0.001);
export const TARGET_TAKE_PROFIT = envNumber('VITE_TARGET_TAKE_PROFIT', 0.02);
export const TARGET_STOP_LOSS = envNumber('VITE_TARGET_STOP_LOSS', 0.015);
export const ASSISTANT_SENTIMENT_THRESHOLD = envNumber('VITE_ASSISTANT_SENTIMENT_THRESHOLD', 0.1);

/** Market lists. */
export const WATCHLIST_SYMBOLS = unique(envList('VITE_WATCHLIST_SYMBOLS', [
  ACTIVE_SYMBOL,
  'BTCUSDT',
  'ETHUSDT',
  'BNBUSDT',
  'ADAUSDT',
  'XRPUSDT',
]).map((symbol) => symbol.replace('/', '').toUpperCase()));
export const CHART_SUPPORTED_SYMBOLS = new Set(
  unique(envList('VITE_CHART_SUPPORTED_SYMBOLS', [ACTIVE_SYMBOL]).map((symbol) => symbol.replace('/', '').toUpperCase())),
);
export const QUOTE_ASSETS = unique(envList('VITE_QUOTE_ASSETS', ['USDT', 'USDC', 'BTC', 'ETH', 'SOL']).map((asset) => asset.toUpperCase()));
export const CHART_KLINES_MAX = envNumber('VITE_CHART_KLINES_MAX', 500);
export const CHART_MA_PERIODS = envNumberList('VITE_CHART_MA_PERIODS', [7, 25, 99]);
export const CHART_MA_COLORS = envList('VITE_CHART_MA_COLORS', ['#8B5CF6', '#14B8A6', '#f6465d']);
export const RECENT_TRADES_MAX = envNumber('VITE_RECENT_TRADES_MAX', 100);
export const RECENT_TRADES_VISIBLE = envNumber('VITE_RECENT_TRADES_VISIBLE', 30);
export const ORDERBOOK_LEVELS = envNumber('VITE_ORDERBOOK_LEVELS', 15);
export const SPOOFING_ORDERBOOK_LEVELS = envNumber('VITE_SPOOFING_ORDERBOOK_LEVELS', 20);
export const ORDERBOOK_WALL_MULTIPLIER = envNumber('VITE_ORDERBOOK_WALL_MULTIPLIER', 3);
export const NEUTRAL_IMBALANCE_PCT = envNumber('VITE_NEUTRAL_IMBALANCE_PCT', 50);
export const IMBALANCE_BUY_PRESSURE_PCT = envNumber('VITE_IMBALANCE_BUY_PRESSURE_PCT', 60);
export const IMBALANCE_SELL_PRESSURE_PCT = envNumber('VITE_IMBALANCE_SELL_PRESSURE_PCT', 40);
export const CLOCK_TICK_INTERVAL_MS = envNumber('VITE_CLOCK_TICK_INTERVAL_MS', 1_000);
export const NOTIFICATION_HISTORY_LIMIT = envNumber('VITE_NOTIFICATION_HISTORY_LIMIT', 20);

/** HTTP fetch timeout for GET requests (ms) */
export const API_TIMEOUT_GET_MS = envNumber('VITE_API_TIMEOUT_GET_MS', 8_000);

/** HTTP fetch timeout for POST requests (ms) */
export const API_TIMEOUT_POST_MS = envNumber('VITE_API_TIMEOUT_POST_MS', 12_000);

/** WebSocket reconnect starting delay (ms) */
export const WS_RECONNECT_BASE_MS = envNumber('VITE_WS_RECONNECT_BASE_MS', 1_000);

/** WebSocket reconnect maximum delay (ms) */
export const WS_RECONNECT_MAX_MS = envNumber('VITE_WS_RECONNECT_MAX_MS', 30_000);

/** ExecutionPanel auto-refresh interval (ms) */
export const EXECUTION_PANEL_REFRESH_MS = envNumber('VITE_EXECUTION_PANEL_REFRESH_MS', 10_000);

/** Default amount (SOL) prefilled in the execution panel and used as the
 *  default maxTradeSizeSol in Settings. Must match EXECUTION_TRADE_DEFAULT_SOL
 *  in the backend .env (default 0.01). */
export const EXECUTION_TRADE_DEFAULT_SOL = envNumber('VITE_EXECUTION_TRADE_DEFAULT_SOL', 0.01);

/** Hard cap (SOL) shown in the Settings max-trade-size input.
 *  Must be ≤ MAX_TRADE_SIZE_SOL in the Guardian's .env (default 0.1). */
export const MAX_TRADE_SIZE_SOL = envNumber('VITE_MAX_TRADE_SIZE_SOL', 0.1);

/** Position-size caps shown next to each risk level in the Trade page.
 *  Keys must match RiskLevel; values are display strings only (UI label). */
export const RISK_LEVEL_CAPS: Record<'LOW' | 'MEDIUM' | 'HIGH', string> = {
  LOW:    envString('VITE_RISK_CAP_LOW',    '≤ 33%'),
  MEDIUM: envString('VITE_RISK_CAP_MEDIUM', '≤ 66%'),
  HIGH:   envString('VITE_RISK_CAP_HIGH',   '≤ 100%'),
};

// ── Wallet / Solana ──────────────────────────────────────────────────────────

/** Solana cluster the frontend targets. Must match the backend SOLANA_CLUSTER. */
export const SOLANA_CLUSTER = envString('VITE_SOLANA_CLUSTER', 'devnet');

/** RPC endpoint used by the wallet adapter Connection. */
export const SOLANA_RPC_URL = envString('VITE_SOLANA_RPC_URL', 'https://api.devnet.solana.com');

/** Whether Phantom/Solflare wallet support is enabled. */
export const ENABLE_PHANTOM_WALLET = envString('VITE_ENABLE_PHANTOM_WALLET', 'true') === 'true';

/**
 * Public key of the backend Devnet system wallet, used by the "Dev Wallet"
 * connect option. The backend GET /api/execution/status is the primary source;
 * this env value is only an offline fallback when the backend is unreachable.
 * Leave empty to require the backend to supply it (no fallback).
 */
export const SYSTEM_WALLET_PUBLIC_KEY = envString('VITE_SYSTEM_WALLET_PUBLIC_KEY', '');

/** Base URL of the Solana block explorer used for transaction links. */
export const SOLANA_EXPLORER_BASE_URL = envString('VITE_SOLANA_EXPLORER_BASE_URL', 'https://explorer.solana.com');

/** Official Phantom wallet download/install page (browser-detecting). */
export const PHANTOM_INSTALL_URL = envString('VITE_PHANTOM_INSTALL_URL', 'https://phantom.app/download');

/**
 * Wallet mode — controls whether the UI requires a connected browser wallet.
 * phantom : browser wallet required; publicKey sent with every execution request.
 * system  : backend uses its own keypair; no browser wallet UI shown.
 * paper   : no real wallet; paper-trading only.
 */
export const WALLET_MODE = envString('VITE_WALLET_MODE', 'phantom') as 'paper' | 'system' | 'phantom';

/** Backend URL (used for direct API calls outside the Vite proxy in production). */
export const BACKEND_URL = envString('VITE_BACKEND_URL', 'http://localhost:3001');
