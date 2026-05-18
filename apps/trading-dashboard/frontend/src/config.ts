/** Active trading pair — must match SYMBOL in services/models/classification/config.py.
 *  Override at build time via VITE_ACTIVE_SYMBOL in .env */
export const ACTIVE_SYMBOL    = (import.meta.env.VITE_ACTIVE_SYMBOL as string | undefined)?.replace('/', '') ?? 'SOLUSDT';

/** Active chart timeframe — must match TIMEFRAME in classification/config.py */
export const ACTIVE_TIMEFRAME = '15m';

/** How often the signal hook polls the backend for updates (ms) */
export const SIGNAL_POLL_INTERVAL_MS = 15_000;

/** How often the Dashboard polls paper-trading metrics and history (ms) */
export const DASHBOARD_POLL_INTERVAL_MS = 30_000;

/** Number of recent pipeline runs shown in the Dashboard history table */
export const HISTORY_SHOW = 8;

/** Fallback confidence gate used when the API hasn't returned a value yet.
 *  Must match CONFIDENCE_THRESHOLD in services/models/classification/config.py */
export const DEFAULT_CONFIDENCE_GATE = 0.69;

/** HTTP fetch timeout for GET requests (ms) */
export const API_TIMEOUT_GET_MS = 8_000;

/** HTTP fetch timeout for POST requests (ms) */
export const API_TIMEOUT_POST_MS = 12_000;

/** WebSocket reconnect starting delay (ms) */
export const WS_RECONNECT_BASE_MS = 1_000;

/** WebSocket reconnect maximum delay (ms) */
export const WS_RECONNECT_MAX_MS = 30_000;
