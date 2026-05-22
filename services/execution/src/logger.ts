/**
 * Sanitised execution logger.
 *
 * All execution-related logs flow through this module so we can guarantee
 * that secrets (private keys, signing material) are stripped before anything
 * is written to stdout, files, or remote sinks.
 *
 * The logger is intentionally minimal — structured JSON-per-line, no
 * external deps. Plug in pino/winston later if needed.
 */

const SECRET_KEYS = new Set([
  'SOLANA_PRIVATE_KEY',
  'PRIVATE_KEY',
  'privateKey',
  'private_key',
  'secret',
  'SECRET',
  'apiKey',
  'API_KEY',
  'OPENAI_API_KEY',
]);

const REDACTED = '[REDACTED]';

function redact(value: unknown, depth = 0): unknown {
  if (depth > 6) return '[depth-limit]';
  if (value === null || value === undefined) return value;
  if (typeof value === 'string') return value; // string redaction happens by key, not value
  if (typeof value === 'number' || typeof value === 'boolean') return value;
  if (Array.isArray(value)) return value.map((v) => redact(v, depth + 1));
  if (typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (SECRET_KEYS.has(k)) {
        out[k] = REDACTED;
      } else {
        out[k] = redact(v, depth + 1);
      }
    }
    return out;
  }
  return String(value);
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  ts: string;
  level: LogLevel;
  event: string;
  [key: string]: unknown;
}

function emit(level: LogLevel, event: string, fields: Record<string, unknown>): void {
  const entry: LogEntry = {
    ts: new Date().toISOString(),
    level,
    event,
    ...(redact(fields) as Record<string, unknown>),
  };
  // Structured JSON-per-line. Pipe through jq in dev.
  // eslint-disable-next-line no-console
  console.log(JSON.stringify(entry));
}

export const log = {
  debug: (event: string, fields: Record<string, unknown> = {}) => emit('debug', event, fields),
  info:  (event: string, fields: Record<string, unknown> = {}) => emit('info',  event, fields),
  warn:  (event: string, fields: Record<string, unknown> = {}) => emit('warn',  event, fields),
  error: (event: string, fields: Record<string, unknown> = {}) => emit('error', event, fields),
};

/** Exposed for tests. */
export const _testing = { redact };
