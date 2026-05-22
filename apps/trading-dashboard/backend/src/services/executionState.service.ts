/**
 * Lightweight execution-state service.
 *
 * - Persists emergency-stop status to a JSON file under the project's
 *   realtime data directory so the flag survives restarts.
 * - Records the last execution result so the dashboard can display "last
 *   trade attempted at HH:MM — result X".
 *
 * The Guardian also reads EMERGENCY_STOP from process.env at validation time
 * (defence in depth). This service writes the persistent flag AND updates
 * process.env so both layers stay in sync without a restart.
 */
import fs from 'fs';
import path from 'path';
import type { ExecutionWorkflowResult } from '@trading/execution';
import { executionEvents, type ExecutionHistoryEntry } from './executionEvents';

/**
 * Directory where every persistent execution-state file lives. Mirrors the
 * fallback used by aiSignal.service.ts so the Node backend and the signal
 * poller agree on the location. Override REALTIME_DIR in tests to redirect
 * all reads/writes to a temp directory.
 */
const REALTIME_DIR = process.env.REALTIME_DIR
  ?? path.resolve(
    __dirname,
    '..',
    '..',
    '..',
    '..',
    '..',
    'services',
    'models',
    'classification',
    'data',
    'realtime',
  );

const EMERGENCY_STOP_PATH  = path.join(REALTIME_DIR, 'emergency_stop.json');
const LAST_EXECUTION_PATH  = path.join(REALTIME_DIR, 'last_execution.json');
const DAILY_VOLUME_PATH    = path.join(REALTIME_DIR, 'daily_volume.json');
const AUTO_EXECUTE_PATH    = path.join(REALTIME_DIR, 'auto_execute.json');
const USER_RISK_LEVEL_PATH = path.join(REALTIME_DIR, 'user_risk_level.json');
const MAX_TRADE_SIZE_PATH  = path.join(REALTIME_DIR, 'max_trade_size_sol.json');
const EXECUTED_KEYS_PATH   = path.join(REALTIME_DIR, 'executed_signal_keys.json');
const EXECUTION_HISTORY_PATH = path.join(REALTIME_DIR, 'execution_history.jsonl');

export type UserRiskLevel = 'LOW' | 'MEDIUM' | 'HIGH';

const HISTORY_MAX_ENTRIES = 100;

interface EmergencyStopFile {
  enabled: boolean;
  updated_at_utc: string;
  reason: string | null;
}

function ensureDir(): void {
  if (!fs.existsSync(REALTIME_DIR)) {
    fs.mkdirSync(REALTIME_DIR, { recursive: true });
  }
}

export function readEmergencyStop(): EmergencyStopFile {
  // Persistent file takes precedence over env so toggles survive restart.
  if (fs.existsSync(EMERGENCY_STOP_PATH)) {
    try {
      return JSON.parse(fs.readFileSync(EMERGENCY_STOP_PATH, 'utf8'));
    } catch {
      /* fall through */
    }
  }
  const fromEnv = String(process.env.EMERGENCY_STOP ?? 'false').toLowerCase() === 'true';
  return {
    enabled: fromEnv,
    updated_at_utc: new Date().toISOString(),
    reason: fromEnv ? 'EMERGENCY_STOP=true in environment' : null,
  };
}

export function writeEmergencyStop(enabled: boolean, reason: string | null): EmergencyStopFile {
  ensureDir();
  const payload: EmergencyStopFile = {
    enabled,
    updated_at_utc: new Date().toISOString(),
    reason,
  };
  fs.writeFileSync(EMERGENCY_STOP_PATH, JSON.stringify(payload, null, 2));
  // Keep process.env in sync so the next Guardian.validate() picks it up
  // without restarting the backend.
  process.env.EMERGENCY_STOP = String(enabled);
  return payload;
}

export function recordLastExecution(result: ExecutionWorkflowResult): void {
  ensureDir();
  const slim = {
    request_received_at_utc: result.request_received_at_utc,
    request_completed_at_utc: result.request_completed_at_utc,
    decision: result.guardian.decision,
    blocked_by: result.guardian.blocked_by,
    reasons: result.guardian.reasons,
    sanitized_action: result.guardian.sanitized_action,
    execution: result.execution
      ? {
          status: result.execution.status,
          mode: result.execution.mode,
          signature: result.execution.signature,
          asset: result.execution.asset,
          side: result.execution.side,
          amount_sol: result.execution.amount_sol,
          submitted_at_utc: result.execution.submitted_at_utc,
          confirmed_at_utc: result.execution.confirmed_at_utc,
          error: result.execution.error,
        }
      : null,
    narration: result.narration,
  };
  fs.writeFileSync(LAST_EXECUTION_PATH, JSON.stringify(slim, null, 2));

  // Build the compact history entry and append it.
  const entry: ExecutionHistoryEntry = {
    request_received_at_utc: result.request_received_at_utc,
    decision: result.guardian.decision,
    blocked_by: result.guardian.blocked_by,
    reasons: result.guardian.reasons,
    execution_status: result.execution?.status ?? null,
    mode: result.execution?.mode ?? result.guardian.sanitized_action?.execution_mode ?? null,
    side: result.execution?.side ?? result.guardian.sanitized_action?.side ?? null,
    amount_sol: result.execution?.amount_sol ?? result.guardian.sanitized_action?.amount_sol ?? null,
    asset: result.execution?.asset ?? result.guardian.sanitized_action?.asset ?? null,
    signature: result.execution?.signature ?? null,
    error: result.execution?.error ?? null,
  };
  appendExecutionHistory(entry);
  executionEvents.emit('execution_update', entry);
}

function appendExecutionHistory(entry: ExecutionHistoryEntry): void {
  let lines: string[] = [];
  if (fs.existsSync(EXECUTION_HISTORY_PATH)) {
    try {
      lines = fs.readFileSync(EXECUTION_HISTORY_PATH, 'utf8')
        .split('\n')
        .filter((l) => l.trim().length > 0);
    } catch { /* start fresh */ }
  }
  lines.push(JSON.stringify(entry));
  // Keep only the last N entries.
  if (lines.length > HISTORY_MAX_ENTRIES) {
    lines = lines.slice(lines.length - HISTORY_MAX_ENTRIES);
  }
  fs.writeFileSync(EXECUTION_HISTORY_PATH, lines.join('\n') + '\n');
}

export function readExecutionHistory(): ExecutionHistoryEntry[] {
  if (!fs.existsSync(EXECUTION_HISTORY_PATH)) return [];
  try {
    return fs.readFileSync(EXECUTION_HISTORY_PATH, 'utf8')
      .split('\n')
      .filter((l) => l.trim().length > 0)
      .map((l) => JSON.parse(l) as ExecutionHistoryEntry)
      .reverse(); // newest first
  } catch {
    return [];
  }
}

export function readLastExecution(): unknown | null {
  if (!fs.existsSync(LAST_EXECUTION_PATH)) return null;
  try {
    return JSON.parse(fs.readFileSync(LAST_EXECUTION_PATH, 'utf8'));
  } catch {
    return null;
  }
}

// ── Daily volume tracker ─────────────────────────────────────────────────────

interface DailyVolumeFile {
  date: string;   // YYYY-MM-DD UTC
  total_sol: number;
}

function todayUtc(): string {
  return new Date().toISOString().slice(0, 10);
}

/** Total SOL executed in confirmed trades today (UTC day). Resets at midnight. */
export function readDailyVolume(): number {
  if (!fs.existsSync(DAILY_VOLUME_PATH)) return 0;
  try {
    const data: DailyVolumeFile = JSON.parse(fs.readFileSync(DAILY_VOLUME_PATH, 'utf8'));
    return data.date === todayUtc() ? data.total_sol : 0;
  } catch {
    return 0;
  }
}

/** Add `amountSol` to today's running total. Call after a confirmed execution. */
export function recordDailyVolume(amountSol: number): void {
  ensureDir();
  const today = todayUtc();
  const current = readDailyVolume();
  const payload: DailyVolumeFile = { date: today, total_sol: current + amountSol };
  fs.writeFileSync(DAILY_VOLUME_PATH, JSON.stringify(payload, null, 2));
}

// ── Auto-execute toggle ──────────────────────────────────────────────────────

interface AutoExecuteFile {
  enabled: boolean;
  updated_at_utc: string;
}

export function readAutoExecute(): boolean {
  if (!fs.existsSync(AUTO_EXECUTE_PATH)) return false;
  try {
    const data: AutoExecuteFile = JSON.parse(fs.readFileSync(AUTO_EXECUTE_PATH, 'utf8'));
    return data.enabled;
  } catch {
    return false;
  }
}

export function writeAutoExecute(enabled: boolean): AutoExecuteFile {
  ensureDir();
  const payload: AutoExecuteFile = { enabled, updated_at_utc: new Date().toISOString() };
  fs.writeFileSync(AUTO_EXECUTE_PATH, JSON.stringify(payload, null, 2));
  return payload;
}

// ── User risk preference ─────────────────────────────────────────────────────
// Persisted server-side so the backend AutoExecutor (which has no browser
// localStorage access) can enforce the user's risk tolerance. Set by the
// frontend Settings page via POST /api/execution/user-risk-level.

interface UserRiskLevelFile {
  level: UserRiskLevel;
  updated_at_utc: string;
}

/** Returns the persisted user risk preference, or null when none has been set. */
export function readUserRiskLevel(): UserRiskLevel | null {
  if (!fs.existsSync(USER_RISK_LEVEL_PATH)) return null;
  try {
    const data: UserRiskLevelFile = JSON.parse(fs.readFileSync(USER_RISK_LEVEL_PATH, 'utf8'));
    if (data.level === 'LOW' || data.level === 'MEDIUM' || data.level === 'HIGH') {
      return data.level;
    }
    return null;
  } catch {
    return null;
  }
}

export function writeUserRiskLevel(level: UserRiskLevel): UserRiskLevelFile {
  ensureDir();
  const payload: UserRiskLevelFile = { level, updated_at_utc: new Date().toISOString() };
  fs.writeFileSync(USER_RISK_LEVEL_PATH, JSON.stringify(payload, null, 2));
  return payload;
}

// ── User max trade size ──────────────────────────────────────────────────────
// The Guardian's MAX_TRADE_SIZE_SOL env var is the hard ceiling. This file
// lets the user pick a smaller per-trade size in Settings that the backend
// AutoExecutor honors. Guardian still BLOCKs anything above the env ceiling.

interface MaxTradeSizeFile {
  amount_sol: number;
  updated_at_utc: string;
}

/** Returns the persisted user max-trade-size in SOL, or null when none set. */
export function readUserMaxTradeSize(): number | null {
  if (!fs.existsSync(MAX_TRADE_SIZE_PATH)) return null;
  try {
    const data: MaxTradeSizeFile = JSON.parse(fs.readFileSync(MAX_TRADE_SIZE_PATH, 'utf8'));
    if (typeof data.amount_sol === 'number' && Number.isFinite(data.amount_sol) && data.amount_sol > 0) {
      return data.amount_sol;
    }
    return null;
  } catch {
    return null;
  }
}

export function writeUserMaxTradeSize(amountSol: number): MaxTradeSizeFile {
  ensureDir();
  const payload: MaxTradeSizeFile = {
    amount_sol: amountSol,
    updated_at_utc: new Date().toISOString(),
  };
  fs.writeFileSync(MAX_TRADE_SIZE_PATH, JSON.stringify(payload, null, 2));
  return payload;
}

// ── Persistent executed-signal dedup ─────────────────────────────────────────
// Stored on disk so a backend crash + restart inside the dedup window cannot
// cause the same signal to fire twice. Entries older than the window are
// pruned on read.

export interface ExecutedSignalRecord {
  signal_key: string;
  executed_at_utc: string;
  action: 'BUY' | 'SELL';
  amount_sol: number;
}

interface ExecutedSignalsFile {
  entries: ExecutedSignalRecord[];
}

/** Returns the persisted executed-signal records that are still within the
 *  given TTL window. Entries outside the window are not returned (and will
 *  be pruned on the next write). */
export function readExecutedSignals(ttlMs: number): ExecutedSignalRecord[] {
  if (!fs.existsSync(EXECUTED_KEYS_PATH)) return [];
  try {
    const data: ExecutedSignalsFile = JSON.parse(fs.readFileSync(EXECUTED_KEYS_PATH, 'utf8'));
    if (!Array.isArray(data.entries)) return [];
    const cutoff = Date.now() - ttlMs;
    return data.entries.filter((e) => {
      const ts = Date.parse(e.executed_at_utc);
      return Number.isFinite(ts) && ts >= cutoff;
    });
  } catch {
    return [];
  }
}

/** Appends a new executed-signal record and prunes anything outside the TTL.
 *  Returns the persisted (still-fresh) list. */
export function appendExecutedSignal(
  record: ExecutedSignalRecord,
  ttlMs: number,
): ExecutedSignalRecord[] {
  ensureDir();
  const fresh = readExecutedSignals(ttlMs);
  // Replace any prior entry with the same key to keep the file compact.
  const next = fresh.filter((e) => e.signal_key !== record.signal_key);
  next.push(record);
  const payload: ExecutedSignalsFile = { entries: next };
  fs.writeFileSync(EXECUTED_KEYS_PATH, JSON.stringify(payload, null, 2));
  return next;
}
