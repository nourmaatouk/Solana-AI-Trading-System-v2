/**
 * AutoExecutor — fires a trade when a new AI signal arrives and the user has
 * enabled automatic execution.
 *
 * Primary path: event-driven via SignalPoller's 'signal_updated' event so
 * execution happens within one poll cycle (~15 s) of a fresh signal rather
 * than waiting up to AUTO_EXECUTOR_POLL_MS (30 s).
 *
 * Fallback path: a configurable polling interval (AUTO_EXECUTOR_POLL_MS)
 * that re-reads the SignalPoller's last cached signal in case the event was
 * missed. CRITICAL: the fallback uses the same dedup key as the event path,
 * so a still-fresh signal is never executed twice across the two paths.
 *
 * Safety invariants (enforced here before calling workflow.execute):
 * - auto_execute.json must be enabled.
 * - Emergency stop blocks execution.
 * - Signal must be BUY or SELL (HOLD is skipped; Guardian also checks this).
 * - Duplicate signal keys are skipped for SIGNAL_DEDUP_WINDOW_SECONDS.
 * - Guardian runs on every call regardless — this class never bypasses it.
 * - WALLET_MODE=phantom means the backend cannot sign; execution is skipped
 *   here (the frontend hook handles Phantom signing in that case).
 * - The user's risk preference is loaded from the persistent user_risk_level
 *   file (set by the frontend Settings page); when absent the Guardian falls
 *   back to DEFAULT_USER_RISK_LEVEL from env.
 */
import { ExecutionWorkflow } from '@trading/execution';
import type { SignalPoller, FusedDecision } from './aiSignal.service';
import { computeSignalKey } from './aiSignal.service';
import {
  appendExecutedSignal,
  readAutoExecute,
  readDailyVolume,
  readEmergencyStop,
  readExecutedSignals,
  readUserMaxTradeSize,
  readUserRiskLevel,
  recordDailyVolume,
  recordLastExecution,
} from './executionState.service';

const POLL_INTERVAL_MS = parseInt(process.env.AUTO_EXECUTOR_POLL_MS ?? '30000', 10);
/** Fallback when the user hasn't set a per-trade size in Settings yet. */
const FALLBACK_AMOUNT_SOL = parseFloat(process.env.AUTO_EXECUTE_AMOUNT_SOL ?? '0.01');
const DEDUP_WINDOW_MS =
  parseInt(process.env.SIGNAL_DEDUP_WINDOW_SECONDS ?? '300', 10) * 1000;

// The backend cannot Phantom-sign. Skip auto-execution and let the frontend
// hook handle it when the wallet mode requires browser signing.
const WALLET_MODE = process.env.WALLET_MODE ?? 'phantom';

/**
 * Two-tier deduplication: an in-memory Map for fast access plus a
 * disk-backed log (executed_signal_keys.json) that survives crashes and
 * restarts. On boot the memory cache is seeded from disk so a recently-
 * executed signal cannot fire again after an unexpected restart.
 */
class SignalDedup {
  private readonly processed = new Map<string, number>(); // key → expiry ms

  constructor() {
    // Seed from disk so restarts don't reset the dedup window.
    const persisted = readExecutedSignals(DEDUP_WINDOW_MS);
    for (const entry of persisted) {
      const ts = Date.parse(entry.executed_at_utc);
      if (Number.isFinite(ts)) {
        this.processed.set(entry.signal_key, ts + DEDUP_WINDOW_MS);
      }
    }
    if (persisted.length > 0) {
      console.log(
        `[auto-executor] Loaded ${persisted.length} executed signal(s) from disk dedup log`,
      );
    }
  }

  has(key: string): boolean {
    this._cleanup();
    return this.processed.has(key);
  }

  /**
   * Records an executed signal in BOTH the in-memory cache and the on-disk
   * log so the dedup decision survives a crash/restart.
   */
  mark(key: string, record: { action: 'BUY' | 'SELL'; amount_sol: number }): void {
    const executedAt = new Date();
    this.processed.set(key, executedAt.getTime() + DEDUP_WINDOW_MS);
    try {
      appendExecutedSignal(
        {
          signal_key: key,
          executed_at_utc: executedAt.toISOString(),
          action: record.action,
          amount_sol: record.amount_sol,
        },
        DEDUP_WINDOW_MS,
      );
    } catch (err) {
      // Persistence failure is non-fatal — we still have the in-memory
      // record. Log so an operator can investigate.
      console.error('[auto-executor] failed to persist dedup record:', (err as Error).message);
    }
  }

  private _cleanup(): void {
    const now = Date.now();
    for (const [k, exp] of this.processed) {
      if (exp <= now) this.processed.delete(k);
    }
  }
}

export interface AutoExecutorStatus {
  enabled: boolean;
  walletMode: string;
  lastSignalKey: string | null;
  lastExecutedAt: string | null;
  lastSkippedReason: string | null;
  /** Duplicate-signal cooldown window (seconds) — a signal key can't re-fire within this. */
  dedupWindowSeconds: number;
  /** Fallback poll cadence (seconds) — the event path normally fires sooner. */
  pollIntervalSeconds: number;
}

export class AutoExecutor {
  private readonly workflow: ExecutionWorkflow;
  private readonly signalPoller: SignalPoller | null;
  private readonly dedup = new SignalDedup();
  private timer: ReturnType<typeof setInterval> | null = null;
  private running = false;
  private lastSignalKey: string | null = null;
  private lastExecutedAt: string | null = null;
  private lastSkippedReason: string | null = null;

  constructor(workflow?: ExecutionWorkflow, signalPoller?: SignalPoller) {
    this.workflow = workflow ?? new ExecutionWorkflow();
    this.signalPoller = signalPoller ?? null;

    if (signalPoller) {
      signalPoller.on('signal_updated', (signal: FusedDecision) => {
        void this._handleSignal(signal, 'event');
      });
    }
  }

  start(): void {
    if (this.timer) return;
    this.timer = setInterval(() => void this._pollTick(), POLL_INTERVAL_MS);
    console.log(
      `[auto-executor] Started — event-driven + poll fallback every ${POLL_INTERVAL_MS / 1000}s,` +
        ` dedup window ${DEDUP_WINDOW_MS / 1000}s, wallet=${WALLET_MODE}`,
    );
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  getStatus(): AutoExecutorStatus {
    return {
      enabled: readAutoExecute(),
      walletMode: WALLET_MODE,
      lastSignalKey: this.lastSignalKey,
      lastExecutedAt: this.lastExecutedAt,
      lastSkippedReason: this.lastSkippedReason,
      dedupWindowSeconds: DEDUP_WINDOW_MS / 1000,
      pollIntervalSeconds: POLL_INTERVAL_MS / 1000,
    };
  }

  private async _pollTick(): Promise<void> {
    // Fallback path: read the SignalPoller's cached signal and run the same
    // dedup + safety check chain as the event path. Without this, a missed
    // event could be re-executed every tick on a still-fresh signal.
    if (!this.signalPoller) return;
    const signal = this.signalPoller.getLatestSignal();
    if (!signal) return;
    await this._handleSignal(signal, 'poll');
  }

  private async _handleSignal(signal: FusedDecision, source: string): Promise<void> {
    if (this.running) {
      this._skip('previous execution still running');
      return;
    }
    if (!readAutoExecute()) return;

    const eStop = readEmergencyStop();
    if (eStop.enabled) {
      this._skip('emergency stop active');
      return;
    }

    if (WALLET_MODE === 'phantom') {
      this._skip('WALLET_MODE=phantom — Phantom signing handled by frontend');
      return;
    }

    const action = (signal.action ?? '').toUpperCase();
    if (action !== 'BUY' && action !== 'SELL') {
      this._skip(`signal action is ${action || 'missing'} — skipping`);
      return;
    }

    const key = computeSignalKey(signal);
    this.lastSignalKey = key;

    if (this.dedup.has(key)) {
      this._skip(`duplicate signal key ${key.slice(0, 30)}… (source=${source})`);
      return;
    }

    await this._execute(key, source);
  }

  private async _execute(key: string, source: string): Promise<void> {
    this.running = true;
    try {
      const daily_executed_sol = readDailyVolume();
      const user_risk_level = readUserRiskLevel();
      const amount_sol = readUserMaxTradeSize() ?? FALLBACK_AMOUNT_SOL;
      const result = await this.workflow.execute({
        amount_sol,
        daily_executed_sol,
        ...(user_risk_level ? { user_risk_level } : {}),
      });

      recordLastExecution(result);

      if (
        result.execution?.status === 'confirmed' ||
        result.execution?.status === 'simulated'
      ) {
        recordDailyVolume(result.execution.amount_sol);
      }

      // Persist the dedup record only when the trade actually went on-chain
      // (or was deliberately simulated in paper mode). Blocked or errored
      // signals are left out so a retry on the next poll is allowed.
      if (
        result.execution?.status === 'confirmed' ||
        result.execution?.status === 'simulated' ||
        result.execution?.status === 'submitted'
      ) {
        const side: 'BUY' | 'SELL' = result.execution.side;
        this.dedup.mark(key, { action: side, amount_sol: result.execution.amount_sol });
      }
      this.lastExecutedAt = new Date().toISOString();
      this.lastSkippedReason = null;

      const decision = result.guardian.decision;
      const blocked = result.guardian.blocked_by ?? 'none';
      const execStatus = result.execution?.status ?? 'none';
      console.log(
        `[auto-executor] ${source}: guardian=${decision} blocked_by=${blocked} exec=${execStatus}` +
          ` user_risk=${user_risk_level ?? 'default'} amount_sol=${amount_sol}`,
      );
    } catch (err) {
      console.error('[auto-executor] execution error:', (err as Error).message);
    } finally {
      this.running = false;
    }
  }

  private _skip(reason: string): void {
    this.lastSkippedReason = reason;
  }
}
