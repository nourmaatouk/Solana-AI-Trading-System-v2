/**
 * Orchestrator types — the public shape that the Node backend consumes.
 */
import type { ExecutionMode, GuardianResult, RiskLevel, TradeAction } from '../guardian/types';
import type { ExecutionResult } from '../solana/types';
import type { NormalisedSignal } from '../signalClient';

export interface ExecutionRequest {
  /** Optional override. Default = current latest signal from FastAPI. */
  signal?: Partial<NormalisedSignal>;

  /** Trade size in SOL. Required because amount cannot come from the AI signal. */
  amount_sol: number;

  /** Optional explicit mode. Default = env EXECUTION_MODE. */
  execution_mode?: ExecutionMode;

  /** Free-form user-facing prompt for ElizaOS narration. NEVER influences the
   *  Guardian decision; the Guardian scans this for override patterns and
   *  blocks the request if any are found. */
  narration_request?: string;

  /** Connected Phantom wallet public key (base58). Required when
   *  WALLET_MODE=phantom. Validated by Guardian rule #12. */
  phantom_public_key?: string;

  /** Total SOL executed in confirmed trades today (UTC day). Set by the
   *  route handler from the persistent daily-volume tracker so the Guardian
   *  can enforce MAX_DAILY_LOSS_SOL without the execution package doing I/O. */
  daily_executed_sol?: number;

  /** User's risk-tolerance preference (from the Settings page). When absent the
   *  Guardian falls back to DEFAULT_USER_RISK_LEVEL from env. */
  user_risk_level?: RiskLevel;
}

export interface PrepareTransactionResult {
  guardian: GuardianResult;
  /** base64-encoded VersionedTransaction for client-side signing.
   *  Null when the Guardian blocks or when execution_mode=paper. */
  transaction_base64: string | null;
  execution_mode: ExecutionMode;
  asset: string;
  side: 'BUY' | 'SELL';
  amount_sol: number;
}

export interface ExecutionWorkflowResult {
  request_received_at_utc: string;
  request_completed_at_utc: string;
  guardian: GuardianResult;
  execution: ExecutionResult | null;
  /** The fused model action (BUY/SELL/HOLD) the Guardian evaluated. Retained
   *  even when the trade is blocked so the history can show the intended
   *  direction (the Guardian's sanitized_action is null on a block). */
  signal_action: TradeAction;
  /** Optional human-readable explanation. Filled only when narration is on
   *  AND the Eliza runtime succeeded. */
  narration: string | null;
}
