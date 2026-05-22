/**
 * SolanaExecutionService — the only public surface for actually executing
 * a trade. Two important invariants:
 *
 *   1. Refuses to execute without a positive GuardianResult.
 *   2. The mode used to pick the executor comes from the GUARDIAN-APPROVED
 *      action, never from the raw caller request. This means a caller can't
 *      sneak in a `mode: "mainnet"` after the Guardian thought it was
 *      validating a devnet request.
 */
import { loadEnv } from '../config/env';
import { log } from '../logger';
import type { GuardianResult } from '../guardian/types';
import { PaperExecutor } from './modes/paperExecutor';
import { DevnetExecutor } from './modes/devnetExecutor';
import { MainnetExecutor } from './modes/mainnetExecutor';
import type {
  ExecutionContext,
  ExecutionResult,
  ITransactionExecutor,
  PreparedTransaction,
} from './types';

export interface ExecutionServiceDeps {
  paper?: ITransactionExecutor;
  devnet?: ITransactionExecutor;
  mainnet?: ITransactionExecutor;
}

export class SolanaExecutionService {
  private readonly executors: Record<string, ITransactionExecutor>;

  constructor(deps: ExecutionServiceDeps = {}) {
    this.executors = {
      paper:   deps.paper   ?? new PaperExecutor(),
      devnet:  deps.devnet  ?? new DevnetExecutor(),
      mainnet: deps.mainnet ?? new MainnetExecutor(),
    };
  }

  /**
   * Execute an approved trade. The guardianResult MUST be the verdict that
   * approved this trade — passing a stale or unrelated APPROVE verdict here
   * is a caller bug, not an attack vector, because:
   *   - the Guardian is the only source of `sanitized_action`, and
   *   - the action's `execution_mode` is what picks the executor.
   *
   * The decision id is the Guardian's evaluated_at_utc, used as an
   * idempotency token in logs.
   */
  async executeApproved(guardianResult: GuardianResult): Promise<ExecutionResult> {
    if (!guardianResult.approved || !guardianResult.sanitized_action) {
      const msg = 'executeApproved called with a BLOCKED Guardian result.';
      log.error('execution.refused.unapproved', { blocked_by: guardianResult.blocked_by });
      throw new Error(msg);
    }

    const env = loadEnv();
    // Belt + braces: if env says EMERGENCY_STOP, refuse even though the
    // Guardian should have already caught this. Reading env at call time
    // (not constructor time) lets the runtime kill-switch take effect
    // without restarting the process.
    if (env.EMERGENCY_STOP) {
      const msg = 'execution refused: EMERGENCY_STOP active at execute time.';
      log.error('execution.refused.emergency_stop');
      throw new Error(msg);
    }

    const mode = guardianResult.sanitized_action.execution_mode;
    const executor = this.executors[mode];
    if (!executor) {
      throw new Error(`No executor configured for mode '${mode}'`);
    }

    const ctx: ExecutionContext = {
      approved_action: guardianResult.sanitized_action,
      guardian_decision_id: guardianResult.evaluated_at_utc,
    };

    log.info('execution.dispatch', {
      mode,
      decision_id: ctx.guardian_decision_id,
      asset: ctx.approved_action.asset,
      side: ctx.approved_action.side,
      amount_sol: ctx.approved_action.amount_sol,
    });

    return executor.execute(ctx);
  }

  /**
   * Build an unsigned VersionedTransaction for client-side (Phantom) signing.
   * Returns null for paper mode — there is no real transaction to sign.
   * Throws for mainnet/devnet when prerequisites are missing (no Jupiter, no phantom key).
   */
  async prepareApproved(
    guardianResult: GuardianResult,
    phantomPublicKey?: string,
  ): Promise<PreparedTransaction | null> {
    if (!guardianResult.approved || !guardianResult.sanitized_action) {
      throw new Error('prepareApproved called with a BLOCKED Guardian result.');
    }

    const env = loadEnv();
    if (env.EMERGENCY_STOP) {
      throw new Error('prepare refused: EMERGENCY_STOP active.');
    }

    const mode = guardianResult.sanitized_action.execution_mode;
    const executor = this.executors[mode];
    if (!executor) throw new Error(`No executor configured for mode '${mode}'`);

    const ctx: ExecutionContext = {
      approved_action: guardianResult.sanitized_action,
      guardian_decision_id: guardianResult.evaluated_at_utc,
      phantom_public_key: phantomPublicKey,
    };

    return executor.prepareTransaction(ctx);
  }
}
