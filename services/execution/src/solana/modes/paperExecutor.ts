/**
 * Paper executor — pure simulation. Does NOT touch the Solana network and
 * does NOT require a private key. This is the default execution mode.
 *
 * It still records a result and a deterministic fake signature so the rest
 * of the system (dashboard, history) treats paper-mode trades the same as
 * real ones.
 */
import { log } from '../../logger';
import type {
  ExecutionContext,
  ExecutionResult,
  ITransactionExecutor,
  PreparedTransaction,
} from '../types';

export class PaperExecutor implements ITransactionExecutor {
  readonly mode = 'paper' as const;

  async execute(ctx: ExecutionContext): Promise<ExecutionResult> {
    const { approved_action: action, guardian_decision_id } = ctx;
    const submitted_at_utc = new Date().toISOString();
    const fakeSig = `PAPER-${guardian_decision_id}-${Date.now()}`;

    log.info('execution.paper.simulated', {
      decision_id: guardian_decision_id,
      asset: action.asset,
      side: action.side,
      amount_sol: action.amount_sol,
      signature: fakeSig,
    });

    return {
      status: 'simulated',
      mode: 'paper',
      signature: fakeSig,
      block_height: null,
      asset: action.asset,
      side: action.side,
      amount_sol: action.amount_sol,
      rpc_url: null,
      submitted_at_utc,
      confirmed_at_utc: submitted_at_utc,
      error: null,
      signed_by: null,
      meta: { note: 'No on-chain activity. This is a simulation.' },
    };
  }

  // Paper mode has no real transaction to sign.
  async prepareTransaction(_ctx: ExecutionContext): Promise<PreparedTransaction | null> {
    return null;
  }
}
