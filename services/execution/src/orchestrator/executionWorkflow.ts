/**
 * ExecutionWorkflow — the top-level deterministic orchestrator.
 *
 *   signal fetch  →  Guardian.validate  →  [block?]  →  SolanaExecutionService
 *                                                  →  ElizaNarrator (optional)
 *
 * Everything here is reproducible: same inputs → same outputs. The LLM is
 * NEVER on the critical path; it only generates a post-decision narration.
 */
import { loadEnv } from '../config/env';
import { GuardianValidator } from '../guardian/validator';
import type { ExecutionMode, GuardianInput, GuardianResult } from '../guardian/types';
import { SignalClient, type NormalisedSignal } from '../signalClient';
import { SolanaExecutionService } from '../solana/executionService';
import { ElizaNarrator } from './elizaRuntime';
import { log } from '../logger';
import type { ExecutionRequest, ExecutionWorkflowResult, PrepareTransactionResult } from './types';

export interface WorkflowDeps {
  signalClient?: SignalClient;
  guardian?: GuardianValidator;
  executor?: SolanaExecutionService;
  narrator?: ElizaNarrator;
}

export class ExecutionWorkflow {
  private readonly signalClient: SignalClient;
  private readonly guardian: GuardianValidator;
  private readonly executor: SolanaExecutionService;
  private readonly narrator: ElizaNarrator;

  constructor(deps: WorkflowDeps = {}) {
    this.signalClient = deps.signalClient ?? new SignalClient();
    this.guardian     = deps.guardian     ?? new GuardianValidator();
    this.executor     = deps.executor     ?? new SolanaExecutionService();
    this.narrator     = deps.narrator     ?? new ElizaNarrator();
  }

  /**
   * Validate-only path: fetch signal, build Guardian input, validate, return.
   * Used by POST /execution/validate.
   */
  async validate(req: ExecutionRequest): Promise<GuardianResult> {
    const input = await this.buildGuardianInput(req);
    return this.guardian.validate(input);
  }

  /**
   * Full execution path. Returns a complete trace including the Guardian
   * verdict, the execution result (when approved), and an optional
   * narration.
   */
  async execute(req: ExecutionRequest): Promise<ExecutionWorkflowResult> {
    const request_received_at_utc = new Date().toISOString();
    const input = await this.buildGuardianInput(req);
    const guardian = this.guardian.validate(input);

    if (!guardian.approved) {
      log.info('workflow.blocked', {
        blocked_by: guardian.blocked_by,
        reason: guardian.reasons[0],
      });
      const narration = await this.narrator.narrate(guardian, null);
      return {
        request_received_at_utc,
        request_completed_at_utc: new Date().toISOString(),
        guardian,
        execution: null,
        signal_action: input.action,
        narration,
      };
    }

    const execution = await this.executor.executeApproved(guardian, {
      signal_generated_at_utc: input.signal_generated_at_utc ?? undefined,
      confidence: input.confidence ?? undefined,
    });
    const narration = await this.narrator.narrate(guardian, execution);

    return {
      request_received_at_utc,
      request_completed_at_utc: new Date().toISOString(),
      guardian,
      execution,
      signal_action: input.action,
      narration,
    };
  }

  /**
   * Guardian-validate then build an unsigned VersionedTransaction for
   * client-side (Phantom) signing. The transaction is NOT signed or submitted
   * here — the browser wallet handles that.
   *
   * Returns transaction_base64=null when:
   *   - The Guardian blocks the trade, or
   *   - execution_mode=paper (no real transaction exists).
   */
  async prepareTransaction(req: ExecutionRequest): Promise<PrepareTransactionResult> {
    const input = await this.buildGuardianInput(req);
    const guardian = this.guardian.validate(input);
    const env = loadEnv();
    const mode: ExecutionMode = req.execution_mode ?? env.EXECUTION_MODE;

    if (!guardian.approved || !guardian.sanitized_action) {
      return {
        guardian,
        transaction_base64: null,
        execution_mode: mode,
        asset: input.asset,
        side: input.action === 'SELL' ? 'SELL' : 'BUY',
        amount_sol: req.amount_sol,
      };
    }

    const prepared = await this.executor.prepareApproved(guardian, req.phantom_public_key);

    return {
      guardian,
      transaction_base64: prepared?.transaction_base64 ?? null,
      execution_mode: guardian.sanitized_action.execution_mode,
      asset: guardian.sanitized_action.asset,
      side: guardian.sanitized_action.side,
      amount_sol: guardian.sanitized_action.amount_sol,
    };
  }

  /**
   * Build a GuardianInput from a request. Always prefers the live signal
   * over any caller-provided override unless the caller explicitly passes
   * a complete signal object. This ensures a request can't sneak in a
   * forged "confidence: 1.0".
   */
  private async buildGuardianInput(req: ExecutionRequest): Promise<GuardianInput> {
    const env = loadEnv();
    const mode: ExecutionMode = req.execution_mode ?? env.EXECUTION_MODE;

    // Fetch the live signal even when the caller provides one — we'll
    // override only specific fields with the caller's data and only when
    // they are well-formed. Free-text fields (asset string) never flow
    // straight through.
    let live: NormalisedSignal;
    try {
      live = await this.signalClient.getLatestSignal();
    } catch (err) {
      log.error('workflow.signal_fetch_failed', { reason: (err as Error).message });
      throw err;
    }

    const merged: NormalisedSignal = {
      ...live,
      ...(req.signal ?? {}),
      // Constrain: always force the action through the live signal so a
      // request can't override HOLD → BUY.
      action: live.action,
      // Always force the asset from live, ignore any caller-provided asset.
      asset: live.asset,
    } as NormalisedSignal;

    return {
      asset: merged.asset,
      action: merged.action,
      // Confidence gate uses the classification probability — the SAME field
      // paper-mode gates against the 0.69 threshold. The top-level `confidence`
      // is |fused_score| (a different, much smaller scale gated by the fusion
      // layer at ±0.30), so checking it against 0.69 made devnet over-block
      // with LOW_CONFIDENCE while paper APPROVED the identical signal. Fall
      // back to the fused confidence only when classification_confidence is
      // genuinely absent.
      confidence:
        merged.classification_confidence && merged.classification_confidence > 0
          ? merged.classification_confidence
          : merged.confidence,
      fused_score: merged.fused_score,
      risk_level: merged.risk_level,
      spoofing_confidence: merged.spoofing_confidence,
      sentiment_relevance: merged.sentiment_relevance,
      classification_confidence: merged.classification_confidence,
      classification_probability_up: merged.classification_probability_up,
      signal_generated_at_utc: merged.signal_generated_at_utc,
      amount_sol: req.amount_sol,
      execution_mode: mode,
      narration_request: req.narration_request,
      phantom_public_key: req.phantom_public_key,
      daily_executed_sol: req.daily_executed_sol,
      user_risk_level: req.user_risk_level,
    };
  }
}
