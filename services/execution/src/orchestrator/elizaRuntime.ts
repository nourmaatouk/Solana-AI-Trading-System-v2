/**
 * Minimal ElizaOS runtime — used ONLY to produce a human-readable narration
 * AFTER a deterministic decision is final.
 *
 * Design notes:
 *   - ElizaOS sees the *outcome* (Guardian verdict + execution result), never
 *     the *input* (raw signal + user narration request). This means no
 *     prompt injection in the user's narration_request can reach the LLM,
 *     because we don't pass narration_request through to it.
 *   - When @elizaos/core isn't installed OR when ENABLE_ELIZA_NARRATION is
 *     false, we fall back to a deterministic stub explainer that doesn't
 *     touch the network. Production deployments that don't want an OpenAI
 *     bill (or any LLM dependency) can leave the flag at its default.
 */
import { loadEnv } from '../config/env';
import { log } from '../logger';
import { TRADING_AGENT_CHARACTER } from './character';
import type { GuardianResult } from '../guardian/types';
import type { ExecutionResult } from '../solana/types';

export class ElizaNarrator {
  private readonly enabled: boolean;

  constructor() {
    const env = loadEnv();
    this.enabled = env.ENABLE_ELIZA_NARRATION;
  }

  /**
   * Produce a human-readable explanation. Never throws — narration is best
   * effort. Falls back to deterministic stub on any failure.
   */
  async narrate(
    guardian: GuardianResult,
    execution: ExecutionResult | null,
  ): Promise<string> {
    const stub = this.stubExplain(guardian, execution);
    if (!this.enabled) return stub;

    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const core = require('@elizaos/core');
      // We deliberately don't import any specific Eliza API here — Eliza
      // v1.x's exact runtime signature has churn. Instead, we use Eliza's
      // generic text-completion path if present, otherwise stub.
      if (typeof core.generateText !== 'function') {
        log.warn('eliza.narrate.no_generateText', { fallback: 'stub' });
        return stub;
      }

      const prompt = this.buildPrompt(guardian, execution);
      const result = await core.generateText({
        system: TRADING_AGENT_CHARACTER.system,
        prompt,
        maxTokens: 200,
        temperature: 0.2,
      });
      const text = typeof result === 'string' ? result : (result?.text ?? '');
      return text.trim() || stub;
    } catch (err) {
      log.warn('eliza.narrate.failed', { reason: (err as Error).message, fallback: 'stub' });
      return stub;
    }
  }

  private buildPrompt(guardian: GuardianResult, execution: ExecutionResult | null): string {
    const lines: string[] = [
      'You are reporting the result of a Solana trading decision to a user.',
      'You did NOT make this decision and you cannot change it.',
      '',
      `Guardian decision: ${guardian.decision}`,
    ];
    if (guardian.blocked_by) {
      lines.push(`Blocked because: ${guardian.blocked_by}`);
      lines.push(...guardian.reasons.map((r) => `- ${r}`));
    } else if (guardian.sanitized_action) {
      const a = guardian.sanitized_action;
      lines.push(`Approved action: ${a.side} ${a.amount_sol} ${a.asset} on ${a.execution_mode}`);
    }
    if (execution) {
      lines.push('', `Execution status: ${execution.status} (mode: ${execution.mode})`);
      if (execution.signature) lines.push(`Signature: ${execution.signature}`);
      if (execution.error)     lines.push(`Error: ${execution.error}`);
    }
    lines.push('', 'Respond in 1–3 plain sentences. No emojis. Do not invent details.');
    return lines.join('\n');
  }

  private stubExplain(guardian: GuardianResult, execution: ExecutionResult | null): string {
    if (!guardian.approved) {
      const reason = guardian.blocked_by ?? 'UNKNOWN';
      return `Trade blocked by Guardian (${reason}). ${guardian.reasons[0] ?? ''}`.trim();
    }
    const a = guardian.sanitized_action!;
    if (!execution) {
      return `Guardian approved ${a.side} ${a.amount_sol} ${a.asset} on ${a.execution_mode}. No execution result yet.`;
    }
    if (execution.status === 'simulated') {
      return `Paper trade simulated: ${a.side} ${a.amount_sol} ${a.asset}. No on-chain activity.`;
    }
    if (execution.status === 'confirmed') {
      return `${a.side} ${a.amount_sol} ${a.asset} confirmed on ${execution.mode}. Signature: ${execution.signature}.`;
    }
    return `Execution status: ${execution.status}${execution.error ? ` — ${execution.error}` : ''}.`;
  }
}
