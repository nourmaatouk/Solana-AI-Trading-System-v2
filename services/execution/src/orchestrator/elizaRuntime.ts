/**
 * ElizaOS runtime — used ONLY to produce a human-readable narration AFTER a
 * deterministic decision is final.
 *
 * Design notes:
 *   - ElizaOS sees the *outcome* (Guardian verdict + execution result), never
 *     the *input* (raw signal + user narration request). No prompt-injection
 *     risk because narration_request is never forwarded to the LLM.
 *   - Provider priority:
 *       1. Ollama (free, local)  — used when OLLAMA_BASE_URL is reachable.
 *          Default model: llama3.2:latest. No API key required.
 *       2. Anthropic (optional)  — used only when ANTHROPIC_API_KEY is set
 *          AND Ollama is not available/configured.
 *       3. Deterministic stub    — when both providers are unavailable or
 *          ENABLE_ELIZA_NARRATION is false.
 *   - @elizaos/core is an OPTIONAL dependency and is require()'d lazily so
 *     the rest of the codebase compiles and tests run without it.
 *   - Never throws — narration is best-effort.
 */
import { loadEnv } from '../config/env';
import { log } from '../logger';
import { TRADING_AGENT_CHARACTER } from './character';
import type { GuardianResult } from '../guardian/types';
import type { ExecutionResult } from '../solana/types';

interface ElizaRuntime {
  registerModel(modelType: string, handler: (rt: unknown, params: Record<string, unknown>) => Promise<unknown>, provider: string): void;
  generateText(input: string, options?: Record<string, unknown>): Promise<{ text?: string } | string>;
}

export class ElizaNarrator {
  private readonly enabled: boolean;
  private readonly fallbackStub: boolean;
  private runtimeInit: Promise<ElizaRuntime | null> | null = null;
  private textSmallModel = 'TEXT_SMALL';

  constructor() {
    const env = loadEnv();
    this.enabled = env.ENABLE_ELIZA_NARRATION;
    this.fallbackStub = env.ENABLE_LLM_NARRATION_FALLBACK_STUB;
  }

  /**
   * Produce a human-readable explanation. Never throws.
   * Falls back to the deterministic stub (or empty string) on any failure.
   */
  async narrate(
    guardian: GuardianResult,
    execution: ExecutionResult | null,
  ): Promise<string> {
    const fallback = this.fallbackStub ? this.stubExplain(guardian, execution) : '';
    if (!this.enabled) return fallback;

    try {
      const runtime = await this.getRuntime();
      if (!runtime) return fallback;

      const prompt = this.buildPrompt(guardian, execution);
      const result = await runtime.generateText(prompt, {
        includeCharacter: true,
        modelType: this.textSmallModel,
        maxTokens: 200,
        temperature: 0.2,
      });
      const text = (typeof result === 'string' ? result : result?.text ?? '').trim();
      return text || fallback;
    } catch (err) {
      log.warn('eliza.narrate.failed', {
        reason: (err as Error).message,
        fallback: this.fallbackStub ? 'stub' : 'empty',
      });
      return fallback;
    }
  }

  private getRuntime(): Promise<ElizaRuntime | null> {
    if (!this.runtimeInit) this.runtimeInit = this.buildRuntime();
    return this.runtimeInit;
  }

  /**
   * Build a real ElizaOS AgentRuntime.
   * Tries Ollama first (free, local), then Anthropic (if key set), then null.
   */
  private async buildRuntime(): Promise<ElizaRuntime | null> {
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const core = require('@elizaos/core');
      if (typeof core.AgentRuntime !== 'function') {
        log.warn('eliza.narrate.core_missing');
        return null;
      }
      if (core.ModelType?.TEXT_SMALL) this.textSmallModel = core.ModelType.TEXT_SMALL;

      const noopAdapter = new Proxy({}, { get: () => async () => undefined });
      const runtime: ElizaRuntime = new core.AgentRuntime({
        character: this.elizaCharacter(),
        adapter: noopAdapter,
        settings: {},
      });

      // ── Try Ollama first (free, local) ──────────────────────────────────
      const ollamaOk = await this.tryRegisterOllama(runtime);
      if (ollamaOk) return runtime;

      // ── Fall back to Anthropic if key is present ─────────────────────────
      const anthropicOk = await this.tryRegisterAnthropic(runtime);
      if (anthropicOk) return runtime;

      log.warn('eliza.narrate.no_provider', {
        hint: 'Start Ollama (ollama serve) or set ANTHROPIC_API_KEY',
        fallback: this.fallbackStub ? 'stub' : 'empty',
      });
      return null;
    } catch (err) {
      log.warn('eliza.narrate.init_failed', {
        reason: (err as Error).message,
        fallback: this.fallbackStub ? 'stub' : 'empty',
      });
      return null;
    }
  }

  /**
   * Register a custom TEXT_SMALL handler that calls the local Ollama API.
   * Returns true when Ollama is reachable and registration succeeds.
   */
  private async tryRegisterOllama(runtime: ElizaRuntime): Promise<boolean> {
    const env = loadEnv();
    const baseUrl = env.OLLAMA_BASE_URL;
    const model = env.OLLAMA_MODEL;

    // Quick liveness check before registering
    try {
      const res = await fetch(`${baseUrl}/api/tags`, { signal: AbortSignal.timeout(3000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch {
      log.info('eliza.narrate.ollama_unavailable', { baseUrl, hint: 'run: ollama serve' });
      return false;
    }

    runtime.registerModel(
      this.textSmallModel,
      async (_rt: unknown, params: Record<string, unknown>) => {
        const prompt = (params.prompt as string) ?? '';
        const res = await fetch(`${baseUrl}/api/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model,
            prompt,
            stream: false,
            options: {
              temperature: (params.temperature as number) ?? 0.2,
              num_predict: (params.maxTokens as number) ?? 200,
            },
          }),
          signal: AbortSignal.timeout(30_000),
        });
        if (!res.ok) throw new Error(`Ollama HTTP ${res.status}`);
        const data = await res.json() as { response?: string };
        // Return plain string — AgentRuntime.generateText wraps the handler
        // return value in { text: ... } itself; returning an object would
        // double-wrap and produce { text: { text: "..." } }.
        return data.response ?? '';
      },
      'ollama',
    );

    log.info('eliza.narrate.ready', { provider: 'ollama', model });
    return true;
  }

  /**
   * Register the @elizaos/plugin-anthropic handler.
   * Returns true only when ANTHROPIC_API_KEY is set and the plugin loads.
   */
  private async tryRegisterAnthropic(runtime: ElizaRuntime): Promise<boolean> {
    const env = loadEnv();
    if (!env.ANTHROPIC_API_KEY) return false;

    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const { anthropicPlugin } = require('@elizaos/plugin-anthropic');
      // AgentRuntime.registerPlugin is async and sets up settings internally
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      await (runtime as any).registerPlugin(anthropicPlugin);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (runtime as any).setSetting('ANTHROPIC_API_KEY', env.ANTHROPIC_API_KEY);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (runtime as any).setSetting('ANTHROPIC_SMALL_MODEL', env.ANTHROPIC_SMALL_MODEL);
      log.info('eliza.narrate.ready', { provider: 'anthropic', model: env.ANTHROPIC_SMALL_MODEL });
      return true;
    } catch (err) {
      log.warn('eliza.narrate.anthropic_failed', { reason: (err as Error).message });
      return false;
    }
  }

  private elizaCharacter(): Record<string, unknown> {
    const c = TRADING_AGENT_CHARACTER;
    return {
      name: c.name,
      username: c.username,
      bio: [...c.bio],
      system: c.system,
      topics: [...c.topics],
    };
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
