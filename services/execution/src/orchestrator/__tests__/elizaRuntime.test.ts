/**
 * ElizaNarrator tests — exercise the fallback branches that don't require a
 * live LLM call. These tests pin down the deterministic behaviour when both
 * Ollama and Anthropic are unavailable. A non-routable Ollama URL is used so
 * the liveness check fails fast, just as it would when Ollama is not running.
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { ElizaNarrator } from '../elizaRuntime';
import { resetEnvCache } from '../../config/env';
import type { GuardianResult } from '../../guardian/types';
import type { ExecutionResult } from '../../solana/types';

const APPROVED: GuardianResult = {
  approved: true,
  decision: 'APPROVE',
  reasons: ['all checks passed'],
  blocked_by: null,
  evaluated_at_utc: '2026-05-24T00:00:00.000Z',
  sanitized_action: {
    asset: 'SOL',
    side: 'BUY',
    amount_sol: 0.05,
    execution_mode: 'devnet',
  },
};

const SIMULATED_EXEC: ExecutionResult = {
  status: 'simulated',
  mode: 'paper',
  signature: null,
  block_height: null,
  asset: 'SOL',
  side: 'BUY',
  amount_sol: 0.05,
  rpc_url: null,
  submitted_at_utc: '2026-05-24T00:00:00.000Z',
  confirmed_at_utc: null,
  error: null,
};

const SAVED = { ...process.env };

function setEnv(env: Record<string, string | undefined>): void {
  for (const [k, v] of Object.entries(env)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
  resetEnvCache();
}

beforeEach(() => {
  // Point at a non-routable address so the Ollama liveness check fails fast
  // (same behaviour as when Ollama is simply not running), and clear any
  // stray ANTHROPIC_API_KEY so neither provider is reachable in these tests.
  process.env.OLLAMA_BASE_URL = 'http://127.0.0.1:1';
  delete process.env.ANTHROPIC_API_KEY;
  resetEnvCache();
});

afterEach(() => {
  process.env = { ...SAVED };
  resetEnvCache();
});

describe('ElizaNarrator', () => {
  it('returns the deterministic stub when narration is disabled', async () => {
    setEnv({ ENABLE_ELIZA_NARRATION: 'false' });
    const out = await new ElizaNarrator().narrate(APPROVED, SIMULATED_EXEC);
    expect(out).toContain('Paper trade simulated');
  });

  it('falls back to the stub when enabled but ANTHROPIC_API_KEY is missing', async () => {
    setEnv({
      ENABLE_ELIZA_NARRATION: 'true',
      ENABLE_LLM_NARRATION_FALLBACK_STUB: 'true',
      ANTHROPIC_API_KEY: undefined,
    });
    const out = await new ElizaNarrator().narrate(APPROVED, SIMULATED_EXEC);
    expect(out).toContain('Paper trade simulated');
  });

  it('returns an empty string when the LLM is unavailable and stub fallback is off', async () => {
    setEnv({
      ENABLE_ELIZA_NARRATION: 'true',
      ENABLE_LLM_NARRATION_FALLBACK_STUB: 'false',
      ANTHROPIC_API_KEY: undefined,
    });
    const out = await new ElizaNarrator().narrate(APPROVED, SIMULATED_EXEC);
    expect(out).toBe('');
  });

  it('never throws — best-effort narration', async () => {
    setEnv({ ENABLE_ELIZA_NARRATION: 'true', ANTHROPIC_API_KEY: undefined });
    await expect(
      new ElizaNarrator().narrate(APPROVED, null),
    ).resolves.toBeTypeOf('string');
  });
});
