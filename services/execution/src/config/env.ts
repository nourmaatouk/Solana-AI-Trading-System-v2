/**
 * Centralised environment loader for the execution layer.
 *
 * Every configurable value the Guardian and Solana code use flows through here.
 * Anything security-critical (mainnet flag, private key presence, RPC URL) is
 * validated with zod at module load time so misconfigurations fail loud and
 * early instead of becoming runtime surprises.
 */
import { z } from 'zod';

// ── Coercion helpers ────────────────────────────────────────────────────────
const boolFromString = z
  .union([z.boolean(), z.string()])
  .transform((v) => (typeof v === 'boolean' ? v : v.trim().toLowerCase() === 'true'));

const positiveNumberString = z
  .string()
  .transform((v) => Number(v))
  .pipe(z.number().positive());

const nonNegativeNumberString = z
  .string()
  .transform((v) => Number(v))
  .pipe(z.number().nonnegative());

const csvList = z
  .string()
  .transform((v) =>
    v
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean),
  );

// ── Schema ──────────────────────────────────────────────────────────────────
const ExecutionEnvSchema = z.object({
  EXECUTION_MODE: z.enum(['paper', 'devnet', 'mainnet']).default('paper'),
  REQUIRE_GUARDIAN_APPROVAL: boolFromString.default('true'),
  ALLOW_MAINNET_EXECUTION: boolFromString.default('false'),
  EMERGENCY_STOP: boolFromString.default('false'),
  ENABLE_ELIZA_NARRATION: boolFromString.default('false'),

  // Solana network
  SOLANA_CLUSTER: z.enum(['devnet', 'testnet', 'mainnet-beta']).default('devnet'),
  SOLANA_RPC_URL: z.string().url().default('https://api.devnet.solana.com'),
  SOLANA_PUBLIC_KEY: z.string().optional(),
  // Intentionally optional and never logged. Only required when EXECUTION_MODE
  // is devnet/mainnet AND a signing operation is requested.
  SOLANA_PRIVATE_KEY: z.string().optional(),

  // Limits
  MAX_TRADE_SIZE_SOL: positiveNumberString.default('0.1'),
  MAX_DAILY_LOSS_SOL: positiveNumberString.default('0.5'),
  MAX_SPOOFING_CONFIDENCE: nonNegativeNumberString.default('0.15'),
  SIGNAL_MAX_AGE_SECONDS: positiveNumberString.default('300'),
  CONFIDENCE_THRESHOLD: nonNegativeNumberString.default('0.69'),

  ALLOWED_ASSETS: csvList.default('SOL/USDT'),

  // ── Risk tolerance ──────────────────────────────────────────────────────────
  // When false, any signal with risk_level=HIGH is always blocked even when the
  // user has set their preference to HIGH.  An additional explicit opt-in.
  ALLOW_HIGH_RISK_TRADES: boolFromString.default('false'),
  // Server-side default when the client omits user_risk_level. Set to LOW to
  // fail safe; raise it when the deployment is risk-tolerant by default.
  DEFAULT_USER_RISK_LEVEL: z.enum(['LOW', 'MEDIUM', 'HIGH']).default('LOW'),

  // Wallet mode: phantom = browser wallet required, system = backend keypair, paper = no wallet
  WALLET_MODE: z.enum(['paper', 'system', 'phantom']).default('phantom'),

  // Devnet airdrop limits (ignored on mainnet/testnet — routes reject there)
  DEVNET_AIRDROP_AMOUNT_SOL: positiveNumberString.default('1'),
  MAX_DEVNET_AIRDROP_SOL: positiveNumberString.default('2'),

  // ── Jupiter DEX integration ────────────────────────────────────────────────
  // Enable real Jupiter v6 swaps instead of the SOL self-transfer placeholder.
  // NOTE: Jupiter's quote/swap API is mainnet-only. When EXECUTION_MODE=devnet,
  // the transaction will be built against mainnet program addresses and will
  // fail on a devnet RPC. For real devnet testing, point SOLANA_RPC_URL at a
  // mainnet RPC and use EXECUTION_MODE=mainnet with tiny amounts.
  JUPITER_SWAP_ENABLED: boolFromString.default('false'),
  JUPITER_API_URL: z.string().url().default('https://quote-api.jup.ag/v6'),
  // Slippage tolerance in basis points (50 = 0.5 %). Guardian blocks trades
  // above MAX_SPOOFING_CONFIDENCE, so tighten slippage here independently.
  JUPITER_SLIPPAGE_BPS: positiveNumberString.default('50'),
  // Quote currency mint address. Defaults to USDC (mainnet). For SOL/USDT
  // pairs, set to the USDT mint: Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB
  QUOTE_CURRENCY_MINT: z
    .string()
    .default('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'),

  // ── On-chain Guardian ──────────────────────────────────────────────────────
  // Program ID: DVQ3uopMVLznw3tLJahTkfE1VwzNSPHo1Q8tTvUbjdrG (devnet)
  // Set ENABLE_ONCHAIN_GUARDIAN=true after deploying + initializing the program.
  // When false, only the off-chain TypeScript Guardian runs (default behaviour).
  ENABLE_ONCHAIN_GUARDIAN: boolFromString.default('false'),
  GUARDIAN_PROGRAM_ID: z.string().optional(),
  // Public key of the wallet that called initialize_config (the GuardianConfig
  // PDA admin). Defaults to SOLANA_PUBLIC_KEY when not set.
  GUARDIAN_ADMIN_PUBKEY: z.string().optional(),
  // Private key that signs authorize_execution. Defaults to SOLANA_PRIVATE_KEY
  // when not set. Use this when the allowed_executor in the GuardianConfig PDA
  // differs from the trading keypair (SOLANA_PRIVATE_KEY).
  GUARDIAN_PRIVATE_KEY: z.string().optional(),

  PYTHON_API: z.string().url().default('http://localhost:8000'),
  PYTHON_API_KEY: z.string().optional(),

  ENABLE_LLM_NARRATION_FALLBACK_STUB: boolFromString.default('true'),

  // ── ElizaOS narration — Ollama (free, local) ─────────────────────────────
  // Default provider. Requires Ollama running locally (https://ollama.com).
  // Set OLLAMA_MODEL to any model you have pulled, e.g. llama3.2:latest.
  OLLAMA_BASE_URL: z.string().url().default('http://localhost:11434'),
  OLLAMA_MODEL: z.string().default('llama3.2:latest'),

  // ── ElizaOS narration — Anthropic (optional, paid) ───────────────────────
  // When set, Anthropic is used instead of Ollama. Leave empty to use Ollama.
  // Never logged.
  ANTHROPIC_API_KEY: z.string().optional(),
  ANTHROPIC_SMALL_MODEL: z.string().default('claude-3-5-haiku-20241022'),
});

export type ExecutionEnv = z.infer<typeof ExecutionEnvSchema>;

let cached: ExecutionEnv | null = null;

/**
 * Load and validate the execution-layer environment.
 *
 * Lazy: read on first call so tests can mutate process.env, then call
 * resetEnvCache() between cases.
 */
export function loadEnv(): ExecutionEnv {
  if (cached) return cached;
  const parsed = ExecutionEnvSchema.safeParse(process.env);
  if (!parsed.success) {
    const issues = parsed.error.issues
      .map((i) => `  - ${i.path.join('.')}: ${i.message}`)
      .join('\n');
    throw new Error(`Invalid execution-layer environment:\n${issues}`);
  }
  cached = parsed.data;
  return cached;
}

/** Test helper — drop the cached config so the next loadEnv() re-reads process.env. */
export function resetEnvCache(): void {
  cached = null;
}

/**
 * Return whether the runtime is configured to perform real on-chain operations.
 * paper mode → false (simulated). devnet/mainnet → true.
 */
export function isOnChainMode(env: ExecutionEnv = loadEnv()): boolean {
  return env.EXECUTION_MODE !== 'paper';
}
