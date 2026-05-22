/**
 * Solana Agent Kit v2 client wrapper.
 *
 * Why this exists:
 *   - We want a single, narrow surface that the rest of the execution layer
 *     uses for Solana operations. The actual SAK API can evolve and we will
 *     only update this file.
 *   - SAK v2 is imported lazily via require() so the rest of the codebase
 *     compiles (and tests run) without the package installed.
 *
 * Security:
 *   - Private key never appears in any log.
 *   - The client refuses to construct in mainnet mode unless
 *     ALLOW_MAINNET_EXECUTION=true.
 */
import { loadEnv, type ExecutionEnv } from '../config/env';
import { log } from '../logger';

/** Narrow interface representing the part of SAK v2 we use. */
export interface ISolanaAgentKit {
  wallet: { publicKey: { toBase58(): string } };
  connection: {
    getBalance(pubkey: unknown): Promise<number>;
    getLatestBlockhash(): Promise<{ blockhash: string; lastValidBlockHeight: number }>;
  };
}

export interface BuildSolanaClientOptions {
  env?: ExecutionEnv;
}

export interface SolanaClientHandle {
  /** Lazily-loaded SAK instance. Null in paper mode or when SDK absent. */
  agent: ISolanaAgentKit | null;
  cluster: string;
  rpcUrl: string;
  publicKey: string | null;
  signingCapable: boolean;
}

/**
 * Build a SAK handle. Returns a structurally-typed handle even when SAK
 * isn't installed (paper/devnet read-only) so callers can introspect without
 * branching on undefined.
 */
export function buildSolanaClient(opts: BuildSolanaClientOptions = {}): SolanaClientHandle {
  const env = opts.env ?? loadEnv();
  const wantsSigning = env.EXECUTION_MODE !== 'paper' && !!env.SOLANA_PRIVATE_KEY;

  if (env.EXECUTION_MODE === 'mainnet' && !env.ALLOW_MAINNET_EXECUTION) {
    throw new Error(
      'Refusing to build a Solana client in mainnet mode while ALLOW_MAINNET_EXECUTION=false',
    );
  }

  let agent: ISolanaAgentKit | null = null;

  if (wantsSigning) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const { SolanaAgentKit } = require('solana-agent-kit');
      // SAK v2 constructor signature (per docs):
      //   new SolanaAgentKit(privateKeyBase58: string, rpcUrl: string, config?: object)
      agent = new SolanaAgentKit(env.SOLANA_PRIVATE_KEY, env.SOLANA_RPC_URL, {});
      log.info('solana.client.constructed', {
        cluster: env.SOLANA_CLUSTER,
        signing: true,
        // public_key intentionally NOT logged here; logged at narrate time
      });
    } catch (err) {
      log.warn('solana.client.sdk_missing', {
        reason: (err as Error).message,
        fallback: 'read-only',
      });
      agent = null;
    }
  }

  return {
    agent,
    cluster: env.SOLANA_CLUSTER,
    rpcUrl: env.SOLANA_RPC_URL,
    publicKey: agent ? agent.wallet.publicKey.toBase58() : env.SOLANA_PUBLIC_KEY ?? null,
    signingCapable: agent !== null,
  };
}
