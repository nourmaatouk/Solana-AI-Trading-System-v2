/**
 * Mainnet executor.
 *
 * Hard-blocked unless ALLOW_MAINNET_EXECUTION=true. Even when armed, the
 * Guardian has already validated the trade — this executor is the last
 * code that touches real money.
 *
 * Two modes, controlled by JUPITER_SWAP_ENABLED:
 *
 *   false (default) — refuses with an error. The mainnet self-transfer
 *     placeholder has been removed: on mainnet there is no safe no-op.
 *     If you want to test the signing path, use EXECUTION_MODE=devnet.
 *
 *   true — Jupiter v6 swap: same flow as DevnetExecutor but on mainnet
 *     RPC. SOLANA_CLUSTER must be 'mainnet-beta'.
 *
 * Security:
 *   - Double-checks ALLOW_MAINNET_EXECUTION at execute() time (not only at
 *     construction) so a live env-var change takes effect without restart.
 *   - Private key never appears in any log.
 */
import { buildSolanaClient, type SolanaClientHandle } from '../client';
import { loadEnv } from '../../config/env';
import { log } from '../../logger';
import { fetchJupiterSwap } from '../jupiterSwap';
import type {
  ExecutionContext,
  ExecutionResult,
  ITransactionExecutor,
  PreparedTransaction,
} from '../types';

export class MainnetExecutor implements ITransactionExecutor {
  readonly mode = 'mainnet' as const;
  private readonly handle: SolanaClientHandle | null;

  constructor() {
    const env = loadEnv();
    if (!env.ALLOW_MAINNET_EXECUTION) {
      log.warn('execution.mainnet.constructed_but_disabled');
      this.handle = null;
      return;
    }
    log.warn('execution.mainnet.armed', { cluster: env.SOLANA_CLUSTER });
    this.handle = buildSolanaClient();
  }

  async execute(ctx: ExecutionContext): Promise<ExecutionResult> {
    const { approved_action: action } = ctx;
    const submitted_at_utc = new Date().toISOString();
    const env = loadEnv();

    if (!env.ALLOW_MAINNET_EXECUTION) {
      const msg = 'Mainnet execution is disabled. Set ALLOW_MAINNET_EXECUTION=true.';
      log.error('execution.mainnet.refused', { reason: msg });
      return failed(action, this.handle?.rpcUrl ?? null, submitted_at_utc, msg);
    }

    if (!this.handle?.signingCapable) {
      const msg = 'Mainnet execution armed but signing not configured (missing SOLANA_PRIVATE_KEY or solana-agent-kit).';
      log.error('execution.mainnet.refused', { reason: msg });
      return failed(action, this.handle?.rpcUrl ?? null, submitted_at_utc, msg);
    }

    if (!env.JUPITER_SWAP_ENABLED) {
      const msg =
        'Mainnet execution requires JUPITER_SWAP_ENABLED=true. ' +
        'The self-transfer placeholder has been removed on mainnet — there is no safe no-op.';
      log.error('execution.mainnet.refused', { reason: msg });
      return failed(action, this.handle.rpcUrl, submitted_at_utc, msg);
    }

    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const web3 = require('@solana/web3.js');
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    let bs58 = require('bs58');
    // bs58 v6 is ESM-first: under CommonJS interop the decode() helpers live on
    // the .default export. Normalise so bs58.decode is always callable.
    if (bs58.default) bs58 = bs58.default;
    const connection = new web3.Connection(this.handle.rpcUrl, 'confirmed');
    const secret = bs58.decode(env.SOLANA_PRIVATE_KEY!);
    const keypair = web3.Keypair.fromSecretKey(secret);

    const { guardian_decision_id } = ctx;

    log.warn('execution.mainnet.invoked', {
      decision_id: guardian_decision_id,
      amount_sol: action.amount_sol,
      side: action.side,
    });

    const conn = connection as {
      sendRawTransaction(raw: Uint8Array, opts: object): Promise<string>;
      confirmTransaction(strategy: object, commitment: string): Promise<unknown>;
      getLatestBlockhash(): Promise<{ blockhash: string; lastValidBlockHeight: number }>;
    };

    try {
      const swapPkg = await fetchJupiterSwap(
        env.JUPITER_API_URL,
        action.side,
        action.amount_sol,
        keypair.publicKey.toBase58(),
        env.QUOTE_CURRENCY_MINT,
        env.JUPITER_SLIPPAGE_BPS,
      );

      log.info('execution.mainnet.jupiter.quoted', {
        decision_id: guardian_decision_id,
        in_amount: swapPkg.inAmountLamports.toString(),
        out_amount: swapPkg.outAmountLamports.toString(),
        price_impact_pct: swapPkg.priceImpactPct,
      });

      const VersionedTransaction = (web3 as Record<string, unknown>)['VersionedTransaction'] as {
        deserialize(data: Uint8Array): {
          sign(signers: { publicKey: unknown; secretKey: Uint8Array }[]): void;
          serialize(): Uint8Array;
        };
      };

      const txBuffer = Buffer.from(swapPkg.swapTransaction, 'base64');
      const transaction = VersionedTransaction.deserialize(new Uint8Array(txBuffer));
      transaction.sign([keypair as { publicKey: unknown; secretKey: Uint8Array }]);

      const rawTx = transaction.serialize();
      const { blockhash, lastValidBlockHeight } = await conn.getLatestBlockhash();
      const sig = await conn.sendRawTransaction(rawTx, {
        skipPreflight: false,
        preflightCommitment: 'confirmed',
      });

      await conn.confirmTransaction(
        { signature: sig, blockhash, lastValidBlockHeight },
        'confirmed',
      );

      const confirmed_at_utc = new Date().toISOString();
      log.info('execution.mainnet.jupiter.confirmed', {
        decision_id: guardian_decision_id,
        signature: sig,
        amount_sol: action.amount_sol,
      });

      return {
        status: 'confirmed', mode: 'mainnet',
        signature: sig, block_height: null,
        asset: action.asset, side: action.side, amount_sol: action.amount_sol,
        rpc_url: this.handle.rpcUrl,
        submitted_at_utc, confirmed_at_utc: confirmed_at_utc, error: null,
        signed_by: keypair.publicKey.toBase58(),
        meta: {
          swap_type: 'jupiter_v6',
          in_amount_lamports: swapPkg.inAmountLamports.toString(),
          out_amount_lamports: swapPkg.outAmountLamports.toString(),
          price_impact_pct: swapPkg.priceImpactPct,
        },
      };
    } catch (err) {
      const msg = (err as Error).message;
      log.error('execution.mainnet.jupiter.failed', { reason: msg });
      return failed(action, this.handle.rpcUrl, submitted_at_utc, msg);
    }
  }

  // ── prepareTransaction — build unsigned tx for Phantom signing ───────────────

  async prepareTransaction(ctx: ExecutionContext): Promise<PreparedTransaction | null> {
    const env = loadEnv();

    if (!env.ALLOW_MAINNET_EXECUTION) {
      throw new Error('Mainnet execution is disabled. Set ALLOW_MAINNET_EXECUTION=true.');
    }
    if (!env.JUPITER_SWAP_ENABLED) {
      throw new Error('prepareTransaction requires JUPITER_SWAP_ENABLED=true on mainnet.');
    }

    const phantomKey = ctx.phantom_public_key;
    if (!phantomKey) {
      throw new Error('phantom_public_key is required for Phantom transaction signing.');
    }

    const { approved_action: action, guardian_decision_id } = ctx;

    log.warn('execution.mainnet.prepare_tx', {
      decision_id: guardian_decision_id,
      side: action.side,
      amount_sol: action.amount_sol,
      user: `${phantomKey.slice(0, 6)}…`,
    });

    const swapPkg = await fetchJupiterSwap(
      env.JUPITER_API_URL,
      action.side,
      action.amount_sol,
      phantomKey,
      env.QUOTE_CURRENCY_MINT,
      env.JUPITER_SLIPPAGE_BPS,
    );

    return {
      transaction_base64: swapPkg.swapTransaction,
      user_public_key: phantomKey,
      mode: 'mainnet',
      asset: action.asset,
      side: action.side,
      amount_sol: action.amount_sol,
      meta: {
        swap_type: 'jupiter_v6',
        in_amount_lamports: swapPkg.inAmountLamports.toString(),
        out_amount_lamports: swapPkg.outAmountLamports.toString(),
        price_impact_pct: swapPkg.priceImpactPct,
      },
    };
  }
}

function failed(
  action: { asset: string; side: 'BUY' | 'SELL'; amount_sol: number },
  rpcUrl: string | null,
  submitted_at_utc: string,
  error: string,
): ExecutionResult {
  return {
    status: 'failed', mode: 'mainnet',
    signature: null, block_height: null,
    asset: action.asset, side: action.side, amount_sol: action.amount_sol,
    rpc_url: rpcUrl,
    submitted_at_utc, confirmed_at_utc: null, error,
    signed_by: null,
  };
}
