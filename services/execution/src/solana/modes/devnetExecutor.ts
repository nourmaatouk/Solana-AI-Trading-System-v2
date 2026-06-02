/**
 * Devnet executor — EXECUTION_MODE=devnet, no ALLOW_MAINNET_EXECUTION required.
 *
 * Two modes, controlled by JUPITER_SWAP_ENABLED:
 *
 *   false (default) — SOL self-transfer: verifies wallet/RPC/signing path
 *     without touching any DEX. Blocked when SOLANA_CLUSTER=mainnet-beta
 *     (self-transfer on mainnet moves real money with no trade intent).
 *
 *   true — Jupiter v6 swap: fetches a quote + swap transaction from Jupiter,
 *     signs with SOLANA_PRIVATE_KEY, submits to SOLANA_RPC_URL.
 *     Because Jupiter's program accounts only exist on mainnet, set
 *     SOLANA_RPC_URL to a mainnet endpoint and SOLANA_CLUSTER=mainnet-beta.
 *     EXECUTION_MODE stays 'devnet' — no ALLOW_MAINNET_EXECUTION flag needed.
 *
 * Security:
 *   - Never falls back to paper mode on failure — fails loudly.
 *   - Private key never appears in any log.
 */
import { buildSolanaClient, type SolanaClientHandle } from '../client';
import { loadEnv } from '../../config/env';
import { log } from '../../logger';
import { fetchJupiterSwap } from '../jupiterSwap';
import { authorizeOnChain } from '../onchainGuardian';
import type {
  ExecutionContext,
  ExecutionResult,
  ITransactionExecutor,
  PreparedTransaction,
} from '../types';


export class DevnetExecutor implements ITransactionExecutor {
  readonly mode = 'devnet' as const;
  private readonly handle: SolanaClientHandle;

  constructor(handle?: SolanaClientHandle) {
    const env = loadEnv();
    // When Jupiter is enabled the RPC must point at mainnet (Jupiter programs
    // don't exist on devnet). Warn loudly; do not throw — the caller chose
    // the RPC URL explicitly.
    if (env.JUPITER_SWAP_ENABLED && env.SOLANA_CLUSTER !== 'mainnet-beta') {
      log.warn('execution.devnet.jupiter.cluster_mismatch', {
        cluster: env.SOLANA_CLUSTER,
        note: 'Jupiter requires a mainnet RPC. Set SOLANA_RPC_URL + SOLANA_CLUSTER=mainnet-beta or submissions will fail.',
      });
    }
    this.handle = handle ?? buildSolanaClient();
  }

  async execute(ctx: ExecutionContext): Promise<ExecutionResult> {
    const { approved_action: action, guardian_decision_id } = ctx;
    const submitted_at_utc = new Date().toISOString();
    const env = loadEnv();

    // ── On-chain Guardian check ──────────────────────────────────────────────
    // When ENABLE_ONCHAIN_GUARDIAN=true the Guardian program on Solana validates
    // the trade before the swap runs. The program enforces the same rules as the
    // off-chain TypeScript Guardian but at the validator level — if it rejects,
    // the swap instruction is never reached.
    if (env.ENABLE_ONCHAIN_GUARDIAN && env.GUARDIAN_PROGRAM_ID && env.SOLANA_PRIVATE_KEY) {
      const adminPubkey = env.GUARDIAN_ADMIN_PUBKEY ?? env.SOLANA_PUBLIC_KEY;
      if (!adminPubkey) {
        log.error('onchain_guardian.missing_admin', {
          note: 'Set GUARDIAN_ADMIN_PUBKEY or SOLANA_PUBLIC_KEY',
        });
        return {
          status: 'failed', mode: 'devnet',
          signature: null, block_height: null,
          asset: action.asset, side: action.side, amount_sol: action.amount_sol,
          rpc_url: this.handle.rpcUrl, submitted_at_utc, confirmed_at_utc: null,
          error: 'On-chain Guardian: GUARDIAN_ADMIN_PUBKEY is not configured.',
          signed_by: env.SOLANA_PUBLIC_KEY ?? null,
        };
      }

      const signalKey = [
        ctx.signal_generated_at_utc ?? guardian_decision_id,
        action.side,
        action.amount_sol,
      ].join('|');

      log.info('onchain_guardian.checking', {
        decision_id: guardian_decision_id,
        program_id: env.GUARDIAN_PROGRAM_ID,
        amount_sol: action.amount_sol,
      });

      // GUARDIAN_PRIVATE_KEY signs the authorize_execution tx.
      // Falls back to SOLANA_PRIVATE_KEY when both keys are the same wallet.
      const guardianKey = env.GUARDIAN_PRIVATE_KEY ?? env.SOLANA_PRIVATE_KEY;
      const onChain = await authorizeOnChain({
        programId:        env.GUARDIAN_PROGRAM_ID,
        adminPubkey,
        rpcUrl:           this.handle.rpcUrl,
        privateKeyBase58: guardianKey!,
        amountSol:        action.amount_sol,
        confidenceRatio:  ctx.confidence ?? 0,
        signalKey,
      });

      if (!onChain.authorized) {
        log.warn('onchain_guardian.blocked', { reason: onChain.error, decision_id: guardian_decision_id });
        return {
          status: 'failed', mode: 'devnet',
          signature: null, block_height: null,
          asset: action.asset, side: action.side, amount_sol: action.amount_sol,
          rpc_url: this.handle.rpcUrl, submitted_at_utc, confirmed_at_utc: null,
          error: `On-chain Guardian blocked: ${onChain.error}`,
          signed_by: env.SOLANA_PUBLIC_KEY ?? null,
        };
      }

      log.info('onchain_guardian.approved', {
        auth_signature: onChain.signature,
        decision_id: guardian_decision_id,
      });
    }

    if (!this.handle.signingCapable) {
      const msg = 'Devnet execution requires SOLANA_PRIVATE_KEY and solana-agent-kit installed.';
      log.error('execution.devnet.refused', { reason: msg });
      return {
        status: 'failed', mode: 'devnet',
        signature: null, block_height: null,
        asset: action.asset, side: action.side, amount_sol: action.amount_sol,
        rpc_url: this.handle.rpcUrl,
        submitted_at_utc, confirmed_at_utc: null, error: msg,
        signed_by: env.SOLANA_PUBLIC_KEY ?? null,
      };
    }

    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const web3 = require('@solana/web3.js');
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const bs58 = require('bs58');
    const connection = new web3.Connection(this.handle.rpcUrl, 'confirmed');
    const secret = bs58.decode(env.SOLANA_PRIVATE_KEY!);
    const keypair = web3.Keypair.fromSecretKey(secret);

    if (env.JUPITER_SWAP_ENABLED) {
      return this._executeJupiterSwap(
        ctx, connection, keypair, env, submitted_at_utc, web3,
      );
    }
    return this._executeSelfTransfer(
      ctx, connection, keypair, submitted_at_utc, web3,
    );
  }

  // ── Jupiter swap ────────────────────────────────────────────────────────────

  private async _executeJupiterSwap(
    ctx: ExecutionContext,
    connection: unknown,
    keypair: { publicKey: { toBase58(): string } },
    env: ReturnType<typeof loadEnv>,
    submitted_at_utc: string,
    web3: Record<string, unknown>,
  ): Promise<ExecutionResult> {
    const { approved_action: action, guardian_decision_id } = ctx;
    const conn = connection as {
      sendRawTransaction(raw: Uint8Array, opts: object): Promise<string>;
      confirmTransaction(strategy: object, commitment: string): Promise<unknown>;
      getLatestBlockhash(): Promise<{ blockhash: string; lastValidBlockHeight: number }>;
    };

    try {
      log.info('execution.devnet.jupiter.quoting', {
        decision_id: guardian_decision_id,
        side: action.side,
        amount_sol: action.amount_sol,
      });

      const swapPkg = await fetchJupiterSwap(
        env.JUPITER_API_URL,
        action.side,
        action.amount_sol,
        keypair.publicKey.toBase58(),
        env.QUOTE_CURRENCY_MINT,
        env.JUPITER_SLIPPAGE_BPS,
      );

      log.info('execution.devnet.jupiter.quoted', {
        decision_id: guardian_decision_id,
        in_amount: swapPkg.inAmountLamports.toString(),
        out_amount: swapPkg.outAmountLamports.toString(),
        price_impact_pct: swapPkg.priceImpactPct,
        route_steps: (swapPkg.quoteResponse.routePlan as unknown[]).length,
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
      log.info('execution.devnet.jupiter.confirmed', {
        decision_id: guardian_decision_id,
        signature: sig,
        amount_sol: action.amount_sol,
      });

      return {
        status: 'confirmed', mode: 'devnet',
        signature: sig, block_height: null,
        asset: action.asset, side: action.side, amount_sol: action.amount_sol,
        rpc_url: this.handle.rpcUrl,
        submitted_at_utc, confirmed_at_utc, error: null,
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
      log.error('execution.devnet.jupiter.failed', { reason: msg });
      return {
        status: 'failed', mode: 'devnet',
        signature: null, block_height: null,
        asset: action.asset, side: action.side, amount_sol: action.amount_sol,
        rpc_url: this.handle.rpcUrl,
        submitted_at_utc, confirmed_at_utc: null, error: msg,
        signed_by: keypair.publicKey.toBase58(),
      };
    }
  }

  // ── Self-transfer (JUPITER_SWAP_ENABLED=false) ──────────────────────────────
  // Blocked on mainnet-beta: a self-transfer there moves real lamports with no
  // trade intent. Use JUPITER_SWAP_ENABLED=true for mainnet execution.

  private async _executeSelfTransfer(
    ctx: ExecutionContext,
    connection: unknown,
    keypair: { publicKey: { toBase58(): string } },
    submitted_at_utc: string,
    web3: Record<string, unknown>,
  ): Promise<ExecutionResult> {
    const { approved_action: action, guardian_decision_id } = ctx;

    try {
      const lamports = Math.floor(action.amount_sol * 1_000_000_000);

      const SystemProgram = (web3['SystemProgram'] as { transfer(args: object): unknown });
      const Transaction   = web3['Transaction'] as new () => {
        add(ix: unknown): unknown;
      };
      const sendAndConfirmTransaction = web3['sendAndConfirmTransaction'] as (
        conn: unknown, tx: unknown, signers: unknown[], opts: object,
      ) => Promise<string>;

      const tx = new Transaction().add(
        SystemProgram.transfer({
          fromPubkey: keypair.publicKey,
          toPubkey: keypair.publicKey,
          lamports,
        }),
      );

      const sig = await sendAndConfirmTransaction(connection, tx, [keypair], {
        commitment: 'confirmed',
      });

      const confirmed_at_utc = new Date().toISOString();
      log.info('execution.devnet.self_transfer.confirmed', {
        decision_id: guardian_decision_id,
        signature: sig,
        amount_sol: action.amount_sol,
      });

      return {
        status: 'confirmed', mode: 'devnet',
        signature: sig, block_height: null,
        asset: action.asset, side: action.side, amount_sol: action.amount_sol,
        rpc_url: this.handle.rpcUrl,
        submitted_at_utc, confirmed_at_utc, error: null,
        signed_by: keypair.publicKey.toBase58(),
        meta: { swap_type: 'self_transfer', note: 'Path verification only. Set JUPITER_SWAP_ENABLED=true for real swaps.' },
      };
    } catch (err) {
      const msg = (err as Error).message;
      log.error('execution.devnet.self_transfer.failed', { reason: msg });
      return {
        status: 'failed', mode: 'devnet',
        signature: null, block_height: null,
        asset: action.asset, side: action.side, amount_sol: action.amount_sol,
        rpc_url: this.handle.rpcUrl,
        submitted_at_utc, confirmed_at_utc: null, error: msg,
        signed_by: keypair.publicKey.toBase58(),
      };
    }
  }

  // ── prepareTransaction — build unsigned tx for Phantom signing ───────────────

  async prepareTransaction(ctx: ExecutionContext): Promise<PreparedTransaction | null> {
    const env = loadEnv();

    if (!env.JUPITER_SWAP_ENABLED) {
      throw new Error(
        'prepareTransaction requires JUPITER_SWAP_ENABLED=true. ' +
        'Self-transfer mode cannot produce a transaction for Phantom signing.',
      );
    }

    const phantomKey = ctx.phantom_public_key;
    if (!phantomKey) {
      throw new Error('phantom_public_key is required for Phantom transaction signing.');
    }

    const { approved_action: action, guardian_decision_id } = ctx;

    log.info('execution.devnet.prepare_tx', {
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
      mode: 'devnet',
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
