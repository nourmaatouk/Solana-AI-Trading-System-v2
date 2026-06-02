/**
 * Solana execution types.
 *
 * Defines the abstract Executor interface that each mode (paper/devnet/
 * mainnet) implements. The orchestrator never sees mode-specific code — it
 * only consumes the interface.
 */
import type { ApprovedAction, ExecutionMode } from '../guardian/types';

export interface ExecutionContext {
  approved_action: ApprovedAction;
  guardian_decision_id: string; // Idempotency token from the validator
  /** Signal generation timestamp — used as part of the on-chain signal hash. */
  signal_generated_at_utc?: string;
  /** Raw confidence [0,1] — forwarded to the on-chain Guardian as bps. */
  confidence?: number;
  /** Phantom wallet public key (base58). Present when WALLET_MODE=phantom.
   *  Used by prepareTransaction() to build the tx for client-side signing. */
  phantom_public_key?: string;
}

export type ExecutionStatus = 'simulated' | 'submitted' | 'confirmed' | 'failed';

export interface ExecutionResult {
  status: ExecutionStatus;
  mode: ExecutionMode;
  signature: string | null;
  block_height: number | null;
  asset: string;
  side: 'BUY' | 'SELL';
  amount_sol: number;
  rpc_url: string | null;
  submitted_at_utc: string;
  confirmed_at_utc: string | null;
  error: string | null;
  /** Base58 public key of the keypair that signed the transaction. */
  signed_by: string | null;
  // Free-form mode-specific metadata. Logged but not part of the contract.
  meta?: Record<string, unknown>;
}

/** Unsigned transaction ready for client-side signing (e.g. Phantom). */
export interface PreparedTransaction {
  /** base64-encoded VersionedTransaction. Deserialize, sign, and submit. */
  transaction_base64: string;
  /** The public key this transaction was built for (must be the signer). */
  user_public_key: string;
  mode: ExecutionMode;
  asset: string;
  side: 'BUY' | 'SELL';
  amount_sol: number;
  meta?: Record<string, unknown>;
}

/**
 * Every executor implements this. The orchestrator only ever calls
 * `executor.execute(ctx)` or `executor.prepareTransaction(ctx)` —
 * mode-switching is policy, not orchestrator concern.
 */
export interface ITransactionExecutor {
  readonly mode: ExecutionMode;
  /** Sign and submit the transaction server-side (uses SOLANA_PRIVATE_KEY). */
  execute(ctx: ExecutionContext): Promise<ExecutionResult>;
  /**
   * Build an unsigned VersionedTransaction for client-side signing.
   * Returns null for paper mode (no real transaction exists).
   * Throws when the executor cannot produce a signable transaction
   * (e.g. Jupiter disabled, missing phantom_public_key).
   */
  prepareTransaction(ctx: ExecutionContext): Promise<PreparedTransaction | null>;
}

export interface WalletInfo {
  public_key: string | null;
  cluster: string;
  rpc_url: string;
  signing_capable: boolean;
}
