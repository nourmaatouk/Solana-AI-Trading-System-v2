/**
 * Read-only wallet operations. These work without a private key — they only
 * need a public key and an RPC URL.
 */
import { buildSolanaClient, type SolanaClientHandle } from './client';
import { loadEnv } from '../config/env';
import { log } from '../logger';
import type { WalletInfo } from './types';

export class WalletService {
  private readonly handle: SolanaClientHandle;

  constructor(handle?: SolanaClientHandle) {
    this.handle = handle ?? buildSolanaClient();
  }

  getInfo(): WalletInfo {
    return {
      public_key: this.handle.publicKey,
      cluster: this.handle.cluster,
      rpc_url: this.handle.rpcUrl,
      signing_capable: this.handle.signingCapable,
    };
  }

  /**
   * Returns SOL balance via RPC. Returns null when:
   *   - no public key is configured, or
   *   - the @solana/web3.js package isn't installed (we degrade gracefully
   *     so the rest of the dashboard keeps working).
   */
  async getBalanceSol(): Promise<number | null> {
    const pubkey = this.handle.publicKey;
    if (!pubkey) {
      log.warn('wallet.balance.no_pubkey');
      return null;
    }

    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const web3 = require('@solana/web3.js');
      const conn = new web3.Connection(this.handle.rpcUrl, 'confirmed');
      const lamports = await conn.getBalance(new web3.PublicKey(pubkey));
      const sol = lamports / 1_000_000_000;
      log.debug('wallet.balance', { public_key: pubkey, sol });
      return sol;
    } catch (err) {
      log.warn('wallet.balance.failed', { reason: (err as Error).message });
      return null;
    }
  }
}

/**
 * Convenience: build a wallet service tied to env defaults.
 */
export function defaultWalletService(): WalletService {
  loadEnv(); // surfaces config errors early
  return new WalletService();
}
