/**
 * Helpers for building Solana Explorer links. Cluster is derived from the
 * execution mode / wallet cluster reported by the backend — never hard-coded.
 * The explorer base URL comes from config (VITE_SOLANA_EXPLORER_BASE_URL).
 */
import { SOLANA_EXPLORER_BASE_URL } from '../config';

/** Map an execution mode (paper|devnet|mainnet) or cluster string to an explorer cluster. */
export function clusterForMode(modeOrCluster?: string | null): string {
  const v = (modeOrCluster ?? '').toLowerCase();
  if (v === 'mainnet' || v === 'mainnet-beta') return 'mainnet-beta';
  if (v === 'testnet') return 'testnet';
  return 'devnet';
}

/** Explorer URL for a transaction signature on the given cluster. */
export function explorerTxUrl(signature: string, modeOrCluster?: string | null): string {
  const cluster = clusterForMode(modeOrCluster);
  const suffix = cluster === 'mainnet-beta' ? '' : `?cluster=${cluster}`;
  return `${SOLANA_EXPLORER_BASE_URL}/tx/${signature}${suffix}`;
}
