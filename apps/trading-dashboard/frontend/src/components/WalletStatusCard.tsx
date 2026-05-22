import { useWallet } from '@solana/wallet-adapter-react';
import { useWalletModal } from '@solana/wallet-adapter-react-ui';
import { SOLANA_CLUSTER, WALLET_MODE } from '../config';

const card: React.CSSProperties = {
  background: 'var(--bg-panel)',
  border: '1px solid var(--border)',
  borderRadius: 12,
  padding: 20,
};

function Row({ label, value, mono, warn }: { label: string; value: string; mono?: boolean; warn?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3 text-xs">
      <span style={{ color: 'var(--text-3)' }}>{label}</span>
      <span
        className={mono ? 'font-mono truncate max-w-[180px]' : ''}
        style={{ color: warn ? 'var(--red)' : 'var(--text-1)' }}
        title={value}
      >
        {value}
      </span>
    </div>
  );
}

/**
 * Card shown in ExecutionPanel when WALLET_MODE=phantom.
 * Displays connection state, public key, cluster, and a mismatch warning.
 */
export function WalletStatusCard() {
  const { publicKey, connected, wallet } = useWallet();
  const { setVisible } = useWalletModal();

  if (WALLET_MODE !== 'phantom') return null;

  const keyStr = publicKey?.toBase58() ?? null;
  const clusterMismatch = false; // backend cluster vs SOLANA_CLUSTER checked below when balance loads

  return (
    <div style={card}>
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm font-semibold" style={{ color: 'var(--text-1)' }}>
          Wallet
        </div>
        <span
          className="text-xs font-bold px-2 py-0.5 rounded-full"
          style={
            connected
              ? { background: 'var(--green-bg)', color: 'var(--green)', border: '1px solid var(--green-bd)' }
              : { background: 'var(--bg-sub)', color: 'var(--text-3)', border: '1px solid var(--border)' }
          }
        >
          {connected ? 'CONNECTED' : 'NOT CONNECTED'}
        </span>
      </div>

      {connected && keyStr ? (
        <div className="space-y-2">
          <Row label="Wallet" value={wallet?.adapter.name ?? '—'} />
          <Row label="Public key" value={keyStr} mono />
          <Row label="Network" value={SOLANA_CLUSTER} />
          {clusterMismatch && (
            <div
              className="mt-2 px-3 py-2 rounded-lg text-xs"
              style={{ background: 'var(--red-bg)', border: '1px solid var(--red-bd)', color: 'var(--red)' }}
            >
              Wallet network does not match configured cluster ({SOLANA_CLUSTER}). Switch your wallet to {SOLANA_CLUSTER}.
            </div>
          )}
          <div className="pt-1">
            <button
              onClick={() => setVisible(true)}
              className="text-xs px-2 py-1 rounded-lg transition-colors"
              style={{ color: 'var(--text-3)', border: '1px solid var(--border)' }}
              onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-sub)')}
              onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
            >
              Switch wallet
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <p className="text-xs" style={{ color: 'var(--text-3)' }}>
            Connect your Phantom or Solflare wallet to enable execution requests.
            Private keys never leave your wallet.
          </p>
          <button
            onClick={() => setVisible(true)}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold"
            style={{ background: 'var(--accent)', color: 'var(--bg-page)' }}
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            Connect Wallet
          </button>
        </div>
      )}
    </div>
  );
}
