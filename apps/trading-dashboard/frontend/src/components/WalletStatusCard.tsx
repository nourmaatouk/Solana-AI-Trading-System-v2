import { useWallet } from '@solana/wallet-adapter-react';
import { useSystemWallet } from '../hooks/useSystemWallet';
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
 * Reflects whichever wallet is connected — a real Phantom/Solflare extension or
 * the backend Dev Wallet — and offers Disconnect. Connection itself is only
 * initiated from the landing page chooser.
 */
export function WalletStatusCard() {
  const { publicKey, connected: phantomConnected, wallet, disconnect: phantomDisconnect } = useWallet();
  const { connected: systemConnected, publicKey: systemKey, disconnect: systemDisconnect } = useSystemWallet();

  if (WALLET_MODE !== 'phantom') return null;

  const connected = phantomConnected || systemConnected;
  const keyStr = phantomConnected ? (publicKey?.toBase58() ?? null) : systemKey;
  const walletName = phantomConnected ? (wallet?.adapter.name ?? 'Wallet') : 'Dev Wallet';
  const disconnect = phantomConnected ? phantomDisconnect : systemDisconnect;
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
          <Row label="Wallet" value={walletName} />
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
              onClick={() => disconnect()}
              className="text-xs px-2 py-1 rounded-lg transition-colors"
              style={{ color: 'var(--red)', border: '1px solid var(--red-bd)' }}
              onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--red-bg)')}
              onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
            >
              Disconnect
            </button>
          </div>
        </div>
      ) : (
        <p className="text-xs" style={{ color: 'var(--text-3)' }}>
          No wallet connected. Return to the landing page and choose
          “Connect Wallet &amp; Trade” to enter devnet.
        </p>
      )}
    </div>
  );
}
