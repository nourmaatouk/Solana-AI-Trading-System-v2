import { useState } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { useTradingStore, type ExecutionHistoryEntry } from '../store';
import { explorerTxUrl } from '../utils/solanaExplorer';

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '—';
  }
}

function Row({ entry }: { entry: ExecutionHistoryEntry }) {
  const approved = entry.decision === 'APPROVE';
  const statusColor = approved
    ? entry.execution_status === 'confirmed'
      ? 'var(--green)'
      : 'var(--accent)'
    : 'var(--red)';

  const sideColor =
    entry.side === 'BUY' ? 'var(--green)' : entry.side === 'SELL' ? 'var(--red)' : 'var(--text-3)';

  return (
    <div
      className="grid gap-2 px-3 py-2 text-xs rounded-lg"
      style={{
        gridTemplateColumns: '70px 52px 42px 48px 64px 1fr',
        background: 'var(--bg-sub)',
        border: '1px solid var(--border)',
      }}
    >
      {/* Time */}
      <span style={{ color: 'var(--text-3)', fontVariantNumeric: 'tabular-nums' }}>
        {formatTime(entry.request_received_at_utc)}
      </span>

      {/* Decision badge */}
      <span
        className="font-semibold"
        style={{ color: statusColor }}
      >
        {approved ? (entry.execution_status ?? 'approved') : 'blocked'}
      </span>

      {/* Side */}
      <span className="font-semibold" style={{ color: sideColor }}>
        {entry.side ?? '—'}
      </span>

      {/* Amount */}
      <span style={{ color: 'var(--text-2)', fontVariantNumeric: 'tabular-nums' }}>
        {entry.amount_sol != null ? `${entry.amount_sol} SOL` : '—'}
      </span>

      {/* Mode */}
      <span style={{ color: 'var(--text-3)' }}>
        {entry.mode ?? '—'}
      </span>

      {/* Reason / signature (signature → clickable explorer link) */}
      {entry.signature ? (
        <a
          href={explorerTxUrl(entry.signature, entry.mode)}
          target="_blank"
          rel="noreferrer"
          className="truncate font-mono hover:underline"
          style={{ color: 'var(--accent)' }}
          title={`View ${entry.signature} on Solana Explorer`}
        >
          {entry.signature.slice(0, 8)}… ↗
        </a>
      ) : (
        <span
          className="truncate font-mono"
          style={{ color: 'var(--text-3)' }}
          title={entry.blocked_by ?? entry.error ?? entry.reasons[0] ?? ''}
        >
          {entry.decision === 'BLOCK'
            ? `Blocked before chain: ${entry.blocked_by ?? entry.reasons[0] ?? 'Guardian'}`
            : entry.error ?? entry.reasons[0] ?? '—'}
        </span>
      )}
    </div>
  );
}

export function ExecutionHistory() {
  const history = useTradingStore((s) => s.executionHistory);
  const { connected, publicKey } = useWallet();
  const [filterByWallet, setFilterByWallet] = useState(false);

  const walletPubkey = connected && publicKey ? publicKey.toBase58() : null;
  const visible = filterByWallet && walletPubkey
    ? history.filter((e) => e.signed_by === walletPubkey)
    : history;

  return (
    <div>
      {/* Header row: title filter toggle */}
      <div className="flex items-center justify-between mb-2">
        <div
          className="grid gap-2 px-3 text-[11px] flex-1"
          style={{
            gridTemplateColumns: '70px 52px 42px 48px 64px 1fr',
            color: 'var(--text-3)',
          }}
        >
          <span>Time</span>
          <span>Result</span>
          <span>Side</span>
          <span>Amount</span>
          <span>Mode</span>
          <span>Detail</span>
        </div>
        {walletPubkey && (
          <button
            onClick={() => setFilterByWallet((v) => !v)}
            className="text-[11px] px-2 py-0.5 rounded flex-shrink-0 ml-2"
            style={{
              background: filterByWallet ? 'var(--accent)' : 'var(--bg-sub)',
              color: filterByWallet ? 'var(--bg-page)' : 'var(--text-3)',
              border: `1px solid ${filterByWallet ? 'var(--accent)' : 'var(--border)'}`,
            }}
            title={filterByWallet
              ? `Showing trades signed by ${walletPubkey.slice(0, 6)}…${walletPubkey.slice(-4)}`
              : 'Show only trades signed by connected wallet'}
          >
            {filterByWallet
              ? `${walletPubkey.slice(0, 4)}…${walletPubkey.slice(-4)} ✕`
              : 'My wallet'}
          </button>
        )}
      </div>

      {visible.length === 0 ? (
        <p className="text-xs px-3 py-4 text-center" style={{ color: 'var(--text-3)' }}>
          {filterByWallet && walletPubkey
            ? `No trades signed by ${walletPubkey.slice(0, 6)}…${walletPubkey.slice(-4)} yet.`
            : 'No executions yet — trades will appear here in real time.'}
        </p>
      ) : (
        <div className="space-y-1 max-h-72 overflow-y-auto pr-1">
          {visible.map((entry, i) => (
            <Row key={`${entry.request_received_at_utc}-${i}`} entry={entry} />
          ))}
        </div>
      )}
    </div>
  );
}
