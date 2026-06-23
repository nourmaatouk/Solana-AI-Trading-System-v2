import { useState } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { useTradingStore, type ExecutionHistoryEntry } from '../store';
import { explorerTxUrl } from '../utils/solanaExplorer';

// Column layout shared by the header and every row so they never drift.
// Mirrors the paper-mode table: Time | Signal | Action | Guardian | Detail.
const GRID_COLUMNS = '70px 64px 52px 78px 1fr';

// Model signal direction, the same labels paper-mode uses (UP/DOWN/NEUTRAL).
const SIGNAL_LABEL: Record<string, string> = { BUY: 'UP', SELL: 'DOWN', HOLD: 'NEUTRAL' };

function dirColor(a: string | null | undefined): string {
  const u = (a ?? '').toUpperCase();
  if (u === 'BUY' || u === 'UP') return 'var(--green)';
  if (u === 'SELL' || u === 'DOWN') return 'var(--red)';
  return 'var(--accent)'; // HOLD / NEUTRAL
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '—';
  }
}

function Row({ entry }: { entry: ExecutionHistoryEntry }) {
  const approved = entry.decision === 'APPROVE';
  // A HOLD is not a rejected trade — the model deliberately chose not to trade.
  // Keep it neutral ("hold") rather than a red "blocked" row.
  const isHold = !approved && entry.blocked_by === 'ACTION_HOLD';

  // Intended action survives a block via signal_action (the Guardian's
  // sanitized_action — and therefore `side` — is null on a block). Older rows
  // without signal_action fall back to side, then to HOLD for ACTION_HOLD.
  const action = entry.signal_action ?? entry.side ?? (isHold ? 'HOLD' : null);
  const signal = action ? (SIGNAL_LABEL[action] ?? action) : '—';

  // Guardian verdict column (paper-mode style: ALLOW / BLOCKED, plus neutral hold).
  const guardianLabel = approved ? 'ALLOW' : isHold ? 'hold' : 'BLOCKED';
  const guardianColor = approved
    ? entry.execution_status === 'confirmed' ? 'var(--green)' : 'var(--accent)'
    : isHold ? 'var(--text-3)' : 'var(--red)';

  return (
    <div
      className="grid gap-2 px-3 py-2 text-xs rounded-lg items-center"
      style={{
        gridTemplateColumns: GRID_COLUMNS,
        background: 'var(--bg-sub)',
        border: '1px solid var(--border)',
      }}
    >
      {/* Time */}
      <span style={{ color: 'var(--text-3)', fontVariantNumeric: 'tabular-nums' }}>
        {formatTime(entry.request_received_at_utc)}
      </span>

      {/* Signal (model direction) */}
      <span className="font-semibold" style={{ color: dirColor(action) }}>
        {signal}
      </span>

      {/* Action (intended trade) */}
      <span className="font-semibold" style={{ color: dirColor(action) }}>
        {action ?? '—'}
      </span>

      {/* Guardian verdict */}
      <span className="font-semibold" style={{ color: guardianColor }}>
        {guardianLabel}
      </span>

      {/* Detail — on-chain signature for confirmed trades, else the reason */}
      {entry.signature ? (
        <a
          href={explorerTxUrl(entry.signature, entry.mode)}
          target="_blank"
          rel="noreferrer"
          className="truncate font-mono hover:underline"
          style={{ color: 'var(--accent)' }}
          title={`View ${entry.signature} on Solana Explorer`}
        >
          {entry.amount_sol != null ? `${entry.amount_sol} SOL · ` : ''}{entry.signature.slice(0, 8)}… ↗
        </a>
      ) : (
        <span
          className="truncate"
          style={{ color: 'var(--text-3)' }}
          title={entry.blocked_by ?? entry.error ?? entry.reasons[0] ?? ''}
        >
          {isHold
            ? 'No trade — model signal is HOLD'
            : !approved
              ? `Blocked: ${entry.blocked_by ?? entry.reasons[0] ?? 'Guardian'}`
              : entry.error ?? entry.reasons[0] ?? 'all checks passed'}
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
            gridTemplateColumns: GRID_COLUMNS,
            color: 'var(--text-3)',
          }}
        >
          <span>Time</span>
          <span>Signal</span>
          <span>Action</span>
          <span>Guardian</span>
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
