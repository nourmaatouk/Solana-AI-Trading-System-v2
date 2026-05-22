import { useTradingStore, type ExecutionHistoryEntry } from '../store';

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

      {/* Reason / signature */}
      <span
        className="truncate font-mono"
        style={{ color: 'var(--text-3)' }}
        title={entry.signature ?? entry.blocked_by ?? entry.error ?? entry.reasons[0] ?? ''}
      >
        {entry.signature
          ? `${entry.signature.slice(0, 8)}…`
          : entry.blocked_by ?? entry.error ?? entry.reasons[0] ?? '—'}
      </span>
    </div>
  );
}

export function ExecutionHistory() {
  const history = useTradingStore((s) => s.executionHistory);

  return (
    <div>
      {/* Column headers */}
      <div
        className="grid gap-2 px-3 pb-1 text-[11px]"
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

      {history.length === 0 ? (
        <p className="text-xs px-3 py-4 text-center" style={{ color: 'var(--text-3)' }}>
          No executions yet — trades will appear here in real time.
        </p>
      ) : (
        <div className="space-y-1 max-h-72 overflow-y-auto pr-1">
          {history.map((entry, i) => (
            <Row key={`${entry.request_received_at_utc}-${i}`} entry={entry} />
          ))}
        </div>
      )}
    </div>
  );
}
