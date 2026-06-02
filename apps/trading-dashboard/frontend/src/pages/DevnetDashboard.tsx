import { useEffect, useState } from 'react';
import { fetchExecutionStatus, type ExecutionStatusDto } from '../services/executionApi';
import { useAISignal } from '../hooks/useAISignal';
import { MetricCard } from '../components/MetricCard';
import { SignalBreakdown } from '../components/SignalBreakdown';
import { ExecutionHistory } from '../components/ExecutionHistory';
import { WalletStatusCard } from '../components/WalletStatusCard';
import { GuardianBadge } from '../components/GuardianBadge';
import { explorerTxUrl } from '../utils/solanaExplorer';
import { EXECUTION_PANEL_REFRESH_MS } from '../config';

const card = { background: 'var(--bg-panel)', border: '1px solid var(--border)' } as React.CSSProperties;
const ic = (d: string) => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d={d} />
  </svg>
);

/**
 * Devnet-mode dashboard: a wallet / on-chain view (balance, network, last
 * transaction, Guardian decision, execution history) — distinct from the
 * paper-mode Dashboard. All data comes from /api/execution/status and the
 * WebSocket-fed execution history; nothing is hard-coded.
 */
export function DevnetDashboard() {
  useAISignal(); // keep the store signal fresh for the breakdown
  const [status, setStatus] = useState<ExecutionStatusDto | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const s = await fetchExecutionStatus();
        if (alive) { setStatus(s); setError(null); }
      } catch {
        if (alive) setError('Could not reach the execution API. Is the backend running?');
      } finally {
        if (alive) setLoading(false);
      }
    };
    load();
    const id = setInterval(load, EXECUTION_PANEL_REFRESH_MS);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const w = status?.wallet;
  // Guard against old flat-format last_execution records written before the
  // schema fix (they had guardian fields at root, not nested under .guardian).
  const lastRaw = status?.last_execution;
  const last = lastRaw && (lastRaw as { guardian?: unknown }).guardian ? lastRaw : null;
  const exec = last?.execution ?? null;
  const eStop = status?.emergency_stop.enabled ?? false;
  const dash = (v: unknown) => (v === null || v === undefined ? '—' : String(v));

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5" style={{ background: 'var(--bg-page)' }}>
      <div>
        <h1 className="text-xl font-bold" style={{ color: 'var(--text-1)' }}>Devnet Dashboard</h1>
        <p className="text-sm mt-1" style={{ color: 'var(--text-3)' }}>
          Live on-chain execution on Solana's test network. Every trade is Guardian-validated before it's signed —
          mainnet stays disabled.
        </p>
      </div>

      {error && (
        <div className="text-sm px-4 py-3 rounded-lg"
          style={{ color: 'var(--red)', background: 'var(--red-bg)', border: '1px solid var(--red-bd)' }}>{error}</div>
      )}

      {/* Metric cards (wallet / network / safety) */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard
          title="Wallet Balance"
          value={loading && !status ? '…' : w?.balance_sol != null ? `${w.balance_sol.toFixed(4)}` : '—'}
          subtitle="SOL · devnet"
          icon={ic('M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z')}
        />
        <MetricCard
          title="Network"
          value={dash(w?.cluster)}
          subtitle="Solana cluster"
          icon={ic('M13 7h8m0 0v8m0-8l-8 8-4-4-6 6')}
        />
        <MetricCard
          title="Signing"
          value={loading && !status ? '…' : w?.signing_capable ? 'Enabled' : 'Read-only'}
          trend={w?.signing_capable ? 'up' : 'neutral'}
          subtitle="Backend keypair"
          icon={ic('M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z')}
        />
        <MetricCard
          title="Last Execution"
          value={loading && !status ? '…' : exec?.status ?? last?.guardian.decision ?? 'none'}
          trend={exec?.status === 'confirmed' ? 'up' : exec?.status === 'failed' ? 'down' : 'neutral'}
          subtitle="Most recent result"
          icon={ic('M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z')}
        />
        <MetricCard
          title="Mainnet"
          value={status ? (status.allow_mainnet ? 'Allowed' : 'Disabled') : '—'}
          trend={status?.allow_mainnet ? 'down' : 'up'}
          subtitle={status?.allow_mainnet ? '⚠ real funds' : '🔒 protected'}
          icon={ic('M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z')}
        />
        <MetricCard
          title="Emergency Stop"
          value={eStop ? 'ACTIVE' : 'Off'}
          trend={eStop ? 'down' : 'up'}
          subtitle={eStop ? 'execution halted' : 'normal'}
          icon={ic('M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z')}
        />
      </div>

      {/* Two-column: wallet + last tx | signal breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="space-y-4">
          <WalletStatusCard />

          {/* Last transaction */}
          <div className="rounded-xl p-4" style={card}>
            <div className="flex items-center justify-between mb-3">
              <div className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-2)' }}>Last transaction</div>
              {last && <GuardianBadge result={last.guardian} />}
            </div>
            {!last ? (
              <div className="text-xs text-center py-4" style={{ color: 'var(--text-3)' }}>
                No executions yet — approved BUY/SELL signals will appear here with their on-chain signature.
              </div>
            ) : (
              <div className="space-y-1.5 text-xs">
                <Row label="Action" value={exec ? `${exec.side} ${exec.amount_sol} SOL` : (last.guardian.sanitized_action ? `${last.guardian.sanitized_action.side} ${last.guardian.sanitized_action.amount_sol} SOL` : '—')} />
                <Row label="Status" value={dash(exec?.status ?? (last.guardian.approved ? 'approved' : 'blocked'))} />
                <Row label="Mode" value={dash(exec?.mode)} />
                {!last.guardian.approved && <Row label="Blocked by" value={dash(last.guardian.blocked_by)} warn />}
                {exec?.signature && (
                  <div className="flex items-center justify-between gap-3">
                    <span style={{ color: 'var(--text-3)' }}>Signature</span>
                    <a href={explorerTxUrl(exec.signature, exec.mode)} target="_blank" rel="noreferrer"
                      className="font-mono hover:underline truncate" style={{ color: 'var(--accent)' }}>
                      {exec.signature.slice(0, 8)}…{exec.signature.slice(-8)} ↗
                    </a>
                  </div>
                )}
                {last.narration && (
                  <p className="pt-1.5 italic" style={{ color: 'var(--text-3)', borderTop: '1px solid var(--border)' }}>“{last.narration}”</p>
                )}
              </div>
            )}
          </div>
        </div>

        <SignalBreakdown />
      </div>

      {/* Guardian / execution history */}
      <div className="rounded-xl p-4" style={card}>
        <div className="flex flex-col gap-1 mb-3">
          <div className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-2)' }}>
            Guardian decision history
          </div>
          <p className="text-xs" style={{ color: 'var(--text-3)' }}>
            Confirmed rows include a devnet signature. Blocked rows stopped before on-chain execution.
          </p>
        </div>
        <ExecutionHistory />
      </div>
    </div>
  );
}

function Row({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span style={{ color: 'var(--text-3)' }}>{label}</span>
      <span className="font-mono truncate" style={{ color: warn ? 'var(--red)' : 'var(--text-1)' }} title={value}>{value}</span>
    </div>
  );
}
