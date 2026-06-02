import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useWallet } from '@solana/wallet-adapter-react';
import { useWalletModal } from '@solana/wallet-adapter-react-ui';
import { useTradingStore } from '../store';
import { useAISignal } from '../hooks/useAISignal';
import { SignalBreakdown } from '../components/SignalBreakdown';
import {
  fetchPaperTrading,
  fetchHistory,
  executePaperTrade,
  fetchScheduler,
  resumeScheduler,
  stopScheduler,
} from '../services/api';
import { RISK_LEVEL_CAPS, DASHBOARD_POLL_INTERVAL_MS, WALLET_MODE } from '../config';

type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH';
const RISK_LEVELS: RiskLevel[] = ['LOW', 'MEDIUM', 'HIGH'];
const card = { background: 'var(--bg-panel)', border: '1px solid var(--border)' } as React.CSSProperties;

interface PaperState {
  metrics?: { n_trades?: number; win_rate?: number; total_return?: number; max_drawdown?: number; sharpe?: number; equity?: number };
  open_position?: { direction?: string; entry_price?: number; position_size?: number; risk_level?: string } | null;
  last_action?: string;
  last_signal?: string;
  guardian?: { decision?: string; reason?: string; ppo_action?: string | null; confidence?: number; fused_score?: number };
  reason?: string;
  risk_level?: string;
  status?: string;
  signal_age_minutes?: number;
}
interface HistoryEntry {
  run_utc?: string; signal?: string; action?: string;
  guardian_decision?: string; guardian_reason?: string;
  equity?: number; total_return?: number;
}
interface SchedulerState { status?: string; next_run_utc?: string | null; interval_seconds?: number; mode?: string }

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const pct = (v: number | null, dp = 1) => (v === null ? '—' : `${(v * 100).toFixed(dp)}%`);

function actionColor(a?: string | null): string {
  const u = (a ?? '').toUpperCase();
  return u === 'BUY' ? 'var(--green)' : u === 'SELL' ? 'var(--red)' : 'var(--accent)';
}

export function PaperTrading() {
  useAISignal(); // keep the store signal fresh so SignalBreakdown + header are live
  const ticker = useTradingStore((s) => s.ticker);
  const signal = useTradingStore((s) => s.signal);
  const setMode = useTradingStore((s) => s.setMode);
  const navigate = useNavigate();
  const { connected, disconnect, wallet, publicKey } = useWallet();
  const { setVisible } = useWalletModal();

  const [paper, setPaper] = useState<PaperState | null>(null);
  const [history, setHistory] = useState<HistoryEntry[] | null>(null);
  const [scheduler, setScheduler] = useState<SchedulerState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [risk, setRisk] = useState<RiskLevel>('MEDIUM');
  const [running, setRunning] = useState(false);
  const [runMsg, setRunMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [p, h, s] = await Promise.allSettled([fetchPaperTrading(), fetchHistory(), fetchScheduler()]);
    if (p.status === 'fulfilled') {
      setPaper(p.value as PaperState);
      const rl = (p.value as PaperState).risk_level;
      if (rl && RISK_LEVELS.includes(rl.toUpperCase() as RiskLevel)) setRisk(rl.toUpperCase() as RiskLevel);
      setError(null);
    } else {
      setError('Could not reach the paper-trading API. Is the backend running?');
    }
    if (h.status === 'fulfilled') setHistory((h.value as HistoryEntry[]) ?? []);
    if (s.status === 'fulfilled') setScheduler(s.value as SchedulerState);
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, DASHBOARD_POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [load]);

  const runNow = async () => {
    setRunning(true);
    setRunMsg('Running the pipeline and recording a paper trade… (~20s)');
    try {
      await executePaperTrade({ risk_level: risk });
      setRunMsg('Done — refreshing results.');
      await load();
    } catch (e) {
      setRunMsg(`Failed: ${(e as Error).message}`);
    } finally {
      setRunning(false);
      setTimeout(() => setRunMsg(null), 4000);
    }
  };

  const isAuto = !!scheduler?.next_run_utc || /run|active|continuous/i.test(scheduler?.mode ?? scheduler?.status ?? '');
  const toggleAuto = async () => {
    try {
      if (isAuto) await stopScheduler(); else await resumeScheduler();
      await load();
    } catch { /* surfaced via status refresh */ }
  };

  // ── Derived metrics (all from real backend data) ──
  const m = paper?.metrics ?? {};
  const equity = num(m.equity);
  const totalReturn = num(m.total_return);
  const winRate = num(m.win_rate);
  const nTrades = num(m.n_trades);
  const sharpe = num(m.sharpe);
  const maxDd = num(m.max_drawdown);
  const wins = winRate !== null && nTrades !== null ? Math.round(winRate * nTrades) : null;
  const losses = wins !== null && nTrades !== null ? nTrades - wins : null;

  const pos = paper?.open_position;
  const posSide = pos?.direction ? (pos.direction.toUpperCase() === 'UP' ? 'LONG' : 'SHORT') : null;
  const gDecision = paper?.guardian?.decision ?? null;
  const gReason = paper?.guardian?.reason ?? paper?.reason ?? null;
  const ppoAction = paper?.guardian?.ppo_action ?? null;
  const aiAction = (signal?.action ?? signal?.signal ?? paper?.last_action ?? 'HOLD').toUpperCase();
  const aiConf = num(signal?.confidence) ?? num(paper?.guardian?.confidence);

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold" style={{ color: 'var(--text-1)' }}>Paper Trading</h1>
            <span className="text-xs font-bold px-2 py-0.5 rounded"
              style={{ color: '#0ea5e9', border: '1px solid #0ea5e9' }}>PAPER MODE</span>
          </div>
          <p className="text-sm mt-1" style={{ color: 'var(--text-3)' }}>
            No wallet required. No real money. Simulated trading only — driven by the live AI signal and validated by the Guardian.
          </p>
        </div>
        <div className="text-right">
          <div className="text-xs uppercase tracking-widest" style={{ color: 'var(--text-3)' }}>SOL/USDT</div>
          <div className="text-lg font-bold font-mono" style={{ color: 'var(--text-1)' }}>
            {ticker ? `$${parseFloat(ticker.price).toFixed(2)}` : '—'}
          </div>
        </div>
      </div>

      {/* Current AI action + Guardian status strip */}
      <div className="rounded-xl p-4 flex flex-wrap items-center gap-x-6 gap-y-3" style={card}>
        <div>
          <div className="text-xs uppercase tracking-widest mb-1" style={{ color: 'var(--text-3)' }}>Current AI action</div>
          <div className="flex items-center gap-2">
            <span className="text-base font-bold" style={{ color: actionColor(aiAction) }}>{aiAction}</span>
            {aiConf !== null && <span className="text-sm font-mono" style={{ color: 'var(--text-2)' }}>{pct(aiConf)}</span>}
            {ppoAction && (
              <span className="text-xs font-bold px-1.5 py-0.5 rounded"
                style={{ background: 'rgba(139,92,246,0.2)', color: '#a78bfa' }}>PPO DECIDER</span>
            )}
          </div>
        </div>
        <div className="h-8 w-px" style={{ background: 'var(--border)' }} />
        <div>
          <div className="text-xs uppercase tracking-widest mb-1" style={{ color: 'var(--text-3)' }}>Guardian</div>
          {gDecision ? (
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full border"
              style={gDecision === 'APPROVE'
                ? { color: 'var(--green)', background: 'var(--green-bg)', borderColor: 'var(--green-bd)' }
                : { color: 'var(--red)', background: 'var(--red-bg)', borderColor: 'var(--red-bd)' }}>
              {gDecision === 'APPROVE' ? '✓ APPROVED' : `⛔ BLOCKED — ${gReason ?? 'see reason'}`}
            </span>
          ) : <span className="text-xs" style={{ color: 'var(--text-3)' }}>{loading ? 'loading…' : 'no decision yet'}</span>}
        </div>
      </div>

      {error && (
        <div className="text-sm px-4 py-3 rounded-lg"
          style={{ color: 'var(--red)', background: 'var(--red-bg)', border: '1px solid var(--red-bd)' }}>{error}</div>
      )}

      {/* Wallet banner — connect prompt or connected state with disconnect button */}
      {WALLET_MODE === 'phantom' && (
        connected ? (
          <div
            className="flex items-center justify-between gap-4 px-4 py-3 rounded-xl flex-wrap"
            style={{ background: 'var(--green-bg)', border: '1px solid var(--green-bd)' }}
          >
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: 'var(--green)' }} />
              <div>
                <div className="text-sm font-semibold" style={{ color: 'var(--green)' }}>
                  {wallet?.adapter.name ?? 'Wallet'} connected
                </div>
                {publicKey && (
                  <div className="text-xs font-mono mt-0.5" style={{ color: 'var(--text-3)' }}>
                    {publicKey.toBase58().slice(0, 6)}…{publicKey.toBase58().slice(-6)}
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <button
                onClick={() => { setMode('devnet'); navigate('/execution'); }}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold"
                style={{ background: 'var(--accent)', color: 'var(--bg-page)' }}
              >
                Go to Devnet →
              </button>
              <button
                onClick={() => disconnect()}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold"
                style={{ background: 'var(--red-bg)', border: '1px solid var(--red-bd)', color: 'var(--red)' }}
              >
                Disconnect
              </button>
            </div>
          </div>
        ) : (
          <div
            className="flex items-center justify-between gap-4 px-4 py-3 rounded-xl flex-wrap"
            style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)' }}
          >
            <div>
              <div className="text-sm font-semibold" style={{ color: 'var(--accent)' }}>
                Ready for real devnet trades?
              </div>
              <div className="text-xs mt-0.5" style={{ color: 'var(--text-2)' }}>
                Connect a wallet to switch to devnet mode and execute Guardian-validated on-chain transactions.
              </div>
            </div>
            <button
              onClick={() => setVisible(true)}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold flex-shrink-0"
              style={{ background: 'var(--accent)', color: 'var(--bg-page)' }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
              Connect Wallet
            </button>
          </div>
        )
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Left: signal breakdown (reused) */}
        <SignalBreakdown />

        {/* Right: metrics + position + controls */}
        <div className="space-y-5">
          {/* Performance metrics */}
          <div className="rounded-xl p-5" style={card}>
            <div className="text-xs font-semibold tracking-widest uppercase mb-4" style={{ color: 'var(--text-3)' }}>Performance (simulated)</div>
            {loading && !paper ? (
              <div className="h-20 rounded-lg animate-pulse" style={{ background: 'var(--bg-sub)' }} />
            ) : (
              <div className="grid grid-cols-3 gap-4">
                <Metric label="Equity (×, base 1.00)" value={equity !== null ? equity.toFixed(4) : '—'} />
                <Metric label="Total return" value={pct(totalReturn)} color={totalReturn !== null ? (totalReturn >= 0 ? 'var(--green)' : 'var(--red)') : undefined} />
                <Metric label="Win rate" value={pct(winRate)} />
                <Metric label="Trades" value={nTrades !== null ? String(nTrades) : '—'} />
                <Metric label="Win / Loss" value={wins !== null ? `${wins} / ${losses}` : '—'} />
                <Metric label="Sharpe" value={sharpe !== null ? sharpe.toFixed(2) : '—'} />
                <Metric label="Max drawdown" value={pct(maxDd)} />
              </div>
            )}
            <p className="text-xs mt-3" style={{ color: 'var(--text-3)' }}>
              Equity is normalised (1.00 = starting balance). Position size scales with model confidence,
              capped by your risk level.
            </p>
          </div>

          {/* Open position */}
          <div className="rounded-xl p-5" style={card}>
            <div className="text-xs font-semibold tracking-widest uppercase mb-3" style={{ color: 'var(--text-3)' }}>Open position</div>
            {pos && posSide ? (
              <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
                <span className="font-bold" style={{ color: posSide === 'LONG' ? 'var(--green)' : 'var(--red)' }}>{posSide}</span>
                <span style={{ color: 'var(--text-2)' }}>entry ${num(pos.entry_price)?.toFixed(2) ?? '—'}</span>
                <span style={{ color: 'var(--text-2)' }}>size {pos.position_size != null ? pct(pos.position_size) : '—'} of equity</span>
              </div>
            ) : (
              <div className="text-sm" style={{ color: 'var(--text-3)' }}>Flat — no open position.</div>
            )}
          </div>

          {/* Controls */}
          <div className="rounded-xl p-5 space-y-4" style={card}>
            <div className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--text-3)' }}>Controls</div>

            {/* Risk level */}
            <div>
              <div className="text-sm mb-2" style={{ color: 'var(--text-1)' }}>Risk level <span className="text-xs" style={{ color: 'var(--text-3)' }}>(position-size cap)</span></div>
              <div className="flex gap-2">
                {RISK_LEVELS.map((r) => (
                  <button key={r} onClick={() => setRisk(r)}
                    className="flex-1 py-2 text-xs font-semibold rounded-lg border transition-colors"
                    style={risk === r
                      ? { color: 'var(--accent)', background: 'var(--accent-bg)', borderColor: 'var(--accent-bd)' }
                      : { color: 'var(--text-3)', background: 'var(--bg-sub)', borderColor: 'var(--border)' }}>
                    {r}<div className="text-[10px] opacity-70">{RISK_LEVEL_CAPS[r]} equity</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Auto / Manual */}
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm" style={{ color: 'var(--text-1)' }}>Auto trading</div>
                <div className="text-xs" style={{ color: 'var(--text-3)' }}>
                  {isAuto
                    ? `On — pipeline runs every ${scheduler?.interval_seconds ? Math.round(scheduler.interval_seconds / 60) : 15} min`
                    : 'Off — run trades manually below'}
                </div>
              </div>
              <button onClick={toggleAuto}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold border"
                style={isAuto
                  ? { color: 'var(--green)', background: 'var(--green-bg)', borderColor: 'var(--green-bd)' }
                  : { color: 'var(--text-2)', background: 'var(--bg-sub)', borderColor: 'var(--border)' }}>
                {isAuto ? 'AUTO' : 'MANUAL'}
              </button>
            </div>

            {/* Manual run */}
            <button onClick={runNow} disabled={running}
              className="w-full py-2.5 rounded-lg text-sm font-semibold transition-opacity"
              style={{ background: 'var(--accent)', color: '#fff', opacity: running ? 0.6 : 1, cursor: running ? 'not-allowed' : 'pointer' }}>
              {running ? 'Running…' : 'Run paper trade now'}
            </button>
            {runMsg && <p className="text-xs text-center" style={{ color: 'var(--text-3)' }}>{runMsg}</p>}
            <p className="text-xs" style={{ color: 'var(--text-3)' }}>
              A run executes the <strong>current AI signal</strong> through the Guardian, then records the
              result (or a BLOCK with its reason). The direction is decided by the AI — it can't be overridden manually.
            </p>
          </div>
        </div>
      </div>

      {/* History */}
      <div className="rounded-xl p-5" style={card}>
        <div className="text-xs font-semibold tracking-widest uppercase mb-4" style={{ color: 'var(--text-3)' }}>
          Paper trade history {history && `(${history.length})`}
        </div>
        {loading && !history ? (
          <div className="space-y-2">{[0, 1, 2].map((i) => <div key={i} className="h-9 rounded animate-pulse" style={{ background: 'var(--bg-sub)' }} />)}</div>
        ) : history && history.length === 0 ? (
          <div className="text-sm text-center py-6" style={{ color: 'var(--text-3)' }}>
            No paper trades yet. Click “Run paper trade now” to record your first one.
          </div>
        ) : history ? (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr style={{ color: 'var(--text-3)' }} className="text-left">
                  <th className="py-1.5 pr-3">Time</th>
                  <th className="py-1.5 pr-3">Signal</th>
                  <th className="py-1.5 pr-3">Action</th>
                  <th className="py-1.5 pr-3">Guardian</th>
                  <th className="py-1.5 pr-3">Reason / explanation</th>
                  <th className="py-1.5 pr-3 text-right">Equity</th>
                </tr>
              </thead>
              <tbody>
                {[...history].reverse().map((h, i) => {
                  const blocked = (h.guardian_decision ?? '').toUpperCase() === 'BLOCK';
                  return (
                    <tr key={`${h.run_utc}-${i}`} style={{ borderTop: '1px solid var(--border)' }}>
                      <td className="py-1.5 pr-3 font-mono" style={{ color: 'var(--text-3)' }}>
                        {h.run_utc ? new Date(h.run_utc).toLocaleTimeString() : '—'}
                      </td>
                      <td className="py-1.5 pr-3" style={{ color: 'var(--text-2)' }}>{h.signal ?? '—'}</td>
                      <td className="py-1.5 pr-3 font-semibold" style={{ color: actionColor(h.action) }}>{(h.action ?? '—').toUpperCase()}</td>
                      <td className="py-1.5 pr-3 font-semibold" style={{ color: blocked ? 'var(--red)' : 'var(--green)' }}>
                        {blocked ? 'BLOCKED' : (h.guardian_decision ?? 'APPROVE')}
                      </td>
                      <td className="py-1.5 pr-3" style={{ color: 'var(--text-3)' }}>{h.guardian_reason ?? '—'}</td>
                      <td className="py-1.5 pr-3 text-right font-mono" style={{ color: 'var(--text-2)' }}>
                        {num(h.equity)?.toFixed(4) ?? '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function Metric({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div className="text-xs" style={{ color: 'var(--text-3)' }}>{label}</div>
      <div className="text-base font-bold font-mono" style={{ color: color ?? 'var(--text-1)' }}>{value}</div>
    </div>
  );
}
