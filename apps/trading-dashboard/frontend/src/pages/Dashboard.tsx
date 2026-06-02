import { useEffect, useState } from 'react';
import { useTradingStore } from '../store';
import { fetchHistory, fetchPaperTrading } from '../services/api';
import { SmartAssistant } from '../components/SmartAssistant';
import { MetricCard } from '../components/MetricCard';
import { SignalBreakdown } from '../components/SignalBreakdown';
import { DASHBOARD_POLL_INTERVAL_MS, HISTORY_SHOW } from '../config';

interface HistoryEntry {
  run_utc: string;
  action: string;
  fused_score: number;
  equity: number;
  total_return: number;
  win_rate: number;
  sharpe: number;
  n_trades: number;
  latest_close: number;
}

interface PaperTradingMetrics {
  equity: number;
  total_return: number;
  win_rate: number;
  sharpe: number;
  max_drawdown: number;
  n_trades: number;
}

interface PaperTradingResponse {
  metrics: PaperTradingMetrics;
  open_position: unknown;
  last_run_utc: string;
}

// ── Mini equity sparkline ─────────────────────────────────────────────────────

function Sparkline({ data, color }: { data: number[]; color: string }) {
  if (data.length < 2) return <div className="h-10 flex items-end text-xs" style={{ color: 'var(--text-3)' }}>No data yet</div>;
  const W = 300; const H = 40; const pad = 2;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 0.001;
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (W - pad * 2);
    const y = H - pad - ((v - min) / range) * (H - pad * 2);
    return `${x},${y}`;
  });
  const baseline = H - pad - ((1.0 - min) / range) * (H - pad * 2);
  const area = `M${pts[0]} L${pts.join(' L')} L${W - pad},${H - pad} L${pad},${H - pad} Z`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 40 }}>
      <defs>
        <linearGradient id="eq-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <path d={area} fill="url(#eq-grad)" />
      <polyline points={pts.join(' ')} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
      <line x1={pad} y1={baseline} x2={W - pad} y2={baseline} stroke="#2b3139" strokeWidth="1" strokeDasharray="3 2" />
    </svg>
  );
}

// ── Action badge ──────────────────────────────────────────────────────────────

function ActionBadge({ action }: { action: string }) {
  const up = action.toUpperCase();
  const style: React.CSSProperties = up === 'BUY'
    ? { color: 'var(--green)',  background: 'var(--green-bg)',  border: '1px solid var(--green-bd)'  }
    : up === 'SELL'
    ? { color: 'var(--red)',    background: 'var(--red-bg)',    border: '1px solid var(--red-bd)'    }
    : { color: 'var(--text-2)', background: 'var(--bg-sub)',   border: '1px solid var(--border)'    };
  return (
    <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold tracking-wide" style={style}>
      {up}
    </span>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

const card = { background: 'var(--bg-panel)', border: '1px solid var(--border)' } as React.CSSProperties;

export function Dashboard() {
  const signal = useTradingStore((s) => s.signal);

  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [paperTrading, setPaperTrading] = useState<PaperTradingResponse | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [loadingPaper, setLoadingPaper] = useState(true);

  async function refreshData() {
    try {
      const hist = await fetchHistory() as HistoryEntry[];
      setHistory(hist);
    } catch { /* keep previous */ } finally { setLoadingHistory(false); }
    try {
      const pt = await fetchPaperTrading() as unknown as PaperTradingResponse;
      setPaperTrading(pt);
    } catch { /* keep previous */ } finally { setLoadingPaper(false); }
  }

  useEffect(() => {
    void refreshData();
    const id = setInterval(() => { void refreshData(); }, DASHBOARD_POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  const metrics = paperTrading?.metrics ?? null;
  const loading = loadingHistory && loadingPaper;

  const equityPctChange = metrics ? (metrics.equity - 1.0) * 100 : null;
  const totalReturnPct  = metrics ? metrics.total_return * 100 : null;
  const winRatePct      = metrics ? metrics.win_rate * 100 : null;
  const maxDdPct        = metrics ? metrics.max_drawdown * 100 : null;
  const sharpe          = metrics?.sharpe ?? null;
  const sharpeReliable  = (metrics?.n_trades ?? 0) >= 10;

  const activeAll    = history.filter((h) => h.action?.toUpperCase() !== 'HOLD' && h.action);
  const buyCountAll  = activeAll.filter((h) => h.action.toUpperCase() === 'BUY').length;
  const sellCountAll = activeAll.filter((h) => h.action.toUpperCase() === 'SELL').length;

  const activeSignals24h = history.filter((h) => {
    try { return Date.now() - new Date(h.run_utc).getTime() < 86_400_000 && h.action.toUpperCase() !== 'HOLD'; }
    catch { return false; }
  }).length;

  const fusedScore = signal?.fused_score ?? (history.length > 0 ? history[history.length - 1]?.fused_score : null) ?? null;
  const equityCurve = history.map((h) => h.equity);
  const recentHistory = [...history].reverse().slice(0, HISTORY_SHOW);

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5" style={{ background: 'var(--bg-page)' }}>

      {/* Smart Assistant */}
      <SmartAssistant
        fusedScore={fusedScore}
      />

      {/* ── 6 metric cards ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">

        <MetricCard
          title="Paper Portfolio"
          value={loadingPaper ? <span style={{ color: 'var(--text-3)' }}>…</span> : metrics !== null ? metrics.equity.toFixed(4) : '—'}
          trend={equityPctChange !== null ? (equityPctChange >= 0 ? 'up' : 'down') : 'neutral'}
          subtitle={<span style={{ color: 'var(--accent)' }} className="font-medium">Equity · Baseline 1.0 · Simulated</span>}
          icon={<svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>}
        />

        <MetricCard
          title="Total Return"
          value={loadingPaper ? <span style={{ color: 'var(--text-3)' }}>…</span> : totalReturnPct !== null ? `${totalReturnPct >= 0 ? '+' : ''}${totalReturnPct.toFixed(2)}%` : '—'}
          trend={totalReturnPct !== null ? (totalReturnPct >= 0 ? 'up' : 'down') : 'neutral'}
          subtitle="Since inception · Simulated"
          icon={<svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>}
        />

        <MetricCard
          title="Active Signals"
          value={loadingHistory ? <span style={{ color: 'var(--text-3)' }}>…</span> : activeAll.length}
          subtitle={!loadingHistory && activeAll.length > 0 ? `↑ ${buyCountAll} BUY · ↓ ${sellCountAll} SELL · all runs` : 'all scheduler runs'}
          icon={<svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>}
        />

        <MetricCard
          title="Win Rate"
          value={loadingPaper ? <span style={{ color: 'var(--text-3)' }}>…</span> : winRatePct !== null ? `${winRatePct.toFixed(1)}%` : '—'}
          trend={winRatePct !== null ? (winRatePct >= 50 ? 'up' : 'down') : 'neutral'}
          subtitle={`${metrics?.n_trades ?? '—'} trades · Paper Trading`}
          icon={<svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>}
        />

        <MetricCard
          title="Sharpe Ratio"
          value={loadingPaper ? <span style={{ color: 'var(--text-3)' }}>…</span> : sharpe === null ? '—' : !sharpeReliable ? 'N/A' : sharpe.toFixed(2)}
          trend={sharpe !== null && sharpeReliable ? (sharpe >= 0 ? 'up' : 'down') : 'neutral'}
          subtitle={!sharpeReliable && metrics !== null ? `Need ≥10 trades (have ${metrics.n_trades})` : 'Risk-adjusted return'}
          icon={<svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" /></svg>}
        />

        <MetricCard
          title="Max Drawdown"
          value={loadingPaper ? <span style={{ color: 'var(--text-3)' }}>…</span> : maxDdPct !== null ? `${maxDdPct.toFixed(2)}%` : '—'}
          trend={maxDdPct !== null ? (maxDdPct < 5 ? 'up' : 'down') : 'neutral'}
          subtitle="Peak-to-trough · Simulated"
          icon={<svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" /></svg>}
        />
      </div>

      {/* ── Bottom two-column layout ────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        <div className="space-y-4">

          {/* Equity sparkline card */}
          <div className="rounded-xl p-4" style={card}>
            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-2)' }}>Equity Curve</div>
                <div className="text-xs mt-0.5" style={{ color: 'var(--text-3)' }}>
                  {history.length} pipeline runs · baseline 1.0
                </div>
              </div>
              {equityPctChange !== null && (
                <span className="text-sm font-bold" style={{ color: equityPctChange >= 0 ? '#16a34a' : '#dc2626' }}>
                  {equityPctChange >= 0 ? '+' : ''}{equityPctChange.toFixed(3)}%
                </span>
              )}
            </div>
            {equityCurve.length >= 2
              ? <Sparkline data={equityCurve} color={equityPctChange !== null && equityPctChange >= 0 ? '#16a34a' : '#dc2626'} />
              : <div className="h-10 flex items-center justify-center text-xs" style={{ color: 'var(--text-3)' }}>Run the pipeline to see equity data</div>
            }
            <div className="flex justify-between text-[10px] mt-1 px-0.5" style={{ color: 'var(--text-3)' }}>
              {history.length > 0 && (
                <>
                  <span>{new Date(history[0].run_utc).toLocaleDateString()}</span>
                  <span>{new Date(history[history.length - 1].run_utc).toLocaleDateString()}</span>
                </>
              )}
            </div>
          </div>

          {/* Recent signal history */}
          <div className="rounded-xl p-4" style={card}>
            <div className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: 'var(--text-2)' }}>
              Recent Pipeline Runs
            </div>
            {loadingHistory ? (
              <div className="text-xs text-center py-4" style={{ color: 'var(--text-3)' }}>Loading…</div>
            ) : recentHistory.length === 0 ? (
              <div className="text-xs text-center py-4" style={{ color: 'var(--text-3)' }}>
                No pipeline runs yet. Start the scheduler to see history.
              </div>
            ) : (
              <div className="space-y-0">
                {recentHistory.map((entry, i) => {
                  const equityChg = entry.equity != null ? (entry.equity - 1.0) * 100 : null;
                  const time = new Date(entry.run_utc);
                  return (
                    <div key={i} className="flex items-center gap-2 py-2 text-xs"
                      style={{ borderBottom: i < recentHistory.length - 1 ? '1px solid var(--border)' : 'none' }}>
                      <span className="w-14 shrink-0 font-mono" style={{ color: 'var(--text-3)' }}>
                        {time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
                      </span>
                      <ActionBadge action={entry.action} />
                      <span className="flex-1 font-mono text-[10px]" style={{ color: 'var(--text-2)' }}>
                        score {entry.fused_score?.toFixed(3) ?? '—'}
                      </span>
                      <span className="font-mono font-medium"
                        style={{ color: equityChg == null ? 'var(--text-3)' : equityChg >= 0 ? '#16a34a' : '#dc2626' }}>
                        {equityChg != null ? `${equityChg >= 0 ? '+' : ''}${equityChg.toFixed(3)}%` : '—'}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Signal Breakdown */}
        <SignalBreakdown />
      </div>
    </div>
  );
}
