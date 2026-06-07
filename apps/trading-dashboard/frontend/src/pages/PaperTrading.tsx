import { useEffect, useState, useCallback } from 'react';
import { useTradingStore } from '../store';
import { useAISignal } from '../hooks/useAISignal';
import { SignalBreakdown } from '../components/SignalBreakdown';
import { EquityCurveChart, type EquityPoint } from '../components/EquityCurveChart';
import {
  fetchPaperTrading,
  fetchHistory,
} from '../services/api';
import { DASHBOARD_POLL_INTERVAL_MS } from '../config';

const card = { background: 'var(--bg-panel)', border: '1px solid var(--border)' } as React.CSSProperties;

interface PaperState {
  metrics?: {
    n_trades?: number; win_rate?: number; total_return?: number; max_drawdown?: number;
    sharpe?: number; equity?: number; profit_factor?: number | null; avg_win?: number;
    avg_loss?: number; expectancy?: number; period_days?: number | null;
    equity_series?: EquityPoint[]; benchmark_series?: EquityPoint[];
  };
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

  const [paper, setPaper] = useState<PaperState | null>(null);
  const [history, setHistory] = useState<HistoryEntry[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [p, h] = await Promise.allSettled([fetchPaperTrading(), fetchHistory()]);
    if (p.status === 'fulfilled') {
      setPaper(p.value as PaperState);
      setError(null);
    } else {
      setError('Could not reach the paper-trading API. Is the backend running?');
    }
    if (h.status === 'fulfilled') setHistory((h.value as HistoryEntry[]) ?? []);
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, DASHBOARD_POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [load]);

  // ── Derived metrics (all from real backend data) ──
  const m = paper?.metrics ?? {};
  const equity = num(m.equity);
  const totalReturn = num(m.total_return);
  const winRate = num(m.win_rate);
  const nTrades = num(m.n_trades);
  const sharpe = num(m.sharpe);
  const maxDd = num(m.max_drawdown);
  const profitFactor = num(m.profit_factor);
  const avgWin = num(m.avg_win);
  const avgLoss = num(m.avg_loss);
  const expectancy = num(m.expectancy);
  const periodDays = num(m.period_days);
  const equitySeries = m.equity_series ?? [];
  const benchmarkSeries = m.benchmark_series ?? [];
  const benchEnd = benchmarkSeries.length ? benchmarkSeries[benchmarkSeries.length - 1].value : null;
  const wins = winRate !== null && nTrades !== null ? Math.round(winRate * nTrades) : null;
  const losses = wins !== null && nTrades !== null ? nTrades - wins : null;

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

      {/* Equity curve vs. buy-and-hold SOL */}
      {equitySeries.length > 0 && (
        <div className="rounded-xl p-5" style={card}>
          <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
            <div className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--text-3)' }}>
              Equity curve{periodDays !== null && <span className="normal-case font-normal"> · {periodDays} days</span>}
            </div>
            <div className="flex items-center gap-4 text-xs" style={{ color: 'var(--text-3)' }}>
              <span className="flex items-center gap-1.5"><span style={{ width: 14, height: 2, background: 'var(--green)', display: 'inline-block' }} /> Strategy</span>
              <span className="flex items-center gap-1.5"><span style={{ width: 14, height: 0, borderTop: '2px dashed var(--text-3)', display: 'inline-block' }} /> Hold SOL</span>
            </div>
          </div>
          <EquityCurveChart strategy={equitySeries} benchmark={benchmarkSeries} />
          {totalReturn !== null && benchEnd !== null && (
            <p className="text-xs mt-3" style={{ color: 'var(--text-3)' }}>
              Strategy{' '}
              <strong style={{ color: totalReturn >= 0 ? 'var(--green)' : 'var(--red)' }}>{pct(totalReturn)}</strong>
              {' '}vs. buy-and-hold SOL{' '}
              <strong style={{ color: benchEnd - 1 >= 0 ? 'var(--green)' : 'var(--red)' }}>{pct(benchEnd - 1)}</strong>
              {' '}over the same period — both start at 1.00×.
            </p>
          )}
        </div>
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
                <Metric label="Profit factor" value={profitFactor !== null ? profitFactor.toFixed(2) : '—'}
                  color={profitFactor !== null ? (profitFactor >= 1 ? 'var(--green)' : 'var(--red)') : undefined} />
                <Metric label="Avg win" value={pct(avgWin, 2)} color={avgWin !== null && avgWin > 0 ? 'var(--green)' : undefined} />
                <Metric label="Avg loss" value={pct(avgLoss, 2)} color={avgLoss !== null && avgLoss > 0 ? 'var(--red)' : undefined} />
                <Metric label="Expectancy / trade" value={pct(expectancy, 3)}
                  color={expectancy !== null ? (expectancy >= 0 ? 'var(--green)' : 'var(--red)') : undefined} />
              </div>
            )}
            <p className="text-xs mt-3" style={{ color: 'var(--text-3)' }}>
              Equity is normalised (1.00 = starting balance). Position size scales with model confidence,
              capped by your risk level.
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
