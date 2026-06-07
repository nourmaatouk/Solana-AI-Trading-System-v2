import { useEffect, useState, useMemo } from 'react';
import { fetchBacktest, fetchWalkForward, fetchHistory } from '../services/api';
import {
  ACTIVE_SYMBOL,
  BACKTEST_ACCURACY_GOOD_THRESHOLD,
  BACKTEST_AUC_GOOD_THRESHOLD,
  BACKTEST_MCC_GOOD_THRESHOLD,
  BACKTEST_MODEL_LABEL,
  BACKTEST_PRECISION_GOOD_THRESHOLD,
  DEFAULT_CONFIDENCE_GATE,
  RANDOM_BASELINE_AUC,
} from '../config';
import { formatMarketPair, parseMarketSymbol } from '../utils/symbol';

// ── Types ─────────────────────────────────────────────────────────────────────

interface BacktestData {
  confidence_gate: number;
  n_total_trades: number;
  n_up_signals: number;
  n_dn_signals: number;
  win_rate: number;
  win_rate_up: number;
  win_rate_dn: number;
  avg_gross_pnl: number;
  avg_net_pnl: number;
  compounded_net_return: number;
  max_drawdown: number;
  sharpe: number;
  buy_and_hold_pct: number;
  fold_summaries: FoldSummary[];
}

interface FoldSummary {
  fold: number;
  n_trades: number;
  win_rate: number;
  avg_gross_pnl: number;
  sum_gross_pnl: number;
}

interface WFAggregate {
  overall_accuracy: number;
  overall_auc: number;
  overall_f1: number;
  overall_mcc: number;
  overall_precision: number;
  overall_recall: number;
  random_baseline_acc: number;
  random_baseline_prec: number;
  std_accuracy: number;
  gated_production: { n_trades: number; precision_up: number; precision_dn: number; combined_precision: number; trade_rate: number; n_up: number; n_dn: number };
}

interface WFFold {
  fold: number;
  n_test: number;
  accuracy: number;
  auc: number;
  f1: number;
  mcc: number;
  gated_prod: { n_trades: number; precision_up: number | null; precision_dn: number | null; combined_precision: number | null };
}

interface WalkForwardData {
  n_folds: number;
  confidence_gate: number;
  fold_results: WFFold[];
  aggregate: WFAggregate;
}

interface HistoryEntry {
  run_utc: string;
  action: string;
  equity: number;
  latest_close: number;
  fused_score: number;
}

// ── SVG Sparkline ─────────────────────────────────────────────────────────────

function LineChart({ data, color, height = 120, showBaseline = true }: {
  data: number[];
  color: string;
  height?: number;
  showBaseline?: boolean;
}) {
  const W = 600; const H = height;
  if (data.length < 2) return <div className="h-28 flex items-center justify-center text-xs" style={{ color: 'var(--text-3)' }}>Not enough data</div>;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pad = 8;

  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (W - pad * 2);
    const y = H - pad - ((v - min) / range) * (H - pad * 2);
    return `${x},${y}`;
  });

  const baselineY = H - pad - ((1.0 - min) / range) * (H - pad * 2);
  const areaPath = `M${pts[0]} L${pts.join(' L')} L${pad + (W - pad * 2)},${H - pad} L${pad},${H - pad} Z`;
  const linePath = `M${pts[0]} L${pts.join(' L')}`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height }}>
      <defs>
        <linearGradient id={`grad-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.18" />
          <stop offset="100%" stopColor={color} stopOpacity="0.01" />
        </linearGradient>
      </defs>
      <path d={areaPath} fill={`url(#grad-${color.replace('#', '')})`} />
      <path d={linePath} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
      {showBaseline && (
        <line x1={pad} y1={baselineY} x2={W - pad} y2={baselineY} stroke="#2b3139" strokeWidth="1" strokeDasharray="4 3" />
      )}
      <circle
        cx={parseFloat(pts[pts.length - 1].split(',')[0])}
        cy={parseFloat(pts[pts.length - 1].split(',')[1])}
        r="3" fill={color}
      />
    </svg>
  );
}

// ── Stat card ─────────────────────────────────────────────────────────────────

function Stat({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="rounded-xl p-4" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
      <div className="text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-3)' }}>{label}</div>
      <div className="text-2xl font-bold" style={{ color: color ?? 'var(--text-1)' }}>{value}</div>
      {sub && <div className="text-xs mt-0.5" style={{ color: 'var(--text-3)' }}>{sub}</div>}
    </div>
  );
}

// ── Inline bar ────────────────────────────────────────────────────────────────

function MiniBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-sub)' }}>
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono w-10 text-right" style={{ color: 'var(--text-2)' }}>{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

const card = { background: 'var(--bg-panel)', border: '1px solid var(--border)' } as React.CSSProperties;

// ── Main page ─────────────────────────────────────────────────────────────────

export function BacktestPage() {
  const marketPair = formatMarketPair(ACTIVE_SYMBOL);
  const { base } = parseMarketSymbol(ACTIVE_SYMBOL);
  const [bt, setBt] = useState<BacktestData | null>(null);
  const [wf, setWf] = useState<WalkForwardData | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.allSettled([
      fetchBacktest(),
      fetchWalkForward(),
      fetchHistory(),
    ]).then(([btRes, wfRes, histRes]) => {
      if (btRes.status === 'fulfilled') {
        const d = btRes.value as Record<string, unknown>;
        setBt((d.backtest as BacktestData) ?? null);
      }
      if (wfRes.status === 'fulfilled') setWf(wfRes.value as unknown as WalkForwardData);
      if (histRes.status === 'fulfilled') setHistory(histRes.value as HistoryEntry[]);
      if (btRes.status === 'rejected' && wfRes.status === 'rejected') {
        setError('Could not load backtest or walk-forward data. Run 20_walk_forward_validation.py first.');
      } else if (btRes.status === 'rejected') {
        setError('Could not load backtest metrics. Run 20_walk_forward_validation.py first.');
      } else if (wfRes.status === 'rejected') {
        setError('Could not load walk-forward data. Run 20_walk_forward_validation.py first.');
      }
      setLoading(false);
    });
  }, []);

  const equityCurve = useMemo(() => history.map((h) => h.equity), [history]);
  const agg = wf?.aggregate;

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center" style={{ background: 'var(--bg-page)' }}>
        <div className="text-sm" style={{ color: 'var(--text-3)' }}>Loading backtest data…</div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: 'var(--bg-page)' }}>
      <div className="max-w-4xl mx-auto space-y-5">

        {/* Header */}
        <div>
          <h1 className="text-xl font-bold mb-1" style={{ color: 'var(--text-1)' }}>Backtesting Results</h1>
          <p className="text-sm" style={{ color: 'var(--text-3)' }}>
            Walk-forward validation on historical {marketPair} data · {BACKTEST_MODEL_LABEL} · Confidence gate {((wf?.confidence_gate ?? DEFAULT_CONFIDENCE_GATE) * 100).toFixed(0)}%
          </p>
        </div>

        {error && (
          <div className="rounded-xl px-4 py-3 text-sm" style={{ background: 'var(--red-bg)', border: '1px solid var(--red-bd)', color: 'var(--red)' }}>
            {error}
          </div>
        )}

        {/* ── Backtest hero metrics ─────────────────────────────────────── */}
        {bt && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <Stat label="Sharpe Ratio"  value={bt.sharpe.toFixed(2)}                        sub="Risk-adjusted return"       color="#8B5CF6" />
            <Stat label="Win Rate"      value={`${(bt.win_rate * 100).toFixed(1)}%`}         sub={`${bt.n_total_trades.toLocaleString()} trades`} color="#0ecb81" />
            <Stat label="Max Drawdown"  value={`${(bt.max_drawdown * 100).toFixed(1)}%`}    sub="Worst peak-to-trough"       color="#f6465d" />
            <Stat label="Avg Net PnL"   value={`+${(bt.avg_net_pnl * 100).toFixed(2)}%`}   sub="Per trade after fees"       color="#0ecb81" />
            <Stat label="vs Buy & Hold" value={`+${bt.buy_and_hold_pct.toFixed(0)}%`}       sub={`${base} buy & hold period`}      color="#8B5CF6" />
            <Stat label="Total Trades"  value={bt.n_total_trades.toLocaleString()}           sub={`↑ ${bt.n_up_signals} · ↓ ${bt.n_dn_signals}`} />
          </div>
        )}

        {/* ── Walk-forward aggregate ────────────────────────────────────── */}
        {agg && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <Stat label="Overall AUC"     value={agg.overall_auc.toFixed(4)}
              sub={`Random baseline: ${RANDOM_BASELINE_AUC.toFixed(4)}`}
              color={agg.overall_auc > BACKTEST_AUC_GOOD_THRESHOLD ? '#0ecb81' : '#8B5CF6'} />
            <Stat label="Accuracy"        value={`${(agg.overall_accuracy * 100).toFixed(2)}%`}
              sub={`±${(agg.std_accuracy * 100).toFixed(2)}% across folds`} />
            <Stat label="MCC"             value={agg.overall_mcc.toFixed(4)}
              sub="0 = random · 1 = perfect"
              color={agg.overall_mcc > BACKTEST_MCC_GOOD_THRESHOLD ? '#0ecb81' : '#8B5CF6'} />
            <Stat label="Gated Precision" value={`${(agg.gated_production.combined_precision * 100).toFixed(1)}%`}
              sub={`Baseline: ${(agg.random_baseline_prec * 100).toFixed(1)}%`}
              color="#14B8A6" />
          </div>
        )}

        {/* ── Live paper trading equity curve ──────────────────────────── */}
        {history.length > 1 && (
          <div className="rounded-xl p-5" style={card}>
            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="font-semibold text-sm" style={{ color: 'var(--text-1)' }}>Live Paper Trading Equity</div>
                <div className="text-xs mt-0.5" style={{ color: 'var(--text-3)' }}>
                  {history.length} pipeline runs · baseline 1.0
                  {history.length > 0 && (
                    <span className="ml-2">
                      Current: <span className="font-semibold" style={{ color: equityCurve[equityCurve.length - 1] >= 1 ? '#0ecb81' : '#f6465d' }}>
                        {((equityCurve[equityCurve.length - 1] - 1) * 100).toFixed(3)}%
                      </span>
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-3 text-xs" style={{ color: 'var(--text-2)' }}>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-0.5 inline-block" style={{ background: '#8B5CF6' }} /> Equity
                </span>
                <span className="flex items-center gap-1">
                  <svg width="12" height="2" className="inline-block"><line x1="0" y1="1" x2="12" y2="1" stroke="#2b3139" strokeWidth="2" strokeDasharray="4 3"/></svg>
                  Baseline
                </span>
              </div>
            </div>
            <LineChart data={equityCurve} color="#8B5CF6" height={140} showBaseline />
            <div className="flex justify-between text-[10px] mt-1 px-1" style={{ color: 'var(--text-3)' }}>
              <span>{history[0]?.run_utc ? new Date(history[0].run_utc).toLocaleTimeString() : ''}</span>
              <span>{history[history.length - 1]?.run_utc ? new Date(history[history.length - 1].run_utc).toLocaleTimeString() : ''}</span>
            </div>
          </div>
        )}

        {/* ── Signal action distribution ────────────────────────────────── */}
        {history.length > 0 && (() => {
          const counts = history.reduce((acc, h) => {
            const a = h.action?.toUpperCase() ?? 'HOLD';
            acc[a] = (acc[a] ?? 0) + 1;
            return acc;
          }, {} as Record<string, number>);
          const total = history.length;
          return (
            <div className="rounded-xl p-5" style={card}>
              <div className="font-semibold text-sm mb-4" style={{ color: 'var(--text-1)' }}>Signal Distribution (Live Runs)</div>
              <div className="space-y-3">
                {[['BUY', '#0ecb81'], ['SELL', '#f6465d'], ['HOLD', '#848e9c']].map(([action, color]) => (
                  <div key={action} className="flex items-center gap-3">
                    <div className="w-12 text-xs font-bold" style={{ color }}>{action}</div>
                    <div className="flex-1">
                      <MiniBar value={(counts[action] ?? 0) / total} max={1} color={color} />
                    </div>
                    <div className="text-xs w-8 text-right" style={{ color: 'var(--text-3)' }}>{counts[action] ?? 0}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        })()}

        {/* ── Walk-forward fold table ───────────────────────────────────── */}
        {wf && (
          <div className="rounded-xl p-5" style={card}>
            <div className="font-semibold text-sm mb-4" style={{ color: 'var(--text-1)' }}>
              Walk-Forward Validation — {wf.n_folds} Folds
              <span className="ml-2 text-xs font-normal" style={{ color: 'var(--text-3)' }}>
                (expanding window, {wf.confidence_gate * 100}% gate)
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ color: 'var(--text-3)', borderBottom: '1px solid var(--border)' }}>
                    <th className="text-left py-2 pr-4 font-medium">Fold</th>
                    <th className="text-right py-2 px-3 font-medium">Test bars</th>
                    <th className="text-right py-2 px-3 font-medium">Accuracy</th>
                    <th className="text-right py-2 px-3 font-medium">AUC</th>
                    <th className="text-right py-2 px-3 font-medium">F1</th>
                    <th className="text-right py-2 px-3 font-medium">MCC</th>
                    <th className="text-right py-2 px-3 font-medium">Gated trades</th>
                    <th className="text-right py-2 pl-3 font-medium">Combined prec.</th>
                  </tr>
                </thead>
                <tbody>
                  {wf.fold_results.map((fold) => (
                    <tr key={fold.fold} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td className="py-2 pr-4 font-semibold" style={{ color: 'var(--text-1)' }}>Fold {fold.fold}</td>
                      <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--text-2)' }}>{fold.n_test.toLocaleString()}</td>
                      <td className="py-2 px-3 text-right font-mono" style={{ color: fold.accuracy > BACKTEST_ACCURACY_GOOD_THRESHOLD ? '#0ecb81' : '#8B5CF6' }}>
                        {(fold.accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="py-2 px-3 text-right font-mono" style={{ color: fold.auc > BACKTEST_AUC_GOOD_THRESHOLD ? '#0ecb81' : 'var(--text-2)' }}>
                        {fold.auc.toFixed(4)}
                      </td>
                      <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--text-2)' }}>{fold.f1.toFixed(4)}</td>
                      <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--text-2)' }}>{fold.mcc.toFixed(4)}</td>
                      <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--text-2)' }}>{fold.gated_prod.n_trades.toLocaleString()}</td>
                      <td className="py-2 pl-3 text-right font-mono" style={{ color: (fold.gated_prod.combined_precision ?? 0) > BACKTEST_PRECISION_GOOD_THRESHOLD ? '#0ecb81' : 'var(--text-2)' }}>
                        {fold.gated_prod.combined_precision != null ? `${(fold.gated_prod.combined_precision * 100).toFixed(1)}%` : '—'}
                      </td>
                    </tr>
                  ))}
                  {/* Aggregate row */}
                  <tr style={{ background: 'var(--accent-bg)' }}>
                    <td className="py-2 pr-4 font-semibold" style={{ color: 'var(--accent)' }}>Aggregate</td>
                    <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--accent)' }}>—</td>
                    <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--accent)' }}>{(agg!.overall_accuracy * 100).toFixed(1)}%</td>
                    <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--accent)' }}>{agg!.overall_auc.toFixed(4)}</td>
                    <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--accent)' }}>{agg!.overall_f1.toFixed(4)}</td>
                    <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--accent)' }}>{agg!.overall_mcc.toFixed(4)}</td>
                    <td className="py-2 px-3 text-right font-mono" style={{ color: 'var(--accent)' }}>{agg!.gated_production.n_trades.toLocaleString()}</td>
                    <td className="py-2 pl-3 text-right font-mono" style={{ color: 'var(--accent)' }}>{(agg!.gated_production.combined_precision * 100).toFixed(1)}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── UP vs DOWN signal precision ───────────────────────────────── */}
        {bt && agg && (
          <div className="rounded-xl p-5" style={card}>
            <div className="font-semibold text-sm mb-4" style={{ color: 'var(--text-1)' }}>
              Signal Precision at Gate {((wf?.confidence_gate ?? DEFAULT_CONFIDENCE_GATE) * 100).toFixed(0)}%
            </div>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <div className="flex justify-between text-xs mb-2">
                  <span className="font-semibold" style={{ color: '#0ecb81' }}>↑ UP signals</span>
                  <span style={{ color: 'var(--text-3)' }}>{agg.gated_production.n_up.toLocaleString()} trades</span>
                </div>
                <MiniBar value={agg.gated_production.precision_up} max={1} color="#0ecb81" />
                <div className="text-xs mt-1" style={{ color: 'var(--text-3)' }}>
                  Win rate (backtest): <span className="font-mono" style={{ color: 'var(--text-1)' }}>{(bt.win_rate_up * 100).toFixed(1)}%</span>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-2">
                  <span className="font-semibold" style={{ color: '#f6465d' }}>↓ DOWN signals</span>
                  <span style={{ color: 'var(--text-3)' }}>{agg.gated_production.n_dn.toLocaleString()} trades</span>
                </div>
                <MiniBar value={agg.gated_production.precision_dn} max={1} color="#f6465d" />
                <div className="text-xs mt-1" style={{ color: 'var(--text-3)' }}>
                  Win rate (backtest): <span className="font-mono" style={{ color: 'var(--text-1)' }}>{(bt.win_rate_dn * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
            <div className="mt-4 text-xs rounded-lg px-3 py-2" style={{ background: 'var(--bg-sub)', color: 'var(--text-3)' }}>
              Random baseline precision: <span className="font-mono" style={{ color: 'var(--text-2)' }}>{(agg.random_baseline_prec * 100).toFixed(1)}%</span>
              {' · '}Edge: <span className="font-mono" style={{ color: 'var(--accent)' }}>+{((agg.gated_production.combined_precision - agg.random_baseline_prec) * 100).toFixed(1)}pp</span>
              {' · '}Trade rate: <span className="font-mono" style={{ color: 'var(--text-2)' }}>{(agg.gated_production.trade_rate * 100).toFixed(1)}%</span> of bars
            </div>
          </div>
        )}

        {/* ── Backtest fold summaries ───────────────────────────────────── */}
        {bt && bt.fold_summaries?.length > 0 && (
          <div className="rounded-xl p-5" style={card}>
            <div className="font-semibold text-sm mb-4" style={{ color: 'var(--text-1)' }}>Backtest P&amp;L by Fold</div>
            <div className="space-y-2">
              {bt.fold_summaries.map((f) => (
                <div key={f.fold} className="flex items-center gap-3">
                  <span className="text-xs w-12" style={{ color: 'var(--text-2)' }}>Fold {f.fold}</span>
                  <div className="flex-1">
                    <MiniBar value={f.win_rate} max={1} color="#8B5CF6" />
                  </div>
                  <span className="text-xs font-mono w-20 text-right" style={{ color: 'var(--text-2)' }}>
                    +{(f.sum_gross_pnl * 100).toFixed(1)}% gross
                  </span>
                  <span className="text-xs font-mono w-16 text-right" style={{ color: 'var(--text-3)' }}>
                    {f.n_trades} trades
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
