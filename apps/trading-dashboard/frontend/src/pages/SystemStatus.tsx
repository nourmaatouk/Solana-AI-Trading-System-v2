import { useEffect, useState } from 'react';
import { useTradingStore } from '../store';
import { fetchStatus, fetchScheduler, fetchDrift, fetchRetrainStatus, triggerRetrain, fetchAccuracy } from '../services/api';

interface Step {
  name: string;
  status: string;
  duration_ms?: number;
  error?: string;
}

interface PipelineStatus {
  run_utc?: string;
  steps?: Step[];
  overall?: string;
  [key: string]: unknown;
}

interface SchedulerInfo {
  next_run_utc?: string;
  n_runs?: number;
  last_pipeline_status?: string;
  interval_seconds?: number;
  [key: string]: unknown;
}

interface DriftStatus {
  status: string;
  detail: string;
  baseline_accuracy?: number;
  live_accuracy?: number | null;
  live_accuracy_all?: number | null;
  n_signals?: number;
  generated_at_utc?: string;
}

interface RetrainLog {
  status: string;
  started_at_utc?: string;
  finished_at_utc?: string;
  duration_seconds?: number;
  auc_before?: number | null;
  auc_after?: number | null;
  model_swapped?: boolean;
  aborted?: boolean;
}

interface AccuracySummary {
  status?: string;
  n_signals: number;
  n_correct?: number;
  accuracy?: number | null;
  accuracy_30?: number | null;
  buy_accuracy?: number | null;
  sell_accuracy?: number | null;
}

function StatusDot({ ok }: { ok: boolean | null }) {
  if (ok === null) return <span className="w-2 h-2 rounded-full flex-shrink-0 inline-block" style={{ background: 'var(--border)' }} />;
  return ok
    ? <span className="w-2 h-2 rounded-full flex-shrink-0 inline-block animate-pulse-dot" style={{ background: 'var(--green)' }} />
    : <span className="w-2 h-2 rounded-full flex-shrink-0 inline-block" style={{ background: 'var(--red)' }} />;
}

const card = { background: 'var(--bg-panel)', border: '1px solid var(--border)' } as React.CSSProperties;

export function SystemStatus() {
  const connectionStatus = useTradingStore((s) => s.connectionStatus);
  const exchange = useTradingStore((s) => s.exchange);
  const ticker = useTradingStore((s) => s.ticker);

  const [pipeline, setPipeline]   = useState<PipelineStatus | null>(null);
  const [scheduler, setScheduler] = useState<SchedulerInfo | null>(null);
  const [drift, setDrift]         = useState<DriftStatus | null>(null);
  const [retrain, setRetrain]     = useState<RetrainLog | null>(null);
  const [accuracy, setAccuracy]   = useState<AccuracySummary | null>(null);
  const [retraining, setRetraining] = useState(false);
  const [loading, setLoading]     = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const [p, s, d, r, a] = await Promise.allSettled([
        fetchStatus()        as Promise<PipelineStatus>,
        fetchScheduler()     as Promise<SchedulerInfo>,
        fetchDrift()         as unknown as Promise<DriftStatus>,
        fetchRetrainStatus() as unknown as Promise<RetrainLog>,
        fetchAccuracy()      as unknown as Promise<AccuracySummary>,
      ]);
      if (p.status === 'fulfilled') setPipeline(p.value);
      if (s.status === 'fulfilled') setScheduler(s.value);
      if (d.status === 'fulfilled') setDrift(d.value);
      if (r.status === 'fulfilled') setRetrain(r.value);
      if (a.status === 'fulfilled') setAccuracy(a.value);
      setLastRefresh(new Date());
    } finally {
      setLoading(false);
    }
  };

  const handleTriggerRetrain = async () => {
    setRetraining(true);
    try { await triggerRetrain(); } finally { setRetraining(false); }
  };

  useEffect(() => { void load(); }, []);

  const nextRun = scheduler?.next_run_utc ? new Date(scheduler.next_run_utc) : null;
  const secondsLeft = nextRun ? Math.max(0, Math.floor((nextRun.getTime() - Date.now()) / 1000)) : null;
  const countdown = secondsLeft !== null
    ? `${Math.floor(secondsLeft / 60)}m ${String(secondsLeft % 60).padStart(2, '0')}s`
    : '—';

  const steps: Step[] = Array.isArray(pipeline?.steps) ? pipeline!.steps : [];

  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: 'var(--bg-page)' }}>
      <div className="max-w-2xl mx-auto space-y-4">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-xl font-bold mb-1" style={{ color: 'var(--text-1)' }}>System Status</h1>
            <p className="text-sm" style={{ color: 'var(--text-3)' }}>Live health of all system components.</p>
          </div>
          <button
            onClick={() => { void load(); }}
            disabled={loading}
            className="px-3 py-1.5 text-xs rounded-lg border transition-colors disabled:opacity-50"
            style={{ borderColor: 'var(--border)', color: 'var(--text-2)', background: 'transparent' }}
          >
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
        {lastRefresh && (
          <p className="text-xs" style={{ color: 'var(--text-3)' }}>Last refreshed: {lastRefresh.toLocaleTimeString()}</p>
        )}

        {/* Market data */}
        <div className="rounded-xl p-5" style={card}>
          <div className="font-semibold text-sm mb-3" style={{ color: 'var(--text-1)' }}>Market Data</div>
          <div className="space-y-2">
            {[
              { label: 'WebSocket connection', ok: connectionStatus === 'connected', value: connectionStatus },
              { label: 'Exchange',             ok: true,                             value: exchange.toUpperCase() },
              { label: 'Last ticker',          ok: !!ticker,                         value: ticker ? `$${parseFloat(ticker.price).toFixed(2)}` : 'No data' },
            ].map((row) => (
              <div key={row.label} className="flex items-center justify-between py-1.5 last:border-0"
                style={{ borderBottom: '1px solid var(--border)' }}>
                <div className="flex items-center gap-2">
                  <StatusDot ok={row.ok} />
                  <span className="text-sm" style={{ color: 'var(--text-2)' }}>{row.label}</span>
                </div>
                <span className="text-xs font-mono" style={{ color: 'var(--text-3)' }}>{row.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Scheduler */}
        <div className="rounded-xl p-5" style={card}>
          <div className="font-semibold text-sm mb-3" style={{ color: 'var(--text-1)' }}>Pipeline Scheduler</div>
          {scheduler ? (
            <div className="space-y-2">
              {[
                { label: 'Status',     ok: scheduler.last_pipeline_status === 'ok', value: scheduler.last_pipeline_status ?? 'unknown' },
                { label: 'Total runs', ok: null,                                     value: String(scheduler.n_runs ?? '—') },
                { label: 'Next run',   ok: null,                                     value: nextRun ? `in ${countdown}` : '—' },
                { label: 'Interval',   ok: null,                                     value: scheduler.interval_seconds ? `${scheduler.interval_seconds}s` : '—' },
              ].map((row) => (
                <div key={row.label} className="flex items-center justify-between py-1.5 last:border-0"
                  style={{ borderBottom: '1px solid var(--border)' }}>
                  <div className="flex items-center gap-2">
                    <StatusDot ok={row.ok} />
                    <span className="text-sm" style={{ color: 'var(--text-2)' }}>{row.label}</span>
                  </div>
                  <span className="text-xs font-mono" style={{ color: 'var(--text-3)' }}>{String(row.value)}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm" style={{ color: 'var(--text-3)' }}>
              {loading ? 'Loading…' : 'Could not reach scheduler API.'}
            </div>
          )}
        </div>

        {/* Pipeline steps */}
        <div className="rounded-xl p-5" style={card}>
          <div className="flex items-center justify-between mb-3">
            <div className="font-semibold text-sm" style={{ color: 'var(--text-1)' }}>Last Pipeline Run</div>
            {pipeline?.run_utc && (
              <span className="text-xs" style={{ color: 'var(--text-3)' }}>{new Date(pipeline.run_utc).toLocaleString()}</span>
            )}
          </div>
          {steps.length === 0 ? (
            <div className="text-sm" style={{ color: 'var(--text-3)' }}>
              {loading ? 'Loading…' : 'No pipeline run data available. Run the pipeline first.'}
            </div>
          ) : (
            <div className="space-y-1.5">
              {steps.map((step, i) => {
                const ok = step.status === 'ok' || step.status === 'success' || step.status === 'done';
                return (
                  <div key={i} className="flex items-center justify-between rounded-lg px-3 py-2"
                    style={{ background: ok ? 'var(--green-bg)' : 'var(--red-bg)' }}>
                    <div className="flex items-center gap-2">
                      <StatusDot ok={ok} />
                      <span className="text-xs font-medium" style={{ color: 'var(--text-1)' }}>{step.name}</span>
                      {step.error && <span className="text-xs" style={{ color: 'var(--red)' }}>{step.error}</span>}
                    </div>
                    <div className="flex items-center gap-2">
                      {step.duration_ms !== undefined && (
                        <span className="text-[10px] font-mono" style={{ color: 'var(--text-3)' }}>{step.duration_ms}ms</span>
                      )}
                      <span className="text-[10px] font-semibold" style={{ color: ok ? 'var(--green)' : 'var(--red)' }}>
                        {step.status.toUpperCase()}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          {pipeline?.overall && (
            <div className="mt-3 text-xs font-semibold text-center py-1.5 rounded-lg"
              style={pipeline.overall === 'ok'
                ? { background: 'var(--green-bg)', color: 'var(--green)' }
                : { background: 'var(--red-bg)',   color: 'var(--red)'   }
              }>
              Overall: {pipeline.overall.toUpperCase()}
            </div>
          )}
        </div>
        {/* Live accuracy */}
        <div className="rounded-xl p-5" style={card}>
          <div className="font-semibold text-sm mb-3" style={{ color: 'var(--text-1)' }}>Live Signal Accuracy</div>
          {accuracy && accuracy.n_signals > 0 ? (
            <div className="space-y-2">
              {[
                { label: 'Settled signals',  value: String(accuracy.n_signals) },
                { label: 'Overall accuracy', value: accuracy.accuracy != null ? `${(accuracy.accuracy * 100).toFixed(1)}%` : '—' },
                { label: 'Last-30 accuracy', value: accuracy.accuracy_30 != null ? `${(accuracy.accuracy_30 * 100).toFixed(1)}%` : '—' },
                { label: 'BUY accuracy',     value: accuracy.buy_accuracy != null ? `${(accuracy.buy_accuracy * 100).toFixed(1)}%` : '—' },
                { label: 'SELL accuracy',    value: accuracy.sell_accuracy != null ? `${(accuracy.sell_accuracy * 100).toFixed(1)}%` : '—' },
              ].map((row) => (
                <div key={row.label} className="flex items-center justify-between py-1.5 last:border-0"
                  style={{ borderBottom: '1px solid var(--border)' }}>
                  <span className="text-sm" style={{ color: 'var(--text-2)' }}>{row.label}</span>
                  <span className="text-xs font-mono" style={{ color: 'var(--text-1)' }}>{row.value}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm" style={{ color: 'var(--text-3)' }}>
              {loading ? 'Loading…' : 'No settled signals yet — tracker needs ≥4 pipeline runs after each BUY/SELL.'}
            </div>
          )}
        </div>

        {/* Model drift */}
        <div className="rounded-xl p-5" style={card}>
          <div className="flex items-center justify-between mb-3">
            <div className="font-semibold text-sm" style={{ color: 'var(--text-1)' }}>Model Drift</div>
            {drift && (
              <span className="px-2 py-0.5 rounded-full text-xs font-bold border"
                style={
                  drift.status === 'OK'
                    ? { color: 'var(--green)',  background: 'var(--green-bg)',  borderColor: 'var(--green-bd)'  }
                  : drift.status === 'WARNING'
                    ? { color: 'var(--accent)', background: 'var(--accent-bg)', borderColor: 'var(--accent-bd)' }
                  : drift.status === 'DRIFT_DETECTED'
                    ? { color: 'var(--red)',    background: 'var(--red-bg)',    borderColor: 'var(--red-bd)'    }
                    : { color: 'var(--text-3)', background: 'var(--bg-sub)',    borderColor: 'var(--border)'    }
                }>
                {drift.status}
              </span>
            )}
          </div>
          {drift ? (
            <div className="space-y-2">
              <div className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>{drift.detail}</div>
              <div className="flex gap-4 pt-1">
                {drift.baseline_accuracy != null && (
                  <div>
                    <div className="text-[10px]" style={{ color: 'var(--text-3)' }}>Baseline</div>
                    <div className="text-sm font-mono font-semibold" style={{ color: 'var(--text-1)' }}>{(drift.baseline_accuracy * 100).toFixed(1)}%</div>
                  </div>
                )}
                {drift.live_accuracy != null && (
                  <div>
                    <div className="text-[10px]" style={{ color: 'var(--text-3)' }}>Live (last-30)</div>
                    <div className="text-sm font-mono font-semibold" style={{ color: drift.status === 'DRIFT_DETECTED' ? 'var(--red)' : drift.status === 'WARNING' ? 'var(--accent)' : 'var(--green)' }}>
                      {(drift.live_accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                )}
                {drift.n_signals != null && (
                  <div>
                    <div className="text-[10px]" style={{ color: 'var(--text-3)' }}>Signals</div>
                    <div className="text-sm font-mono font-semibold" style={{ color: 'var(--text-1)' }}>{drift.n_signals}</div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-sm" style={{ color: 'var(--text-3)' }}>{loading ? 'Loading…' : 'No drift data yet.'}</div>
          )}
        </div>

        {/* Auto-retrain */}
        <div className="rounded-xl p-5" style={card}>
          <div className="flex items-center justify-between mb-3">
            <div className="font-semibold text-sm" style={{ color: 'var(--text-1)' }}>Auto-Retrain</div>
            <button
              onClick={handleTriggerRetrain}
              disabled={retraining}
              className="px-3 py-1 rounded text-xs font-medium transition-colors disabled:opacity-50"
              style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)', color: 'var(--accent)' }}
            >
              {retraining ? 'Starting…' : 'Run Now'}
            </button>
          </div>
          {retrain && retrain.status !== 'never_run' ? (
            <div className="space-y-2">
              {[
                { label: 'Last run',     value: retrain.finished_at_utc ? new Date(retrain.finished_at_utc).toLocaleString() : '—' },
                { label: 'Duration',     value: retrain.duration_seconds != null ? `${retrain.duration_seconds.toFixed(0)}s` : '—' },
                { label: 'AUC before',   value: retrain.auc_before != null ? retrain.auc_before.toFixed(4) : '—' },
                { label: 'AUC after',    value: retrain.auc_after  != null ? retrain.auc_after.toFixed(4)  : '—' },
                { label: 'Model swapped', value: retrain.model_swapped ? 'Yes — new model live' : 'No (no improvement)' },
              ].map((row) => (
                <div key={row.label} className="flex items-center justify-between py-1.5 last:border-0"
                  style={{ borderBottom: '1px solid var(--border)' }}>
                  <span className="text-sm" style={{ color: 'var(--text-2)' }}>{row.label}</span>
                  <span className="text-xs font-mono" style={{ color: row.label === 'Model swapped' && retrain.model_swapped ? 'var(--green)' : 'var(--text-1)' }}>{row.value}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs" style={{ color: 'var(--text-3)' }}>
              {loading ? 'Loading…' : 'No retrain log yet. Scheduled automatically every Sunday at 02:00 UTC, or click "Run Now".'}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
