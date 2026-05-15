import { useEffect, useState } from 'react';
import { useTradingStore } from '../store';
import { fetchStatus, fetchScheduler } from '../services/api';

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

function StatusDot({ ok }: { ok: boolean | null }) {
  if (ok === null) return <span className="w-2 h-2 rounded-full bg-gray-300 flex-shrink-0 inline-block" />;
  return ok
    ? <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse-dot flex-shrink-0 inline-block" />
    : <span className="w-2 h-2 rounded-full bg-red-500 flex-shrink-0 inline-block" />;
}

export function SystemStatus() {
  const connectionStatus = useTradingStore((s) => s.connectionStatus);
  const exchange = useTradingStore((s) => s.exchange);
  const ticker = useTradingStore((s) => s.ticker);

  const [pipeline, setPipeline] = useState<PipelineStatus | null>(null);
  const [scheduler, setScheduler] = useState<SchedulerInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const [p, s] = await Promise.allSettled([
        fetchStatus() as Promise<PipelineStatus>,
        fetchScheduler() as Promise<SchedulerInfo>,
      ]);
      if (p.status === 'fulfilled') setPipeline(p.value);
      if (s.status === 'fulfilled') setScheduler(s.value);
      setLastRefresh(new Date());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void load(); }, []);

  const nextRun = scheduler?.next_run_utc ? new Date(scheduler.next_run_utc) : null;
  const secondsLeft = nextRun ? Math.max(0, Math.floor((nextRun.getTime() - Date.now()) / 1000)) : null;
  const countdown = secondsLeft !== null
    ? `${Math.floor(secondsLeft / 60)}m ${String(secondsLeft % 60).padStart(2, '0')}s`
    : '—';

  const steps: Step[] = Array.isArray(pipeline?.steps) ? pipeline!.steps : [];

  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: '#f8f9fd' }}>
      <div className="max-w-2xl mx-auto space-y-4">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-800 mb-1">System Status</h1>
            <p className="text-gray-400 text-sm">Live health of all system components.</p>
          </div>
          <button
            onClick={() => { void load(); }}
            disabled={loading}
            className="px-3 py-1.5 text-xs rounded-lg border border-gray-200 text-gray-500 hover:bg-gray-100 disabled:opacity-50 transition-colors"
          >
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
        {lastRefresh && (
          <p className="text-xs text-gray-400">Last refreshed: {lastRefresh.toLocaleTimeString()}</p>
        )}

        {/* Market data */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="font-semibold text-gray-800 text-sm mb-3">Market Data</div>
          <div className="space-y-2">
            {[
              { label: 'WebSocket connection', ok: connectionStatus === 'connected', value: connectionStatus },
              { label: 'Exchange', ok: true, value: exchange.toUpperCase() },
              { label: 'Last ticker', ok: !!ticker, value: ticker ? `$${parseFloat(ticker.price).toFixed(2)}` : 'No data' },
            ].map((row) => (
              <div key={row.label} className="flex items-center justify-between py-1.5 border-b border-gray-50 last:border-0">
                <div className="flex items-center gap-2">
                  <StatusDot ok={row.ok} />
                  <span className="text-sm text-gray-600">{row.label}</span>
                </div>
                <span className="text-xs font-mono text-gray-500">{row.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Scheduler */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="font-semibold text-gray-800 text-sm mb-3">Pipeline Scheduler</div>
          {scheduler ? (
            <div className="space-y-2">
              {[
                { label: 'Status', ok: scheduler.last_pipeline_status === 'ok', value: scheduler.last_pipeline_status ?? 'unknown' },
                { label: 'Total runs', ok: null, value: String(scheduler.n_runs ?? '—') },
                { label: 'Next run', ok: null, value: nextRun ? `in ${countdown}` : '—' },
                { label: 'Interval', ok: null, value: scheduler.interval_seconds ? `${scheduler.interval_seconds}s` : '—' },
              ].map((row) => (
                <div key={row.label} className="flex items-center justify-between py-1.5 border-b border-gray-50 last:border-0">
                  <div className="flex items-center gap-2">
                    <StatusDot ok={row.ok} />
                    <span className="text-sm text-gray-600">{row.label}</span>
                  </div>
                  <span className="text-xs font-mono text-gray-500">{String(row.value)}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-gray-400">{loading ? 'Loading…' : 'Could not reach scheduler API.'}</div>
          )}
        </div>

        {/* Pipeline steps */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="font-semibold text-gray-800 text-sm">Last Pipeline Run</div>
            {pipeline?.run_utc && (
              <span className="text-xs text-gray-400">{new Date(pipeline.run_utc).toLocaleString()}</span>
            )}
          </div>
          {steps.length === 0 ? (
            <div className="text-sm text-gray-400">{loading ? 'Loading…' : 'No pipeline run data available. Run the pipeline first.'}</div>
          ) : (
            <div className="space-y-1.5">
              {steps.map((step, i) => {
                const ok = step.status === 'ok' || step.status === 'success' || step.status === 'done';
                return (
                  <div key={i} className={`flex items-center justify-between rounded-lg px-3 py-2 ${ok ? 'bg-green-50' : 'bg-red-50'}`}>
                    <div className="flex items-center gap-2">
                      <StatusDot ok={ok} />
                      <span className="text-xs text-gray-700 font-medium">{step.name}</span>
                      {step.error && <span className="text-xs text-red-500">{step.error}</span>}
                    </div>
                    <div className="flex items-center gap-2">
                      {step.duration_ms !== undefined && (
                        <span className="text-[10px] text-gray-400 font-mono">{step.duration_ms}ms</span>
                      )}
                      <span className={`text-[10px] font-semibold ${ok ? 'text-green-600' : 'text-red-500'}`}>
                        {step.status.toUpperCase()}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          {pipeline?.overall && (
            <div className={`mt-3 text-xs font-semibold text-center py-1.5 rounded-lg ${
              pipeline.overall === 'ok' ? 'bg-green-50 text-green-600' : 'bg-red-50 text-red-500'
            }`}>
              Overall: {pipeline.overall.toUpperCase()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
