import { useEffect, useState } from 'react';
import { useTradingStore } from '../store';
import { fetchScheduler } from '../services/api';

interface SmartAssistantProps {
  winRate: number | null;
  activeSignals: number;
  loading: boolean;
  fusedScore: number | null;
}

function getGreeting(): string {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning';
  if (h < 17) return 'Good afternoon';
  return 'Good evening';
}

function getMarketSentiment(score: number | null): string {
  if (score === null) return 'neutral';
  if (score > 0.1) return 'bullish';
  if (score < -0.1) return 'bearish';
  return 'neutral';
}

function formatCountdown(seconds: number): string {
  if (seconds <= 0) return 'soon';
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${String(s).padStart(2, '0')}s`;
}

export function SmartAssistant({ winRate, activeSignals, loading, fusedScore }: SmartAssistantProps) {
  const watchlist = useTradingStore((s) => s.watchlist);
  const sentiment = getMarketSentiment(fusedScore);
  const tokenCount = watchlist.length;

  const [nextRunUtc, setNextRunUtc] = useState<string | null>(null);
  const [nRuns, setNRuns] = useState<number | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<string | null>(null);
  const [secondsLeft, setSecondsLeft] = useState<number | null>(null);

  // Poll scheduler status every 30s
  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchScheduler();
        setNextRunUtc(data.next_run_utc as string ?? null);
        setNRuns(data.n_runs as number ?? null);
        setPipelineStatus(data.last_pipeline_status as string ?? null);
      } catch { /* keep previous values */ }
    };
    load();
    const interval = setInterval(load, 30_000);
    return () => clearInterval(interval);
  }, []);

  // Tick countdown every second
  useEffect(() => {
    if (!nextRunUtc) return;
    const tick = () => {
      const diff = Math.max(0, Math.floor((new Date(nextRunUtc).getTime() - Date.now()) / 1000));
      setSecondsLeft(diff);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [nextRunUtc]);

  const now = new Date();
  const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' });
  const isDegraded = pipelineStatus !== null && pipelineStatus !== 'ok';

  return (
    <div
      className="rounded-xl p-6 text-white relative overflow-hidden"
      style={{ background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)' }}
    >
      {/* Decorative circles */}
      <div className="absolute -top-8 -right-8 w-40 h-40 rounded-full bg-white opacity-5 pointer-events-none" />
      <div className="absolute -bottom-10 -right-16 w-56 h-56 rounded-full bg-white opacity-5 pointer-events-none" />

      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-6 relative">
        {/* Left text */}
        <div className="flex-1">
          <div className="text-xs font-semibold tracking-widest uppercase text-purple-200 mb-2">
            Solvanta Smart Assistant
          </div>
          <h2 className="text-xl font-bold mb-1">
            {getGreeting()}, Trader 👋
          </h2>
          <p className="text-sm text-purple-100 leading-relaxed">
            {tokenCount > 0
              ? <>Our system is tracking <strong>{tokenCount}</strong> tokens with <strong>{activeSignals}</strong> active signal{activeSignals !== 1 ? 's' : ''}. Markets are looking <strong>{sentiment}</strong>.</>
              : <>Connecting to market data… Markets are looking <strong>{sentiment}</strong>.</>
            }
          </p>
          {isDegraded && (
            <div className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-400/20 border border-amber-400/30">
              <span className="text-amber-300 text-xs font-semibold">⚠ Pipeline degraded</span>
            </div>
          )}
        </div>

        {/* Right stats */}
        <div className="flex flex-col gap-3 md:items-end">
          {/* LIVE & ACTIVE badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/20 backdrop-blur-sm">
            <span className="w-2 h-2 rounded-full bg-green-300 animate-pulse-dot" />
            <span className="text-xs font-bold tracking-wide">LIVE &amp; ACTIVE</span>
            <span className="text-xs text-purple-200 hidden sm:inline">· {timeStr}</span>
          </div>

          {/* Mini stats */}
          <div className="flex gap-2 flex-wrap justify-end">
            <div className="text-center px-3 py-2 rounded-lg bg-white/15">
              <div className="text-lg font-bold">{tokenCount || '—'}</div>
              <div className="text-xs text-purple-200">Watchlist</div>
            </div>
            <div className="text-center px-3 py-2 rounded-lg bg-white/15">
              <div className="text-lg font-bold">{activeSignals}</div>
              <div className="text-xs text-purple-200">Signals</div>
            </div>
            <div className="text-center px-3 py-2 rounded-lg bg-white/15">
              <div className="text-lg font-bold">
                {loading ? '—' : winRate !== null ? `${(winRate * 100).toFixed(0)}%` : '—'}
              </div>
              <div className="text-xs text-purple-200">Win Rate</div>
            </div>
            <div className="text-center px-3 py-2 rounded-lg bg-white/15">
              <div className="text-base font-bold font-mono">
                {secondsLeft !== null ? formatCountdown(secondsLeft) : '—'}
              </div>
              <div className="text-xs text-purple-200">Next Run</div>
            </div>
            {nRuns !== null && (
              <div className="text-center px-3 py-2 rounded-lg bg-white/15">
                <div className="text-lg font-bold">{nRuns}</div>
                <div className="text-xs text-purple-200">Runs</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
