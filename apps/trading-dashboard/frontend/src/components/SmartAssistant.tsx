import { useEffect, useState } from 'react';
import { useTradingStore } from '../store';
import { fetchScheduler } from '../services/api';
import { ASSISTANT_SENTIMENT_THRESHOLD, DASHBOARD_POLL_INTERVAL_MS } from '../config';

interface SmartAssistantProps {
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
  if (score > ASSISTANT_SENTIMENT_THRESHOLD) return 'bullish';
  if (score < -ASSISTANT_SENTIMENT_THRESHOLD) return 'bearish';
  return 'neutral';
}

export function SmartAssistant({ fusedScore }: SmartAssistantProps) {
  const sentiment = getMarketSentiment(fusedScore);

  const [pipelineStatus, setPipelineStatus] = useState<string | null>(null);

  // Poll scheduler status on the configured dashboard cadence.
  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchScheduler();
        setPipelineStatus(data.last_pipeline_status as string ?? null);
      } catch { /* keep previous values */ }
    };
    load();
    const interval = setInterval(load, DASHBOARD_POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, []);

  const now = new Date();
  const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' });
  const isDegraded = pipelineStatus !== null && pipelineStatus !== 'ok';

  return (
    <div
      className="rounded-xl p-6 text-white relative overflow-hidden"
      style={{ background: 'linear-gradient(135deg, #6D28D9 0%, #14B8A6 100%)' }}
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
            Markets are looking <strong>{sentiment}</strong>.
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

        </div>
      </div>
    </div>
  );
}
