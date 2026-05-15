import { useTradingStore } from '../store';

const GATE = 0.69;

function SentimentPill({ label }: { label: string }) {
  const upper = label?.toUpperCase() ?? '';
  const colors: Record<string, string> = {
    POSITIVE: 'bg-green-100 text-green-700 border-green-200',
    BULLISH: 'bg-green-100 text-green-700 border-green-200',
    NEGATIVE: 'bg-red-100 text-red-700 border-red-200',
    BEARISH: 'bg-red-100 text-red-700 border-red-200',
    NEUTRAL: 'bg-slate-100 text-slate-600 border-slate-200',
  };
  const cls = colors[upper] ?? 'bg-slate-100 text-slate-600 border-slate-200';
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border ${cls}`}>
      {upper || 'UNKNOWN'}
    </span>
  );
}

function HBar({ value, color, label }: { value: number; color: string; label: string }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="flex items-center gap-2">
      <span className="w-14 text-xs text-slate-500 text-right shrink-0">{label}</span>
      <div className="flex-1 relative h-4 bg-slate-100 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-10 text-xs font-mono text-slate-700 shrink-0">{pct.toFixed(1)}%</span>
    </div>
  );
}

export function SignalBreakdown() {
  const signal = useTradingStore((s) => s.signal);

  if (!signal) {
    return (
      <div className="bg-white rounded-xl border border-slate-100 shadow-sm p-5">
        <div className="text-xs font-semibold tracking-widest uppercase text-slate-400 mb-3">Signal Breakdown</div>
        <p className="text-sm text-slate-400">Waiting for signal data…</p>
      </div>
    );
  }

  const probUp = signal.classification_probability_up ?? 0;
  const probDown = signal.classification_probability_down ?? 0;
  const spoofing = signal.spoofing_confidence ?? 0;
  const passesGate = probUp >= GATE || probDown >= GATE;

  // reasoning.sentiment is a full string like "FinBERT NEGATIVE (-0.135), event=news"
  // Extract just the sentiment word from it
  const rawSentiment = signal.reasoning?.sentiment ?? '';
  const sentimentLabel = rawSentiment.toUpperCase().includes('POSITIVE') ? 'POSITIVE'
    : rawSentiment.toUpperCase().includes('NEGATIVE') ? 'NEGATIVE'
    : rawSentiment.toUpperCase().includes('BULLISH') ? 'BULLISH'
    : rawSentiment.toUpperCase().includes('BEARISH') ? 'BEARISH'
    : 'NEUTRAL';

  const ageMin = signal.generated_at_utc
    ? Math.floor((Date.now() - new Date(signal.generated_at_utc).getTime()) / 60000)
    : null;
  const isFresh = ageMin !== null && ageMin < 20;
  const isStale = ageMin !== null && ageMin >= 20;

  const spoofColor = spoofing >= 0.7 ? 'bg-red-500' : spoofing >= 0.4 ? 'bg-amber-400' : 'bg-green-500';

  const action = (signal.action ?? signal.signal ?? 'HOLD').toUpperCase();
  const actionColors: Record<string, string> = {
    BUY: 'bg-green-100 text-green-700 border-green-200',
    SELL: 'bg-red-100 text-red-700 border-red-200',
    HOLD: 'bg-slate-100 text-slate-600 border-slate-200',
  };
  const actionCls = actionColors[action] ?? actionColors.HOLD;

  return (
    <div className="bg-white rounded-xl border border-slate-100 shadow-sm p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold tracking-widest uppercase text-slate-400">Signal Breakdown</span>
        <div className="flex items-center gap-2">
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold border ${actionCls}`}>
            {action}
          </span>
          {ageMin !== null && (
            <span className={`text-xs font-mono px-2 py-0.5 rounded-full ${isFresh ? 'bg-green-50 text-green-600' : isStale ? 'bg-amber-50 text-amber-600' : 'bg-slate-50 text-slate-500'}`}>
              {isFresh ? '● ' : isStale ? '⚠ ' : ''}{ageMin}m ago
            </span>
          )}
        </div>
      </div>

      {/* XGBoost section */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-600">XGBoost Confidence</span>
          <span className="text-xs text-slate-400">Gate: {(GATE * 100).toFixed(0)}%</span>
        </div>
        <div className="space-y-1.5">
          <HBar value={probUp} color="bg-green-500" label="UP" />
          <HBar value={probDown} color="bg-red-400" label="DOWN" />
        </div>
        {/* Gate line overlay hint */}
        <div className="mt-2 text-xs text-slate-400 flex items-center gap-1">
          {passesGate
            ? <span className="text-green-600 font-medium">✓ Passes confidence gate</span>
            : <span className="text-amber-600 font-medium">Below {(GATE * 100).toFixed(0)}% gate → HOLD</span>
          }
        </div>
      </div>

      {/* FinBERT sentiment */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-600">FinBERT Sentiment</span>
        <SentimentPill label={sentimentLabel} />
      </div>

      {/* TEN-GNN spoofing */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs font-semibold text-slate-600">TEN-GNN Spoofing</span>
          <span className="text-xs font-mono text-slate-500">{(spoofing * 100).toFixed(1)}%</span>
        </div>
        <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
          <div className={`h-full rounded-full ${spoofColor} transition-all duration-500`} style={{ width: `${Math.min(100, spoofing * 100)}%` }} />
        </div>
        <div className="mt-1 text-xs text-slate-400">
          {spoofing >= 0.7 ? '⚠ High spoofing risk' : spoofing >= 0.4 ? '⚠ Moderate spoofing risk' : '✓ No spoofing detected'}
        </div>
      </div>

      {/* Reason */}
      {signal.reason && (
        <div className="pt-1 border-t border-slate-50">
          <p className="text-xs text-slate-500 leading-relaxed">{signal.reason}</p>
        </div>
      )}
    </div>
  );
}
