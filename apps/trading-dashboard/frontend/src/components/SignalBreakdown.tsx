import { useTradingStore } from '../store';
import { parseSentimentLabel, signalAgeMinutes, SIGNAL_STALE_MINUTES } from '../utils/signal';
import { DEFAULT_CONFIDENCE_GATE, SPOOFING_HIGH, SPOOFING_MEDIUM } from '../config';

function SentimentPill({ label }: { label: string }) {
  const upper = label?.toUpperCase() ?? '';
  const styles: Record<string, { color: string; bg: string; border: string }> = {
    POSITIVE: { color: 'var(--green)', bg: 'var(--green-bg)', border: 'var(--green-bd)' },
    BULLISH:  { color: 'var(--green)', bg: 'var(--green-bg)', border: 'var(--green-bd)' },
    NEGATIVE: { color: 'var(--red)',   bg: 'var(--red-bg)',   border: 'var(--red-bd)'   },
    BEARISH:  { color: 'var(--red)',   bg: 'var(--red-bg)',   border: 'var(--red-bd)'   },
    NEUTRAL:  { color: 'var(--text-2)', bg: 'var(--bg-sub)', border: 'var(--border)'   },
  };
  const s = styles[upper] ?? styles.NEUTRAL;
  return (
    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border"
      style={{ color: s.color, backgroundColor: s.bg, borderColor: s.border }}>
      {upper || 'UNKNOWN'}
    </span>
  );
}

function HBar({ value, color, label }: { value: number; color: string; label: string }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="flex items-center gap-2">
      <span className="w-14 text-xs text-right shrink-0" style={{ color: 'var(--text-3)' }}>{label}</span>
      <div className="flex-1 relative h-4 rounded-full overflow-hidden" style={{ background: 'var(--bg-sub)' }}>
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="w-10 text-xs font-mono shrink-0" style={{ color: 'var(--text-1)' }}>{pct.toFixed(1)}%</span>
    </div>
  );
}

export function SignalBreakdown() {
  const signal = useTradingStore((s) => s.signal);

  if (!signal) {
    return (
      <div className="rounded-xl p-5" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
        <div className="text-xs font-semibold tracking-widest uppercase mb-3" style={{ color: 'var(--text-3)' }}>Signal Breakdown</div>
        <p className="text-sm" style={{ color: 'var(--text-3)' }}>Waiting for signal data…</p>
      </div>
    );
  }

  const probUp = signal.classification_probability_up ?? 0;
  const probDown = signal.classification_probability_down ?? 0;
  const spoofing = signal.spoofing_confidence ?? 0;
  const gate = signal.confidence_gate ?? DEFAULT_CONFIDENCE_GATE;
  const passesGate = probUp >= gate || probDown >= gate;

  const sentimentLabel = parseSentimentLabel(signal.reasoning?.sentiment ?? '');
  const ageMin = signalAgeMinutes(signal.generated_at_utc);
  const isFresh = ageMin !== null && ageMin < SIGNAL_STALE_MINUTES;
  const isStale = ageMin !== null && ageMin >= SIGNAL_STALE_MINUTES;

  const spoofColor = spoofing >= SPOOFING_HIGH ? '#f6465d' : spoofing >= SPOOFING_MEDIUM ? '#8B5CF6' : '#0ecb81';

  const action = (signal.action ?? signal.signal ?? 'HOLD').toUpperCase();
  const actionStyles: Record<string, { color: string; bg: string; border: string }> = {
    BUY:  { color: 'var(--green)',  bg: 'var(--green-bg)',  border: 'var(--green-bd)'  },
    SELL: { color: 'var(--red)',    bg: 'var(--red-bg)',    border: 'var(--red-bd)'    },
    HOLD: { color: 'var(--accent)', bg: 'var(--accent-bg)', border: 'var(--accent-bd)' },
  };
  const actionStyle = actionStyles[action] ?? actionStyles.HOLD;

  const ppoOwns = signal.decision_source === 'ppo';
  const ppoStale = signal.decision_source === 'fusion_rl_stale';

  return (
    <div className="rounded-xl p-5 space-y-4" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--text-3)' }}>Signal Breakdown</span>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold border"
            style={{ color: actionStyle.color, backgroundColor: actionStyle.bg, borderColor: actionStyle.border }}>
            {action}
          </span>
          {ageMin !== null && (
            <span className="text-xs font-mono px-2 py-0.5 rounded-full"
              style={{
                color: isFresh ? 'var(--green)' : isStale ? 'var(--accent)' : 'var(--text-3)',
                background: isFresh ? 'var(--green-bg)' : isStale ? 'var(--accent-bg)' : 'var(--bg-sub)',
              }}>
              {isFresh ? '● ' : isStale ? '⚠ ' : ''}{ageMin}m ago
            </span>
          )}
        </div>
      </div>

      {/* Decision-maker banner */}
      {(ppoOwns || ppoStale) && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold"
          style={{
            background: ppoOwns ? 'rgba(139,92,246,0.12)' : 'rgba(246,180,53,0.10)',
            border: `1px solid ${ppoOwns ? 'rgba(139,92,246,0.4)' : 'rgba(246,180,53,0.35)'}`,
            color: ppoOwns ? '#a78bfa' : '#f6b435',
          }}>
          <span style={{ fontSize: '1rem', lineHeight: 1 }}>{ppoOwns ? '🤖' : '⚠'}</span>
          {ppoOwns
            ? 'Decided by: PPO RL Agent'
            : 'PPO signal stale — decided by fusion ensemble'}
        </div>
      )}

      {/* XGBoost section */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold" style={{ color: 'var(--text-1)' }}>XGBoost Confidence</span>
          <span className="text-xs" style={{ color: 'var(--text-3)' }}>Gate: {(gate * 100).toFixed(0)}%</span>
        </div>
        <div className="space-y-1.5">
          <HBar value={probUp}   color="#0ecb81" label="UP"   />
          <HBar value={probDown} color="#f6465d" label="DOWN" />
        </div>
        <div className="mt-2 text-xs flex items-center gap-1">
          {passesGate
            ? <span style={{ color: 'var(--green)' }} className="font-medium">✓ Passes confidence gate</span>
            : <span style={{ color: 'var(--accent)' }} className="font-medium">Below {(gate * 100).toFixed(0)}% gate → HOLD</span>
          }
        </div>
      </div>

      {/* FinBERT sentiment */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold" style={{ color: 'var(--text-1)' }}>FinBERT Sentiment</span>
        <SentimentPill label={sentimentLabel} />
      </div>

      {/* TEN-GNN spoofing */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs font-semibold" style={{ color: 'var(--text-1)' }}>TEN-GNN Spoofing</span>
          <span className="text-xs font-mono" style={{ color: 'var(--text-2)' }}>{(spoofing * 100).toFixed(1)}%</span>
        </div>
        <div className="h-2.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-sub)' }}>
          <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.min(100, spoofing * 100)}%`, backgroundColor: spoofColor }} />
        </div>
        <div className="mt-1 text-xs" style={{ color: 'var(--text-3)' }}>
          {spoofing >= SPOOFING_HIGH ? 'High spoofing risk' : spoofing >= SPOOFING_MEDIUM ? 'Moderate spoofing risk' : 'No spoofing detected'}
        </div>
      </div>

      {/* PPO RL Agent */}
      {signal.rl_signal && (
        <div className="rounded-lg p-3 space-y-2"
          style={{
            background: ppoOwns ? 'rgba(139,92,246,0.07)' : 'var(--bg-sub)',
            border: ppoOwns ? '1px solid rgba(139,92,246,0.25)' : '1px solid var(--border)',
          }}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-semibold" style={{ color: ppoOwns ? '#a78bfa' : 'var(--text-1)' }}>
                PPO RL Agent
              </span>
              {ppoOwns && (
                <span className="text-xs font-bold px-1.5 py-0.5 rounded"
                  style={{ background: 'rgba(139,92,246,0.2)', color: '#a78bfa', letterSpacing: '0.05em' }}>
                  DECIDER
                </span>
              )}
            </div>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border"
              style={
                signal.rl_signal === 'BUY'  ? { color: 'var(--green)', backgroundColor: 'var(--green-bg)', borderColor: 'var(--green-bd)' } :
                signal.rl_signal === 'SELL' ? { color: 'var(--red)',   backgroundColor: 'var(--red-bg)',   borderColor: 'var(--red-bd)'   } :
                                              { color: 'var(--text-2)', backgroundColor: 'var(--bg-sub)',  borderColor: 'var(--border)'   }
              }>
              {signal.rl_signal}
            </span>
          </div>
          {signal.rl_confidence !== undefined && (
            <div className="h-2.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-sub)' }}>
              <div className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${Math.min(100, signal.rl_confidence * 100)}%`,
                  backgroundColor: signal.rl_signal === 'BUY' ? '#0ecb81' : signal.rl_signal === 'SELL' ? '#f6465d' : '#8B5CF6',
                }} />
            </div>
          )}
          <div className="text-xs" style={{ color: 'var(--text-3)' }}>
            Confidence: {((signal.rl_confidence ?? 0) * 100).toFixed(1)}%
            {ppoOwns && <span style={{ color: '#a78bfa' }}> · overrides ensemble</span>}
          </div>
        </div>
      )}

    </div>
  );
}
