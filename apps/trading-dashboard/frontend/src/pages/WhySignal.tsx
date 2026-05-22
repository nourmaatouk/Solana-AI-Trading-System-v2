import { useTradingStore } from '../store';
import { parseSentimentLabel, signalAgeMinutes, SIGNAL_STALE_MINUTES } from '../utils/signal';
import { DEFAULT_CONFIDENCE_GATE, GUARDIAN_SPOOFING_THRESHOLD, SPOOFING_HIGH, SPOOFING_MEDIUM } from '../config';

function Bar({ value, color, label }: { value: number; label: string; color: string }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div>
      <div className="flex justify-between text-xs mb-1" style={{ color: 'var(--text-2)' }}>
        <span>{label}</span>
        <span className="font-mono font-medium" style={{ color }}>{pct.toFixed(1)}%</span>
      </div>
      <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-sub)' }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Gate({ passes, gatePct }: { passes: boolean; gatePct: number }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold border"
      style={passes
        ? { color: 'var(--green)', background: 'var(--green-bg)', borderColor: 'var(--green-bd)' }
        : { color: 'var(--red)',   background: 'var(--red-bg)',   borderColor: 'var(--red-bd)'   }
      }>
      {passes ? `✓ Passes ${gatePct}% gate` : `✗ Below ${gatePct}% gate`}
    </span>
  );
}

const card = { background: 'var(--bg-panel)', border: '1px solid var(--border)' } as React.CSSProperties;

export function WhySignal() {
  const signal = useTradingStore((s) => s.signal);

  if (!signal) {
    return (
      <div className="flex-1 overflow-y-auto p-6" style={{ background: 'var(--bg-page)' }}>
        <div className="max-w-2xl mx-auto">
          <h1 className="text-xl font-bold mb-1" style={{ color: 'var(--text-1)' }}>Why This Signal?</h1>
          <p className="text-sm mb-6" style={{ color: 'var(--text-3)' }}>Detailed breakdown of the current AI decision.</p>
          <div className="rounded-xl p-8 text-center" style={{ ...card, color: 'var(--text-3)' }}>
            No signal available yet — run the pipeline first.
          </div>
        </div>
      </div>
    );
  }

  const rawAction = signal.action ?? (signal as Record<string, unknown>)['signal'];
  const action = typeof rawAction === 'string' ? rawAction : 'HOLD';
  const actionColor = action === 'BUY' ? '#0ecb81' : action === 'SELL' ? '#f6465d' : '#8B5CF6';
  const probUp = signal.classification_probability_up ?? 0;
  const probDown = signal.classification_probability_down ?? 0;
  const gate = signal.confidence_gate ?? DEFAULT_CONFIDENCE_GATE;
  const gatePct = Math.round(gate * 100);
  const passesGate = probUp >= gate || probDown >= gate;
  const spoofing = signal.spoofing_confidence ?? 0;
  const sentimentRelevance = signal.sentiment_relevance ?? 0;

  const ageMin = signalAgeMinutes(signal.generated_at_utc);
  const sentimentLabel = parseSentimentLabel(signal.reasoning?.sentiment ?? '');
  const sentimentColor = sentimentLabel === 'POSITIVE' ? '#0ecb81' : sentimentLabel === 'NEGATIVE' ? '#f6465d' : '#8B5CF6';

  const spoofingBadgeStyle = spoofing >= SPOOFING_HIGH
    ? { color: 'var(--red)',    background: 'var(--red-bg)',    borderColor: 'var(--red-bd)'    }
    : spoofing >= SPOOFING_MEDIUM
    ? { color: 'var(--accent)', background: 'var(--accent-bg)', borderColor: 'var(--accent-bd)' }
    : { color: 'var(--green)',  background: 'var(--green-bg)',  borderColor: 'var(--green-bd)'  };

  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: 'var(--bg-page)' }}>
      <div className="max-w-2xl mx-auto space-y-4">
        <div>
          <h1 className="text-xl font-bold mb-1" style={{ color: 'var(--text-1)' }}>Why This Signal?</h1>
          <p className="text-sm" style={{ color: 'var(--text-3)' }}>Detailed breakdown of the current AI decision.</p>
        </div>

        {/* Decision card */}
        <div className="rounded-xl p-5" style={card}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-3)' }}>Final Decision</div>
              <div className="text-3xl font-bold" style={{ color: actionColor }}>{action}</div>
            </div>
            <div className="text-right">
              <div className="text-xs mb-1" style={{ color: 'var(--text-3)' }}>Confidence</div>
              <div className="text-2xl font-bold" style={{ color: 'var(--text-1)' }}>{((signal.confidence ?? 0) * 100).toFixed(1)}%</div>
              {ageMin !== null && (
                <div className="text-xs mt-1" style={{ color: (ageMin ?? 0) > SIGNAL_STALE_MINUTES ? 'var(--accent)' : 'var(--text-3)' }}>
                  {(ageMin ?? 0) > SIGNAL_STALE_MINUTES ? `⚠ ${ageMin}m ago (stale)` : `${ageMin}m ago`}
                </div>
              )}
            </div>
          </div>
          {signal.reason && (
            <div className="rounded-lg p-3 text-sm leading-relaxed" style={{ background: 'var(--bg-sub)', color: 'var(--text-2)' }}>
              {signal.reason}
            </div>
          )}
        </div>

        {/* XGBoost */}
        <div className="rounded-xl p-5" style={card}>
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="font-semibold text-sm" style={{ color: 'var(--text-1)' }}>XGBoost Classifier</div>
              <div className="text-xs" style={{ color: 'var(--text-3)' }}>Configured classifier horizon</div>
            </div>
            <Gate passes={passesGate} gatePct={gatePct} />
          </div>
          <div className="space-y-3">
            <Bar value={probUp}   color="#0ecb81" label="Probability UP"   />
            <Bar value={probDown} color="#f6465d" label="Probability DOWN" />
          </div>
          <div className="mt-3 text-xs" style={{ color: 'var(--text-3)' }}>
            The {gatePct}% confidence gate requires at least one direction to exceed {gate.toFixed(2)} for a non-HOLD signal.
            {!passesGate && <span className="font-medium" style={{ color: 'var(--accent)' }}> Neither does — model outputs HOLD.</span>}
          </div>
          {signal.reasoning?.classification && (
            <div className="mt-2 text-xs rounded p-2" style={{ background: 'var(--bg-sub)', color: 'var(--text-2)' }}>
              {signal.reasoning.classification}
            </div>
          )}
        </div>

        {/* FinBERT */}
        <div className="rounded-xl p-5" style={card}>
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="font-semibold text-sm" style={{ color: 'var(--text-1)' }}>FinBERT Sentiment</div>
              <div className="text-xs" style={{ color: 'var(--text-3)' }}>NLP model on crypto news · weight 25%</div>
            </div>
            <span className="px-2 py-0.5 rounded-full text-xs font-bold border"
              style={{ color: sentimentColor, borderColor: sentimentColor + '40', backgroundColor: sentimentColor + '15' }}>
              {sentimentLabel}
            </span>
          </div>
          <Bar value={sentimentRelevance} color="#14B8A6" label="Sentiment relevance score" />
          {signal.reasoning?.sentiment && (
            <div className="mt-2 text-xs rounded p-2" style={{ background: 'var(--bg-sub)', color: 'var(--text-2)' }}>
              {signal.reasoning.sentiment}
            </div>
          )}
        </div>

        {/* TEN-GNN */}
        <div className="rounded-xl p-5" style={card}>
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="font-semibold text-sm" style={{ color: 'var(--text-1)' }}>TEN-GNN Spoofing Detector</div>
              <div className="text-xs" style={{ color: 'var(--text-3)' }}>Graph neural network on order book · veto weight</div>
            </div>
            <span className="px-2 py-0.5 rounded-full text-xs font-bold border" style={spoofingBadgeStyle}>
              {spoofing >= SPOOFING_HIGH ? 'SUSPICIOUS' : spoofing >= SPOOFING_MEDIUM ? 'WARNING' : 'NORMAL'}
            </span>
          </div>
          <Bar value={spoofing} color={spoofing >= SPOOFING_HIGH ? '#f6465d' : spoofing >= SPOOFING_MEDIUM ? '#8B5CF6' : '#0ecb81'} label="Spoofing confidence" />
          <div className="mt-2 text-xs" style={{ color: 'var(--text-3)' }}>
            {spoofing >= GUARDIAN_SPOOFING_THRESHOLD
              ? 'Guardian spoofing block active.'
              : spoofing >= SPOOFING_MEDIUM
              ? 'Warning level — BUY confidence reduced but not blocked.'
              : 'Market appears clean. No spoofing penalty applied.'}
          </div>
          {signal.reasoning?.spoofing && (
            <div className="mt-2 text-xs rounded p-2" style={{ background: 'var(--bg-sub)', color: 'var(--text-2)' }}>
              {signal.reasoning.spoofing}
            </div>
          )}
        </div>

        {/* Fusion weights */}
        <div className="rounded-xl p-5" style={card}>
          <div className="font-semibold text-sm mb-3" style={{ color: 'var(--text-1)' }}>Fusion Weights</div>
          <div className="grid grid-cols-4 gap-3 text-center">
            {[['XGBoost', '50%', '#8B5CF6'], ['FinBERT', '25%', '#14B8A6'], ['TEN-GNN', '20%', '#f6465d'], ['GNN', '5%', '#0ecb81']].map(([name, pct, color]) => (
              <div key={name} className="rounded-lg p-3" style={{ background: 'var(--bg-sub)' }}>
                <div className="text-base font-bold" style={{ color }}>{pct}</div>
                <div className="text-xs mt-0.5" style={{ color: 'var(--text-3)' }}>{name}</div>
              </div>
            ))}
          </div>
          <div className="mt-3 text-xs" style={{ color: 'var(--text-3)' }}>
            Fused score: <span className="font-mono" style={{ color: 'var(--text-1)' }}>{(signal.fused_score ?? 0).toFixed(4)}</span>
            {' · '}Risk level: <span className="font-semibold" style={{ color: 'var(--text-1)' }}>{signal.risk_level ?? 'N/A'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
