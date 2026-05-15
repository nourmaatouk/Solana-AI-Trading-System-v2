import { useTradingStore } from '../store';

function Bar({ value, color, label }: { value: number; label: string; color: string }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span>{label}</span>
        <span className="font-mono font-medium" style={{ color }}>{pct.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Gate({ passes }: { passes: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold ${
      passes ? 'bg-green-50 text-green-600 border border-green-200' : 'bg-red-50 text-red-500 border border-red-200'
    }`}>
      {passes ? '✓ Passes 69% gate' : '✗ Below 69% gate'}
    </span>
  );
}

export function WhySignal() {
  const signal = useTradingStore((s) => s.signal);

  if (!signal) {
    return (
      <div className="flex-1 overflow-y-auto p-6" style={{ background: '#f8f9fd' }}>
        <div className="max-w-2xl mx-auto">
          <h1 className="text-xl font-bold text-gray-800 mb-1">Why This Signal?</h1>
          <p className="text-gray-400 text-sm mb-6">Detailed breakdown of the current AI decision.</p>
          <div className="bg-white rounded-xl border border-gray-200 p-8 text-center text-gray-400">
            No signal available yet — run the pipeline first.
          </div>
        </div>
      </div>
    );
  }

  const action = signal.action ?? (signal as unknown as Record<string, unknown>).signal as string ?? 'HOLD';
  const actionColor = action === 'BUY' ? '#0ecb81' : action === 'SELL' ? '#f6465d' : '#f0b90b';
  const probUp = signal.classification_probability_up ?? 0;
  const probDown = signal.classification_probability_down ?? 0;
  const passesGate = probUp >= 0.69 || probDown >= 0.69;
  const spoofing = signal.spoofing_confidence ?? 0;
  const sentimentRelevance = signal.sentiment_relevance ?? 0;

  const ageMin = signal.generated_at_utc
    ? Math.floor((Date.now() - new Date(signal.generated_at_utc).getTime()) / 60000)
    : null;

  const sentimentLabel = signal.reasoning?.sentiment?.toUpperCase().includes('POSITIVE')
    ? 'POSITIVE' : signal.reasoning?.sentiment?.toUpperCase().includes('NEGATIVE')
    ? 'NEGATIVE' : 'NEUTRAL';

  const sentimentColor = sentimentLabel === 'POSITIVE' ? '#0ecb81'
    : sentimentLabel === 'NEGATIVE' ? '#f6465d' : '#f0b90b';

  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: '#f8f9fd' }}>
      <div className="max-w-2xl mx-auto space-y-4">
        <div>
          <h1 className="text-xl font-bold text-gray-800 mb-1">Why This Signal?</h1>
          <p className="text-gray-400 text-sm">Detailed breakdown of the current AI decision.</p>
        </div>

        {/* Decision card */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-xs text-gray-400 uppercase tracking-wider mb-1">Final Decision</div>
              <div className="text-3xl font-bold" style={{ color: actionColor }}>{action}</div>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-400 mb-1">Confidence</div>
              <div className="text-2xl font-bold text-gray-800">{((signal.confidence ?? 0) * 100).toFixed(1)}%</div>
              {ageMin !== null && (
                <div className={`text-xs mt-1 ${ageMin > 20 ? 'text-amber-500' : 'text-gray-400'}`}>
                  {ageMin > 20 ? `⚠ ${ageMin}m ago (stale)` : `${ageMin}m ago`}
                </div>
              )}
            </div>
          </div>
          {signal.reason && (
            <div className="bg-gray-50 rounded-lg p-3 text-sm text-gray-600 leading-relaxed">
              {signal.reason}
            </div>
          )}
        </div>

        {/* XGBoost */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="font-semibold text-gray-800 text-sm">XGBoost Classifier</div>
              <div className="text-xs text-gray-400">43-feature model · 15-min horizon</div>
            </div>
            <Gate passes={passesGate} />
          </div>
          <div className="space-y-3">
            <Bar value={probUp} color="#0ecb81" label="Probability UP" />
            <Bar value={probDown} color="#f6465d" label="Probability DOWN" />
          </div>
          <div className="mt-3 text-xs text-gray-400">
            The 69% confidence gate requires at least one direction to exceed 0.69 for a non-HOLD signal.
            {!passesGate && <span className="text-amber-600 font-medium"> Neither does — model outputs HOLD.</span>}
          </div>
          {signal.reasoning?.classification && (
            <div className="mt-2 text-xs text-gray-500 bg-gray-50 rounded p-2">{signal.reasoning.classification}</div>
          )}
        </div>

        {/* FinBERT */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="font-semibold text-gray-800 text-sm">FinBERT Sentiment</div>
              <div className="text-xs text-gray-400">NLP model on crypto news · weight 25%</div>
            </div>
            <span className="px-2 py-0.5 rounded-full text-xs font-bold border"
              style={{ color: sentimentColor, borderColor: sentimentColor + '40', backgroundColor: sentimentColor + '15' }}>
              {sentimentLabel}
            </span>
          </div>
          <Bar value={sentimentRelevance} color="#6366f1" label="Sentiment relevance score" />
          {signal.reasoning?.sentiment && (
            <div className="mt-2 text-xs text-gray-500 bg-gray-50 rounded p-2">{signal.reasoning.sentiment}</div>
          )}
        </div>

        {/* TEN-GNN */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="font-semibold text-gray-800 text-sm">TEN-GNN Spoofing Detector</div>
              <div className="text-xs text-gray-400">Graph neural network on order book · veto weight</div>
            </div>
            <span className={`px-2 py-0.5 rounded-full text-xs font-bold border ${
              spoofing >= 0.7 ? 'text-red-500 bg-red-50 border-red-200'
              : spoofing >= 0.4 ? 'text-amber-500 bg-amber-50 border-amber-200'
              : 'text-green-600 bg-green-50 border-green-200'
            }`}>
              {spoofing >= 0.7 ? 'SUSPICIOUS' : spoofing >= 0.4 ? 'WARNING' : 'NORMAL'}
            </span>
          </div>
          <Bar value={spoofing} color={spoofing >= 0.7 ? '#f6465d' : spoofing >= 0.4 ? '#f0b90b' : '#0ecb81'} label="Spoofing confidence" />
          <div className="mt-2 text-xs text-gray-400">
            {spoofing >= 0.7
              ? '⚠ Spoofing veto active — BUY signals are blocked regardless of XGBoost output.'
              : spoofing >= 0.4
              ? 'Warning level — BUY confidence reduced but not blocked.'
              : 'Market appears clean. No spoofing penalty applied.'}
          </div>
          {signal.reasoning?.spoofing && (
            <div className="mt-2 text-xs text-gray-500 bg-gray-50 rounded p-2">{signal.reasoning.spoofing}</div>
          )}
        </div>

        {/* Fusion weights */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="font-semibold text-gray-800 text-sm mb-3">Fusion Weights</div>
          <div className="grid grid-cols-4 gap-3 text-center">
            {[['XGBoost', '50%', '#6366f1'], ['FinBERT', '25%', '#8b5cf6'], ['TEN-GNN', '20%', '#f59e0b'], ['GNN', '5%', '#06b6d4']].map(([name, pct, color]) => (
              <div key={name} className="bg-gray-50 rounded-lg p-3">
                <div className="text-base font-bold" style={{ color }}>{pct}</div>
                <div className="text-xs text-gray-400 mt-0.5">{name}</div>
              </div>
            ))}
          </div>
          <div className="mt-3 text-xs text-gray-400">
            Fused score: <span className="font-mono text-gray-700">{(signal.fused_score ?? 0).toFixed(4)}</span>
            {' · '}Risk level: <span className="font-semibold text-gray-700">{signal.risk_level ?? 'N/A'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
