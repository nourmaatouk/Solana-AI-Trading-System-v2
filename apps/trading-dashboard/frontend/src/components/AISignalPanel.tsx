import { useState } from 'react';
import { useAISignal } from '../hooks/useAISignal';
import { triggerSignal } from '../services/api';
import { DEFAULT_CONFIDENCE_GATE, SPOOFING_HIGH, SPOOFING_MEDIUM } from '../config';

function ConfidenceBar({ value, color = 'var(--green)' }: { value: number; color?: string }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
      <div className="h-full rounded-full transition-all duration-300" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  );
}

export function AISignalPanel() {
  const { signal, loading, error, lastUpdated } = useAISignal();
  const [triggering, setTriggering] = useState(false);
  const [triggerMsg, setTriggerMsg] = useState<string | null>(null);

  const handleRefresh = async () => {
    setTriggering(true);
    setTriggerMsg(null);
    try {
      await triggerSignal();
      setTriggerMsg('Pipeline triggered. Signal updating...');
    } catch (err) {
      setTriggerMsg(`Error: ${err instanceof Error ? err.message : 'Unknown'}`);
    } finally {
      setTriggering(false);
    }
  };

  const action = signal?.action ?? signal?.signal ?? null;
  const actionColor = action === 'BUY' ? 'var(--green)' : action === 'SELL' ? 'var(--red)' : 'var(--accent)';
  const actionRaw  = action === 'BUY' ? '#0ecb81' : action === 'SELL' ? '#f6465d' : '#8B5CF6';
  const riskColor  = { LOW: 'var(--green)', MEDIUM: 'var(--accent)', HIGH: 'var(--red)' }[signal?.risk_level ?? 'LOW'] ?? 'var(--text-2)';

  return (
    <div className="flex flex-col h-full" style={{ background: 'var(--bg-panel)', borderLeft: '1px solid var(--border)' }}>
      <div className="px-3 py-2 flex items-center justify-between" style={{ borderBottom: '1px solid var(--border)' }}>
        <span className="text-xs font-semibold" style={{ color: 'var(--text-1)' }}>AI Signal</span>
        <div className="px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)' }}>
          <span className="text-[9px] font-medium" style={{ color: 'var(--accent)' }}>AI Simulation</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {error && (
          <div className="rounded p-2 text-xs" style={{ background: 'var(--red-bg)', border: '1px solid var(--red-bd)', color: 'var(--red)' }}>
            {error}
          </div>
        )}
        {!signal && !loading && (
          <div className="flex items-center justify-center h-20 text-xs" style={{ color: 'var(--text-3)' }}>
            No signal — run pipeline first
          </div>
        )}

        {signal && (
          <>
            <div className="flex items-center justify-between">
              <div className="px-4 py-1.5 rounded text-sm font-bold"
                style={{ backgroundColor: `${actionRaw}20`, color: actionColor, border: `1px solid ${actionRaw}40` }}>
                {action ?? 'HOLD'}
              </div>
              <div className="text-right">
                <div className="text-[9px]" style={{ color: 'var(--text-3)' }}>Risk Level</div>
                <div className="text-xs font-semibold" style={{ color: riskColor }}>{signal.risk_level ?? 'N/A'}</div>
              </div>
            </div>

            <div>
              {action === 'HOLD' || action === null ? (() => {
                const gateProximity = Math.max(signal.classification_probability_up ?? 0, signal.classification_probability_down ?? 0);
                return (
                  <>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px]" style={{ color: 'var(--text-3)' }}>Gate proximity</span>
                      <span className="text-[10px] font-mono" style={{ color: 'var(--text-1)' }}>{(gateProximity * 100).toFixed(1)}%</span>
                    </div>
                    <ConfidenceBar value={gateProximity} color={actionRaw} />
                    <div className="text-[9px] mt-1" style={{ color: 'var(--text-3)' }}>Below {Math.round((signal.confidence_gate ?? DEFAULT_CONFIDENCE_GATE) * 100)}% gate → HOLD</div>
                  </>
                );
              })() : (
                <>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px]" style={{ color: 'var(--text-3)' }}>Confidence</span>
                    <span className="text-[10px] font-mono" style={{ color: 'var(--text-1)' }}>{((signal.confidence ?? 0) * 100).toFixed(1)}%</span>
                  </div>
                  <ConfidenceBar value={signal.confidence ?? 0} color={actionRaw} />
                </>
              )}
            </div>

            <div className="flex items-center justify-between">
              <span className="text-[10px]" style={{ color: 'var(--text-3)' }}>Fused Score</span>
              <span className="text-[10px] font-mono" style={{ color: 'var(--text-1)' }}>{(signal.fused_score ?? 0).toFixed(4)}</span>
            </div>

            <div style={{ borderTop: '1px solid var(--border)' }} />

            <div>
              <div className="text-[10px] mb-1.5 font-medium uppercase tracking-wider" style={{ color: 'var(--text-3)' }}>XGBoost</div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px]" style={{ color: 'var(--green)' }}>UP {((signal.classification_probability_up ?? 0) * 100).toFixed(1)}%</span>
                <span className="text-[10px]" style={{ color: 'var(--red)' }}>DOWN {((signal.classification_probability_down ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <ConfidenceBar value={signal.classification_probability_up ?? 0} color="#0ecb81" />
              <div className="mt-0.5" />
              <ConfidenceBar value={signal.classification_probability_down ?? 0} color="#f6465d" />
              {signal.reasoning?.classification && (
                <div className="text-[9px] mt-1" style={{ color: 'var(--text-3)' }}>{signal.reasoning.classification}</div>
              )}
            </div>

            <div style={{ borderTop: '1px solid var(--border)' }} />

            <div>
              <div className="text-[10px] mb-1.5 font-medium uppercase tracking-wider" style={{ color: 'var(--text-3)' }}>FinBERT Sentiment</div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px]" style={{ color: 'var(--text-2)' }}>Relevance</span>
                <span className="text-[10px] font-mono" style={{ color: 'var(--text-1)' }}>{((signal.sentiment_relevance ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <ConfidenceBar value={signal.sentiment_relevance ?? 0} color="#14B8A6" />
              {signal.reasoning?.sentiment && (
                <div className="text-[9px] mt-1" style={{ color: 'var(--text-3)' }}>{signal.reasoning.sentiment}</div>
              )}
            </div>

            <div style={{ borderTop: '1px solid var(--border)' }} />

            <div>
              <div className="text-[10px] mb-1.5 font-medium uppercase tracking-wider" style={{ color: 'var(--text-3)' }}>TEN-GNN Spoofing</div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px]" style={{ color: 'var(--text-2)' }}>Confidence</span>
                <span className="text-[10px] font-mono" style={{ color: 'var(--text-1)' }}>{((signal.spoofing_confidence ?? 0) * 100).toFixed(2)}%</span>
              </div>
              <ConfidenceBar
                value={signal.spoofing_confidence ?? 0}
                color={(signal.spoofing_confidence ?? 0) >= SPOOFING_HIGH ? '#f6465d' : (signal.spoofing_confidence ?? 0) >= SPOOFING_MEDIUM ? '#8B5CF6' : '#0ecb81'}
              />
              {signal.reasoning?.spoofing && (
                <div className="text-[9px] mt-1" style={{ color: 'var(--text-3)' }}>{signal.reasoning.spoofing}</div>
              )}
            </div>

            <div style={{ borderTop: '1px solid var(--border)' }} />

            {signal.reason && (
              <div>
                <div className="text-[10px] mb-1 font-medium uppercase tracking-wider" style={{ color: 'var(--text-3)' }}>Reasoning</div>
                <div className="text-[10px] leading-relaxed" style={{ color: 'var(--text-2)' }}>{signal.reason}</div>
              </div>
            )}
            {signal.generated_at_utc && (
              <div className="text-[9px]" style={{ color: 'var(--text-3)' }}>
                Generated: {new Date(signal.generated_at_utc).toLocaleString()}
              </div>
            )}
          </>
        )}

        {triggerMsg && (
          <div className="rounded p-2 text-[10px]" style={{ background: 'var(--bg-sub)', border: '1px solid var(--border)', color: 'var(--text-2)' }}>
            {triggerMsg}
          </div>
        )}
      </div>

      <div className="px-3 py-2" style={{ borderTop: '1px solid var(--border)' }}>
        <button
          onClick={handleRefresh}
          disabled={triggering}
          className="w-full py-1.5 rounded text-xs font-medium transition-colors disabled:opacity-50"
          style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)', color: 'var(--accent)' }}
        >
          {triggering ? 'Running pipeline...' : 'Refresh Signal'}
        </button>
        {lastUpdated && (
          <div className="text-[9px] mt-1 text-center" style={{ color: 'var(--text-3)' }}>
            Updated: {lastUpdated.toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
}
