import { useState } from 'react';
import { useAISignal } from '../hooks/useAISignal';
import { triggerSignal } from '../services/api';

function ConfidenceBar({ value, color = '#0ecb81' }: { value: number; color?: string }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-300"
        style={{ width: `${pct}%`, backgroundColor: color }}
      />
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
  const actionColor = action === 'BUY' ? '#0ecb81' : action === 'SELL' ? '#f6465d' : '#f0b90b';
  const riskColor = {
    LOW: '#0ecb81',
    MEDIUM: '#f0b90b',
    HIGH: '#f6465d',
  }[signal?.risk_level ?? 'LOW'] ?? '#6b7280';

  return (
    <div className="bg-white flex flex-col h-full border-l border-gray-200">
      {/* Header */}
      <div className="px-3 py-2 border-b border-gray-200 flex items-center justify-between">
        <span className="text-gray-800 text-xs font-semibold">AI Signal</span>
        <div className="flex items-center gap-1.5">
          <div className="px-1.5 py-0.5 rounded bg-yellow-50 border border-yellow-200">
            <span className="text-yellow-600 text-[9px] font-medium">AI Simulation</span>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {error && (
          <div className="bg-red-50 border border-red-200 rounded p-2 text-red-600 text-xs">
            {error}
          </div>
        )}

        {!signal && !loading && (
          <div className="flex items-center justify-center h-20 text-gray-400 text-xs">
            No signal — run pipeline first
          </div>
        )}

        {signal && (
          <>
            {/* Main action */}
            <div className="flex items-center justify-between">
              <div
                className="px-4 py-1.5 rounded text-sm font-bold"
                style={{ backgroundColor: `${actionColor}20`, color: actionColor, border: `1px solid ${actionColor}40` }}
              >
                {action ?? 'HOLD'}
              </div>
              <div className="text-right">
                <div className="text-gray-400 text-[9px]">Risk Level</div>
                <div className="text-xs font-semibold" style={{ color: riskColor }}>
                  {signal.risk_level ?? 'N/A'}
                </div>
              </div>
            </div>

            {/* Confidence */}
            <div>
              {action === 'HOLD' || action === null ? (() => {
                const gateProximity = Math.max(signal.classification_probability_up ?? 0, signal.classification_probability_down ?? 0);
                return (
                  <>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-gray-400 text-[10px]">Gate proximity</span>
                      <span className="text-gray-700 text-[10px] font-mono">{(gateProximity * 100).toFixed(1)}%</span>
                    </div>
                    <ConfidenceBar value={gateProximity} color={actionColor} />
                    <div className="text-gray-400 text-[9px] mt-1">Below 69% gate → HOLD</div>
                  </>
                );
              })() : (
                <>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-gray-400 text-[10px]">Confidence</span>
                    <span className="text-gray-700 text-[10px] font-mono">
                      {((signal.confidence ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <ConfidenceBar value={signal.confidence ?? 0} color={actionColor} />
                </>
              )}
            </div>

            {/* Fused score */}
            <div className="flex items-center justify-between">
              <span className="text-gray-400 text-[10px]">Fused Score</span>
              <span className="text-gray-700 text-[10px] font-mono">
                {(signal.fused_score ?? 0).toFixed(4)}
              </span>
            </div>

            <div className="border-t border-gray-100" />

            {/* XGBoost */}
            <div>
              <div className="text-gray-400 text-[10px] mb-1.5 font-medium uppercase tracking-wider">XGBoost</div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] text-[#0ecb81]">UP {((signal.classification_probability_up ?? 0) * 100).toFixed(1)}%</span>
                <span className="text-[10px] text-[#f6465d]">DOWN {((signal.classification_probability_down ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <ConfidenceBar value={signal.classification_probability_up ?? 0} color="#0ecb81" />
              <div className="mt-0.5" />
              <ConfidenceBar value={signal.classification_probability_down ?? 0} color="#f6465d" />
              {signal.reasoning?.classification && (
                <div className="text-gray-400 text-[9px] mt-1">{signal.reasoning.classification}</div>
              )}
            </div>

            <div className="border-t border-gray-100" />

            {/* FinBERT */}
            <div>
              <div className="text-gray-400 text-[10px] mb-1.5 font-medium uppercase tracking-wider">FinBERT Sentiment</div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-gray-600 text-[10px]">Relevance</span>
                <span className="text-gray-700 text-[10px] font-mono">
                  {((signal.sentiment_relevance ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <ConfidenceBar value={signal.sentiment_relevance ?? 0} color="#6366f1" />
              {signal.reasoning?.sentiment && (
                <div className="text-gray-400 text-[9px] mt-1">{signal.reasoning.sentiment}</div>
              )}
            </div>

            <div className="border-t border-gray-100" />

            {/* TEN-GNN */}
            <div>
              <div className="text-gray-400 text-[10px] mb-1.5 font-medium uppercase tracking-wider">TEN-GNN Spoofing</div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-gray-600 text-[10px]">Confidence</span>
                <span className="text-gray-700 text-[10px] font-mono">
                  {((signal.spoofing_confidence ?? 0) * 100).toFixed(2)}%
                </span>
              </div>
              <ConfidenceBar
                value={signal.spoofing_confidence ?? 0}
                color={(signal.spoofing_confidence ?? 0) > 0.5 ? '#f6465d' : '#0ecb81'}
              />
              {signal.reasoning?.spoofing && (
                <div className="text-gray-400 text-[9px] mt-1">{signal.reasoning.spoofing}</div>
              )}
            </div>

            <div className="border-t border-gray-100" />

            {/* Reason */}
            {signal.reason && (
              <div>
                <div className="text-gray-400 text-[10px] mb-1 font-medium uppercase tracking-wider">Reasoning</div>
                <div className="text-gray-600 text-[10px] leading-relaxed">{signal.reason}</div>
              </div>
            )}

            {/* Timestamp */}
            {signal.generated_at_utc && (
              <div className="text-gray-400 text-[9px]">
                Generated: {new Date(signal.generated_at_utc).toLocaleString()}
              </div>
            )}
          </>
        )}

        {triggerMsg && (
          <div className="bg-gray-50 border border-gray-200 rounded p-2 text-gray-500 text-[10px]">{triggerMsg}</div>
        )}
      </div>

      {/* Footer */}
      <div className="px-3 py-2 border-t border-gray-200">
        <button
          onClick={handleRefresh}
          disabled={triggering}
          className="w-full py-1.5 rounded bg-indigo-50 border border-indigo-200 text-indigo-600 text-xs font-medium hover:bg-indigo-100 transition-colors disabled:opacity-50"
        >
          {triggering ? 'Running pipeline...' : 'Refresh Signal'}
        </button>
        {lastUpdated && (
          <div className="text-gray-400 text-[9px] mt-1 text-center">
            Updated: {lastUpdated.toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
}
