import { useState } from 'react';
import { useTradingStore } from '../store';
import { executePaperTrade } from '../services/api';
import { signalAgeMinutes, SIGNAL_STALE_MINUTES } from '../utils/signal';
import { DEFAULT_CONFIDENCE_GATE } from '../config';

type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH';

const card = { background: 'var(--bg-panel)', border: '1px solid var(--border)' } as React.CSSProperties;

export function TradePage() {
  const ticker    = useTradingStore((s) => s.ticker);
  const signal    = useTradingStore((s) => s.signal);

  const [riskLevel, setRiskLevel] = useState<RiskLevel>('MEDIUM');
  const [executing, setExecuting] = useState(false);
  const [result, setResult]       = useState<string | null>(null);
  const [error, setError]         = useState<string | null>(null);

  const currentPrice = ticker ? parseFloat(ticker.price) : null;
  const aiAction     = (signal?.action ?? signal?.signal ?? 'HOLD').toUpperCase();
  const aiColor      = aiAction === 'BUY' ? 'var(--green)' : aiAction === 'SELL' ? 'var(--red)' : 'var(--accent)';
  const ageMin       = signalAgeMinutes(signal?.generated_at_utc);
  const isStale      = ageMin !== null && ageMin >= SIGNAL_STALE_MINUTES;

  const handleExecute = async () => {
    setExecuting(true);
    setError(null);
    setResult(null);
    try {
      const res    = await executePaperTrade({ risk_level: riskLevel });
      const status = (res as Record<string, unknown>).status ?? 'triggered';
      setResult(`Pipeline triggered with risk level ${riskLevel} — run ${status}. Dashboard updates in ~20s.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Execution failed');
    } finally {
      setExecuting(false);
    }
  };

  const riskStyle = (r: RiskLevel): React.CSSProperties => {
    const isActive = riskLevel === r;
    if (!isActive) return { color: 'var(--text-2)', borderColor: 'var(--border)', background: 'transparent' };
    if (r === 'LOW')    return { color: 'var(--green)',  background: 'var(--green-bg)',  borderColor: 'var(--green-bd)'  };
    if (r === 'MEDIUM') return { color: 'var(--accent)', background: 'var(--accent-bg)', borderColor: 'var(--accent-bd)' };
    return                     { color: 'var(--red)',    background: 'var(--red-bg)',    borderColor: 'var(--red-bd)'    };
  };

  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: 'var(--bg-page)' }}>
      <div className="max-w-xl mx-auto space-y-4">
        <div>
          <h1 className="text-xl font-bold mb-1" style={{ color: 'var(--text-1)' }}>Paper Trade</h1>
          <p className="text-sm" style={{ color: 'var(--text-3)' }}>
            Runs the AI pipeline and records a simulated trade based on the model's signal. No real funds are used.
          </p>
        </div>

        {/* Info banner */}
        <div className="rounded-xl px-4 py-3 text-xs leading-relaxed"
          style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)', color: 'var(--accent)' }}>
          <strong>How it works:</strong> The AI pipeline (XGBoost + FinBERT + TEN-GNN) generates a BUY, SELL, or HOLD signal.
          If a non-HOLD signal passes the {Math.round((signal?.confidence_gate ?? DEFAULT_CONFIDENCE_GATE) * 100)}% confidence gate, a paper trade is
          recorded with position size scaled by conviction. Your risk level sets a position cap.
        </div>

        {/* AI signal card */}
        <div className="rounded-xl px-5 py-4" style={card}>
          <div className="text-xs uppercase tracking-wider mb-3" style={{ color: 'var(--text-3)' }}>Current AI Signal</div>
          {signal ? (
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold" style={{ color: aiColor }}>{aiAction}</div>
                {ageMin !== null && (
                  <div className="text-xs mt-1" style={{ color: isStale ? 'var(--accent)' : 'var(--text-3)' }}>
                    {isStale ? `⚠ ${ageMin}m ago (stale)` : `${ageMin}m ago`}
                  </div>
                )}
              </div>
              <div className="text-right">
                <div className="text-xs mb-1" style={{ color: 'var(--text-3)' }}>Confidence</div>
                <div className="text-lg font-bold" style={{ color: 'var(--text-1)' }}>
                  {(() => {
                    const act = (signal.action ?? signal.signal ?? 'HOLD').toUpperCase();
                    if (act === 'HOLD') {
                      const prox = Math.max(signal.classification_probability_up ?? 0, signal.classification_probability_down ?? 0);
                      return `${(prox * 100).toFixed(1)}%`;
                    }
                    return `${((signal.confidence ?? 0) * 100).toFixed(1)}%`;
                  })()}
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs mb-1" style={{ color: 'var(--text-3)' }}>Live Price</div>
                <div className="text-lg font-bold font-mono" style={{ color: 'var(--text-1)' }}>
                  {currentPrice ? `$${currentPrice.toFixed(2)}` : '—'}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-sm" style={{ color: 'var(--text-3)' }}>No signal yet — run the pipeline first.</p>
          )}
        </div>

        {/* Risk level selector */}
        <div className="rounded-xl p-5 space-y-4" style={card}>
          <div className="font-semibold text-sm" style={{ color: 'var(--text-1)' }}>Risk Level</div>
          <p className="text-xs -mt-2" style={{ color: 'var(--text-3)' }}>
            Controls the maximum position size fraction. The model scales within this cap based on signal conviction.
          </p>
          <div className="flex gap-3">
            {(['LOW', 'MEDIUM', 'HIGH'] as RiskLevel[]).map((r) => {
              const cap = r === 'LOW' ? '≤ 33%' : r === 'MEDIUM' ? '≤ 66%' : '≤ 100%';
              return (
                <button
                  key={r}
                  onClick={() => setRiskLevel(r)}
                  className="flex-1 py-3 text-xs font-medium rounded-xl border transition-colors flex flex-col items-center gap-1"
                  style={riskStyle(r)}
                >
                  <span className="font-bold">{r}</span>
                  <span className="text-[10px] opacity-70">{cap} equity</span>
                </button>
              );
            })}
          </div>

          <button
            onClick={handleExecute}
            disabled={executing}
            className="w-full py-3 rounded-xl text-sm font-bold transition-colors disabled:opacity-50 text-white"
            style={{ background: 'var(--accent)' }}
          >
            {executing ? 'Running pipeline…' : 'Run Pipeline & Record Trade'}
          </button>
        </div>

        {result && (
          <div className="rounded-xl px-4 py-3 text-xs"
            style={{ background: 'var(--green-bg)', border: '1px solid var(--green-bd)', color: 'var(--green)' }}>
            {result}
          </div>
        )}
        {error && (
          <div className="rounded-xl px-4 py-3 text-xs"
            style={{ background: 'var(--red-bg)', border: '1px solid var(--red-bd)', color: 'var(--red)' }}>
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
