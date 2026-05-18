import { useState } from 'react';
import { triggerSignal, executePaperTrade, stopScheduler, resumeScheduler } from '../services/api';
import { useTradingStore } from '../store';

function parseSymbol(raw: string) {
  if (raw.endsWith('USDT')) return { base: raw.slice(0, -4), quote: 'USDT' };
  if (raw.endsWith('BTC'))  return { base: raw.slice(0, -3),  quote: 'BTC' };
  return { base: raw, quote: '' };
}

type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH';

export function TradingControls() {
  const signal = useTradingStore((s) => s.signal);
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);
  const { base, quote } = parseSymbol(selectedSymbol);

  const [riskLevel, setRiskLevel] = useState<RiskLevel>('MEDIUM');
  const [generating, setGenerating] = useState(false);
  const [executing,  setExecuting]  = useState(false);
  const [stopping,   setStopping]   = useState(false);
  const [botStopped, setBotStopped] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [lastResult, setLastResult]   = useState<string | null>(null);
  const [error, setError]             = useState<string | null>(null);

  const handleGenerateSignal = async () => {
    setGenerating(true); setError(null); setLastResult(null);
    try { await triggerSignal(); setLastResult('Pipeline triggered — signal will update shortly'); }
    catch (err) { setError(err instanceof Error ? err.message : 'Failed to trigger pipeline'); }
    finally { setGenerating(false); }
  };

  const handleExecuteTrade = async () => {
    setShowConfirm(false); setExecuting(true); setError(null); setLastResult(null);
    try {
      const result = await executePaperTrade({ risk_level: riskLevel });
      const status = (result as Record<string, unknown>).status ?? 'triggered';
      setLastResult(`Paper trade submitted — pipeline run ${status}. Check Dashboard in ~20s.`);
    } catch (err) { setError(err instanceof Error ? err.message : 'Execution failed'); }
    finally { setExecuting(false); }
  };

  const handleStopBot = async () => {
    setStopping(true); setError(null);
    try { await stopScheduler(); setBotStopped(true); setLastResult('Bot stopped — pipeline scheduler paused.'); }
    catch { setBotStopped(true); setLastResult('Stopped locally. (Could not reach backend to confirm stop.)'); }
    finally { setStopping(false); }
  };

  const action = signal?.action ?? signal?.signal ?? 'HOLD';
  const actionRaw  = action === 'BUY' ? '#0ecb81' : action === 'SELL' ? '#f6465d' : '#8B5CF6';

  const riskStyle = (r: RiskLevel) =>
    riskLevel === r
      ? r === 'LOW'    ? { background: 'var(--green-bg)',  color: 'var(--green)',  border: '1px solid var(--green-bd)' }
      : r === 'MEDIUM' ? { background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--accent-bd)' }
                       : { background: 'var(--red-bg)',    color: 'var(--red)',    border: '1px solid var(--red-bd)' }
      : { color: 'var(--text-2)', border: '1px solid var(--border)' };

  return (
    <div className="px-4 py-2.5 flex-shrink-0" style={{ background: 'var(--bg-panel)', borderTop: '1px solid var(--border)' }}>
      <div className="flex items-center gap-6 flex-wrap">
        <div className="flex flex-col">
          <span className="text-[9px] uppercase tracking-wider mb-0.5" style={{ color: 'var(--text-3)' }}>Target</span>
          <span className="text-xs font-semibold" style={{ color: 'var(--text-1)' }}>{base}/{quote}</span>
        </div>
        <div className="h-6 w-px" style={{ background: 'var(--border)' }} />
        <div className="flex flex-col">
          <span className="text-[9px] uppercase tracking-wider mb-0.5" style={{ color: 'var(--text-3)' }}>Signal</span>
          <span className="text-xs font-bold" style={{ color: actionRaw }}>{action}</span>
        </div>
        <div className="h-6 w-px" style={{ background: 'var(--border)' }} />
        <div className="flex flex-col">
          <span className="text-[9px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-3)' }}>Risk Level</span>
          <div className="flex gap-1">
            {(['LOW', 'MEDIUM', 'HIGH'] as RiskLevel[]).map((r) => (
              <button key={r} onClick={() => setRiskLevel(r)}
                className="px-2 py-0.5 rounded text-[10px] font-medium transition-colors"
                style={riskStyle(r)}>{r}</button>
            ))}
          </div>
        </div>
        <div className="h-6 w-px" style={{ background: 'var(--border)' }} />
        <div className="flex gap-2 items-center">
          <button onClick={handleGenerateSignal} disabled={generating}
            className="px-3 py-1.5 rounded text-xs font-medium transition-colors disabled:opacity-50"
            style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)', color: 'var(--accent)' }}>
            {generating ? 'Running...' : 'Generate Signal'}
          </button>
          <button onClick={() => setShowConfirm(true)} disabled={executing}
            className="px-3 py-1.5 rounded text-xs font-medium transition-colors disabled:opacity-50"
            style={{ background: 'var(--green-bg)', border: '1px solid var(--green-bd)', color: 'var(--green)' }}>
            {executing ? 'Executing...' : 'Execute (Paper)'}
          </button>
          <button onClick={handleStopBot} disabled={stopping || botStopped}
            className="px-3 py-1.5 rounded text-xs font-medium transition-colors disabled:opacity-50"
            style={{ background: 'var(--red-bg)', border: '1px solid var(--red-bd)', color: 'var(--red)' }}>
            {stopping ? 'Stopping…' : botStopped ? 'Bot Stopped' : 'Stop Bot'}
          </button>
          {botStopped && (
            <button
              onClick={async () => {
                try { await resumeScheduler(); setLastResult('Scheduler restarted — pipeline will run on next cycle.'); }
                catch { setLastResult('Resumed locally. (Could not reach backend to confirm restart.)'); }
                setBotStopped(false);
              }}
              className="px-3 py-1.5 rounded text-xs font-medium transition-colors"
              style={{ background: 'var(--bg-sub)', border: '1px solid var(--border)', color: 'var(--text-2)' }}>
              Resume
            </button>
          )}
        </div>
        <div className="ml-auto">
          <div className="px-2 py-0.5 rounded" style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)' }}>
            <span className="text-[9px] font-medium" style={{ color: 'var(--accent)' }}>No real trades. Simulation only.</span>
          </div>
        </div>
      </div>

      {showConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.6)' }}>
          <div className="rounded-lg p-6 max-w-sm w-full mx-4 shadow-lg"
            style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
            <div className="font-semibold text-sm mb-2" style={{ color: 'var(--text-1)' }}>Confirm Paper Trade</div>
            <div className="text-xs mb-4" style={{ color: 'var(--text-2)' }}>
              Execute a simulated paper trade based on current AI signal ({action})? Risk level: {riskLevel}
            </div>
            <div className="rounded p-2 mb-4" style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)' }}>
              <div className="text-[10px]" style={{ color: 'var(--accent)' }}>This is a paper trade only. No real funds are involved.</div>
            </div>
            <div className="flex gap-2 justify-end">
              <button onClick={() => setShowConfirm(false)}
                className="px-3 py-1.5 rounded text-xs"
                style={{ border: '1px solid var(--border)', color: 'var(--text-2)' }}>Cancel</button>
              <button onClick={handleExecuteTrade}
                className="px-3 py-1.5 rounded text-xs font-medium"
                style={{ background: 'var(--green-bg)', border: '1px solid var(--green-bd)', color: 'var(--green)' }}>
                Confirm Execute
              </button>
            </div>
          </div>
        </div>
      )}

      {(lastResult || error) && (
        <div className="mt-2">
          {lastResult && (
            <div className="text-[10px] px-2 py-1 rounded"
              style={{ color: 'var(--green)', background: 'var(--green-bg)', border: '1px solid var(--green-bd)' }}>
              {lastResult}
            </div>
          )}
          {error && (
            <div className="text-[10px] px-2 py-1 rounded"
              style={{ color: 'var(--red)', background: 'var(--red-bg)', border: '1px solid var(--red-bd)' }}>
              {error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
