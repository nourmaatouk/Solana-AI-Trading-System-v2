import { useState } from 'react';
import { triggerSignal, executePaperTrade, stopScheduler, resumeScheduler } from '../services/api';
import { useTradingStore } from '../store';

function parseSymbol(raw: string) {
  if (raw.endsWith('USDT')) return { base: raw.slice(0, -4), quote: 'USDT' };
  if (raw.endsWith('BTC')) return { base: raw.slice(0, -3), quote: 'BTC' };
  return { base: raw, quote: '' };
}

type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH';
type TradingMode = 'Manual' | 'Semi-Auto' | 'Auto';

export function TradingControls() {
  const signal = useTradingStore((s) => s.signal);
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);
  const { base, quote } = parseSymbol(selectedSymbol);

  const [riskLevel, setRiskLevel] = useState<RiskLevel>('MEDIUM');
  const [tradingMode, setTradingMode] = useState<TradingMode>('Manual');
  const [generating, setGenerating] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [botStopped, setBotStopped] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [lastResult, setLastResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateSignal = async () => {
    setGenerating(true);
    setError(null);
    setLastResult(null);
    try {
      await triggerSignal();
      setLastResult('Pipeline triggered — signal will update shortly');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to trigger pipeline');
    } finally {
      setGenerating(false);
    }
  };

  const handleExecuteTrade = async () => {
    setShowConfirm(false);
    setExecuting(true);
    setError(null);
    setLastResult(null);
    try {
      const result = await executePaperTrade({ risk_level: riskLevel, mode: tradingMode });
      const status = (result as Record<string, unknown>).status ?? 'triggered';
      setLastResult(`Paper trade submitted — pipeline run ${status}. Check Dashboard in ~20s.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Execution failed');
    } finally {
      setExecuting(false);
    }
  };

  const handleStopBot = async () => {
    setStopping(true);
    setError(null);
    try {
      await stopScheduler();
      setTradingMode('Manual');
      setBotStopped(true);
      setLastResult('Bot stopped — switched to Manual mode. Pipeline scheduler paused.');
    } catch {
      setTradingMode('Manual');
      setBotStopped(true);
      setLastResult('Switched to Manual mode. (Could not reach backend to confirm stop.)');
    } finally {
      setStopping(false);
    }
  };

  const action = signal?.action ?? signal?.signal ?? 'HOLD';
  const actionColor = action === 'BUY' ? '#0ecb81' : action === 'SELL' ? '#f6465d' : '#f0b90b';

  return (
    <div className="bg-white border-t border-gray-200 px-4 py-2.5">
      <div className="flex items-center gap-6 flex-wrap">
        {/* Target pair */}
        <div className="flex flex-col">
          <span className="text-gray-400 text-[9px] uppercase tracking-wider mb-0.5">Target</span>
          <span className="text-gray-800 text-xs font-semibold">{base}/{quote}</span>
        </div>

        <div className="h-6 w-px bg-gray-200" />

        {/* Current signal */}
        <div className="flex flex-col">
          <span className="text-gray-400 text-[9px] uppercase tracking-wider mb-0.5">Signal</span>
          <span className="text-xs font-bold" style={{ color: actionColor }}>{action}</span>
        </div>

        <div className="h-6 w-px bg-gray-200" />

        {/* Risk level */}
        <div className="flex flex-col">
          <span className="text-gray-400 text-[9px] uppercase tracking-wider mb-1">Risk Level</span>
          <div className="flex gap-1">
            {(['LOW', 'MEDIUM', 'HIGH'] as RiskLevel[]).map((r) => (
              <button
                key={r}
                onClick={() => setRiskLevel(r)}
                className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                  riskLevel === r
                    ? r === 'LOW'
                      ? 'bg-green-50 text-green-600 border border-green-200'
                      : r === 'MEDIUM'
                        ? 'bg-yellow-50 text-yellow-600 border border-yellow-200'
                        : 'bg-red-50 text-red-500 border border-red-200'
                    : 'text-gray-500 border border-gray-200 hover:bg-gray-50'
                }`}
              >
                {r}
              </button>
            ))}
          </div>
        </div>

        <div className="h-6 w-px bg-gray-200" />

        {/* Trading mode */}
        <div className="flex flex-col">
          <span className="text-gray-400 text-[9px] uppercase tracking-wider mb-1">Mode</span>
          <div className="flex gap-1">
            {(['Manual', 'Semi-Auto', 'Auto'] as TradingMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setTradingMode(m)}
                className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                  tradingMode === m
                    ? 'bg-indigo-50 text-indigo-600 border border-indigo-200'
                    : 'text-gray-500 border border-gray-200 hover:bg-gray-50'
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div className="h-6 w-px bg-gray-200" />

        {/* Action buttons */}
        <div className="flex gap-2 items-center">
          <button
            onClick={handleGenerateSignal}
            disabled={generating}
            className="px-3 py-1.5 rounded bg-indigo-50 border border-indigo-200 text-indigo-600 text-xs font-medium hover:bg-indigo-100 transition-colors disabled:opacity-50"
          >
            {generating ? 'Running...' : 'Generate Signal'}
          </button>

          <button
            onClick={() => setShowConfirm(true)}
            disabled={executing}
            className="px-3 py-1.5 rounded bg-green-50 border border-green-200 text-green-600 text-xs font-medium hover:bg-green-100 transition-colors disabled:opacity-50"
          >
            {executing ? 'Executing...' : 'Execute (Paper)'}
          </button>

          <button
            onClick={handleStopBot}
            disabled={stopping || botStopped}
            className="px-3 py-1.5 rounded bg-red-50 border border-red-200 text-red-500 text-xs font-medium hover:bg-red-100 transition-colors disabled:opacity-50"
          >
            {stopping ? 'Stopping…' : botStopped ? 'Bot Stopped' : 'Stop Bot'}
          </button>
          {botStopped && (
            <button
              onClick={async () => {
                try {
                  await resumeScheduler();
                  setLastResult('Scheduler restarted — pipeline will run on next cycle.');
                } catch {
                  setLastResult('Resumed locally. (Could not reach backend to confirm restart.)');
                }
                setBotStopped(false);
              }}
              className="px-3 py-1.5 rounded bg-gray-50 border border-gray-200 text-gray-500 text-xs font-medium hover:bg-gray-100 transition-colors"
            >
              Resume
            </button>
          )}
        </div>

        {/* Disclaimer */}
        <div className="ml-auto">
          <div className="px-2 py-0.5 rounded bg-yellow-50 border border-yellow-200">
            <span className="text-yellow-600 text-[9px] font-medium">No real trades. Simulation only.</span>
          </div>
        </div>
      </div>

      {/* Confirm dialog */}
      {showConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
          <div className="bg-white border border-gray-200 rounded-lg p-6 max-w-sm w-full mx-4 shadow-lg">
            <div className="text-gray-800 font-semibold text-sm mb-2">Confirm Paper Trade</div>
            <div className="text-gray-500 text-xs mb-4">
              Execute a simulated paper trade based on current AI signal ({action})?
              Risk: {riskLevel} | Mode: {tradingMode}
            </div>
            <div className="bg-yellow-50 border border-yellow-200 rounded p-2 mb-4">
              <div className="text-yellow-600 text-[10px]">This is a paper trade only. No real funds are involved.</div>
            </div>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setShowConfirm(false)}
                className="px-3 py-1.5 rounded border border-gray-200 text-gray-500 text-xs hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleExecuteTrade}
                className="px-3 py-1.5 rounded bg-green-50 border border-green-200 text-green-600 text-xs font-medium hover:bg-green-100"
              >
                Confirm Execute
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Status messages */}
      {(lastResult || error) && (
        <div className="mt-2">
          {lastResult && (
            <div className="text-green-600 text-[10px] bg-green-50 border border-green-200 px-2 py-1 rounded">
              {lastResult}
            </div>
          )}
          {error && (
            <div className="text-red-500 text-[10px] bg-red-50 border border-red-200 px-2 py-1 rounded">
              {error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
