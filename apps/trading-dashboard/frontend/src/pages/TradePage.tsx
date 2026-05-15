import { useState } from 'react';
import { useTradingStore } from '../store';
import { executePaperTrade } from '../services/api';

type Side = 'BUY' | 'SELL';
type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH';

const RISK_SIZES: Record<RiskLevel, number> = { LOW: 5, MEDIUM: 10, HIGH: 20 };

export function TradePage() {
  const ticker = useTradingStore((s) => s.ticker);
  const signal = useTradingStore((s) => s.signal);

  const [side, setSide] = useState<Side>('BUY');
  const [riskLevel, setRiskLevel] = useState<RiskLevel>('MEDIUM');
  const [qty, setQty] = useState('1');
  const [executing, setExecuting] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const currentPrice = ticker ? parseFloat(ticker.price) : null;
  const qtyNum = parseFloat(qty) || 0;
  const notional = currentPrice ? currentPrice * qtyNum : null;

  const aiAction = signal?.action ?? (signal as unknown as Record<string, unknown>)?.signal as string ?? null;
  const aiColor = aiAction === 'BUY' ? 'text-green-600' : aiAction === 'SELL' ? 'text-red-500' : 'text-amber-500';

  const handleExecute = async () => {
    if (qtyNum <= 0) { setError('Enter a valid quantity.'); return; }
    setExecuting(true);
    setError(null);
    setResult(null);
    try {
      const res = await executePaperTrade({ side, quantity: qtyNum, risk_level: riskLevel, mode: 'Manual' });
      const status = (res as Record<string, unknown>).status ?? 'triggered';
      setResult(`Paper ${side} submitted — pipeline run ${status}. Results will appear in the Dashboard in ~20 seconds.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Execution failed');
    } finally {
      setExecuting(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: '#f8f9fd' }}>
      <div className="max-w-xl mx-auto space-y-4">
        <div>
          <h1 className="text-xl font-bold text-gray-800 mb-1">Place a Trade</h1>
          <p className="text-gray-400 text-sm">Paper trading only. No real funds are used.</p>
        </div>

        {/* Disclaimer */}
        <div className="bg-yellow-50 border border-yellow-200 rounded-xl px-4 py-3 flex gap-2 items-start">
          <span className="text-yellow-500 text-base mt-0.5">⚠</span>
          <div className="text-xs text-yellow-700 leading-relaxed">
            <strong>Simulated trades only.</strong> This interface executes paper trades against the AI pipeline log.
            No real SOL wallet interaction occurs. All P&amp;L is simulated from a baseline equity of 1.0.
          </div>
        </div>

        {/* AI recommendation */}
        {signal && (
          <div className="bg-white rounded-xl border border-gray-200 px-5 py-4 flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-400 uppercase tracking-wider mb-1">AI Recommendation</div>
              <div className={`text-2xl font-bold ${aiColor}`}>{aiAction ?? '—'}</div>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-400 mb-1">Confidence</div>
              <div className="text-lg font-bold text-gray-800">{((signal.confidence ?? 0) * 100).toFixed(0)}%</div>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-400 mb-1">Live Price</div>
              <div className="text-lg font-bold font-mono text-gray-800">
                {currentPrice ? `$${currentPrice.toFixed(2)}` : '—'}
              </div>
            </div>
          </div>
        )}

        {/* Order form */}
        <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4">
          <div className="font-semibold text-gray-800 text-sm">Order Details</div>

          {/* Side */}
          <div>
            <label className="block text-xs text-gray-400 mb-2">Side</label>
            <div className="flex rounded-xl border border-gray-200 overflow-hidden">
              <button
                onClick={() => setSide('BUY')}
                className={`flex-1 py-2.5 text-sm font-bold transition-colors ${
                  side === 'BUY' ? 'bg-green-500 text-white' : 'bg-white text-gray-400 hover:bg-green-50 hover:text-green-600'
                }`}
              >
                BUY
              </button>
              <button
                onClick={() => setSide('SELL')}
                className={`flex-1 py-2.5 text-sm font-bold transition-colors ${
                  side === 'SELL' ? 'bg-red-500 text-white' : 'bg-white text-gray-400 hover:bg-red-50 hover:text-red-500'
                }`}
              >
                SELL
              </button>
            </div>
          </div>

          {/* Quantity */}
          <div>
            <label className="block text-xs text-gray-400 mb-2">Quantity (SOL)</label>
            <input
              type="number"
              min="0.01"
              step="0.1"
              value={qty}
              onChange={(e) => setQty(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 font-mono"
            />
            {notional !== null && (
              <div className="text-xs text-gray-400 mt-1">
                Notional: <span className="font-mono text-gray-600">${notional.toFixed(2)} USDT</span>
              </div>
            )}
          </div>

          {/* Risk level */}
          <div>
            <label className="block text-xs text-gray-400 mb-2">Risk Level (position size preset)</label>
            <div className="flex gap-2">
              {(['LOW', 'MEDIUM', 'HIGH'] as RiskLevel[]).map((r) => (
                <button
                  key={r}
                  onClick={() => { setRiskLevel(r); setQty(RISK_SIZES[r].toString()); }}
                  className={`flex-1 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
                    riskLevel === r
                      ? r === 'LOW' ? 'bg-green-50 text-green-600 border-green-200'
                        : r === 'MEDIUM' ? 'bg-yellow-50 text-yellow-600 border-yellow-200'
                        : 'bg-red-50 text-red-500 border-red-200'
                      : 'text-gray-500 border-gray-200 hover:bg-gray-50'
                  }`}
                >
                  {r} ({RISK_SIZES[r]} SOL)
                </button>
              ))}
            </div>
          </div>

          {/* Order preview */}
          <div className="bg-gray-50 rounded-lg p-3 text-xs space-y-1">
            <div className="flex justify-between"><span className="text-gray-400">Order type</span><span className="text-gray-700 font-medium">Market (paper)</span></div>
            <div className="flex justify-between"><span className="text-gray-400">Side</span><span className={`font-bold ${side === 'BUY' ? 'text-green-600' : 'text-red-500'}`}>{side}</span></div>
            <div className="flex justify-between"><span className="text-gray-400">Quantity</span><span className="font-mono text-gray-700">{qtyNum} SOL</span></div>
            <div className="flex justify-between"><span className="text-gray-400">Est. price</span><span className="font-mono text-gray-700">{currentPrice ? `$${currentPrice.toFixed(2)}` : '—'}</span></div>
            <div className="flex justify-between"><span className="text-gray-400">Est. notional</span><span className="font-mono text-gray-700">{notional ? `$${notional.toFixed(2)}` : '—'}</span></div>
          </div>

          <button
            onClick={handleExecute}
            disabled={executing || qtyNum <= 0}
            className={`w-full py-2.5 rounded-xl text-sm font-bold transition-colors disabled:opacity-50 ${
              side === 'BUY'
                ? 'bg-green-500 hover:bg-green-600 text-white'
                : 'bg-red-500 hover:bg-red-600 text-white'
            }`}
          >
            {executing ? 'Executing…' : `Execute Paper ${side}`}
          </button>
        </div>

        {result && (
          <div className="bg-green-50 border border-green-200 rounded-xl px-4 py-3 text-xs text-green-700">{result}</div>
        )}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-xs text-red-600">{error}</div>
        )}
      </div>
    </div>
  );
}
