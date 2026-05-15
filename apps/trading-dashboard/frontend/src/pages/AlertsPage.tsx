import { useState, useEffect, useRef } from 'react';
import { useTradingStore } from '../store';

interface Alert {
  id: string;
  type: 'above' | 'below';
  price: number;
  label: string;
  triggered: boolean;
  createdAt: number;
}

const STORAGE_KEY = 'solvanta_alerts';

function loadAlerts(): Alert[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]') as Alert[];
  } catch {
    return [];
  }
}

function saveAlerts(alerts: Alert[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(alerts));
}

function parseSymbol(raw: string) {
  if (raw.endsWith('USDT')) return { base: raw.slice(0, -4), quote: 'USDT' };
  if (raw.endsWith('BTC')) return { base: raw.slice(0, -3), quote: 'BTC' };
  return { base: raw, quote: '' };
}

export function AlertsPage() {
  const ticker = useTradingStore((s) => s.ticker);
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);
  const { base, quote } = parseSymbol(selectedSymbol);
  const [alerts, setAlerts] = useState<Alert[]>(loadAlerts);
  const [type, setType] = useState<'above' | 'below'>('above');
  const [priceInput, setPriceInput] = useState('');
  const [labelInput, setLabelInput] = useState('');
  const [error, setError] = useState('');
  const prevTriggeredRef = useRef<Set<string>>(new Set());

  const currentPrice = ticker ? parseFloat(ticker.price) : null;

  // Check alerts against current price
  useEffect(() => {
    if (currentPrice === null) return;
    setAlerts((prev) => {
      let changed = false;
      const updated = prev.map((a) => {
        if (a.triggered) return a;
        const triggered = a.type === 'above' ? currentPrice >= a.price : currentPrice <= a.price;
        if (triggered && !prevTriggeredRef.current.has(a.id)) {
          prevTriggeredRef.current.add(a.id);
          changed = true;
          return { ...a, triggered: true };
        }
        return a;
      });
      if (changed) saveAlerts(updated);
      return changed ? updated : prev;
    });
  }, [currentPrice]);

  const handleAdd = () => {
    const price = parseFloat(priceInput);
    if (isNaN(price) || price <= 0) { setError('Enter a valid price.'); return; }
    setError('');
    const newAlert: Alert = {
      id: Date.now().toString(),
      type,
      price,
      label: labelInput.trim() || `Price ${type === 'above' ? '≥' : '≤'} $${price.toFixed(2)}`,
      triggered: false,
      createdAt: Date.now(),
    };
    const updated = [newAlert, ...alerts];
    setAlerts(updated);
    saveAlerts(updated);
    setPriceInput('');
    setLabelInput('');
  };

  const handleDelete = (id: string) => {
    const updated = alerts.filter((a) => a.id !== id);
    setAlerts(updated);
    saveAlerts(updated);
  };

  const handleClearTriggered = () => {
    const updated = alerts.filter((a) => !a.triggered);
    setAlerts(updated);
    saveAlerts(updated);
  };

  const triggeredCount = alerts.filter((a) => a.triggered).length;

  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: '#f8f9fd' }}>
      <div className="max-w-2xl mx-auto space-y-4">
        <div>
          <h1 className="text-xl font-bold text-gray-800 mb-1">Price Alerts</h1>
          <p className="text-gray-400 text-sm">
            Alerts are checked against the live {base}/{quote} price and stored in your browser.
            {currentPrice !== null && (
              <span className="ml-1 text-gray-600 font-medium">Current price: <span className="font-mono">${currentPrice.toFixed(2)}</span></span>
            )}
          </p>
        </div>

        {/* Add alert form */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="font-semibold text-gray-800 text-sm mb-4">Add New Alert</div>
          <div className="flex flex-wrap gap-3 items-end">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Type</label>
              <div className="flex rounded-lg border border-gray-200 overflow-hidden">
                {(['above', 'below'] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setType(t)}
                    className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                      type === t ? 'bg-indigo-500 text-white' : 'bg-white text-gray-500 hover:bg-gray-50'
                    }`}
                  >
                    Price {t === 'above' ? '≥' : '≤'}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Price (USD)</label>
              <input
                type="number"
                value={priceInput}
                onChange={(e) => setPriceInput(e.target.value)}
                placeholder={currentPrice ? currentPrice.toFixed(2) : '0.00'}
                className="w-32 border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
              />
            </div>
            <div className="flex-1 min-w-32">
              <label className="block text-xs text-gray-400 mb-1">Label (optional)</label>
              <input
                type="text"
                value={labelInput}
                onChange={(e) => setLabelInput(e.target.value)}
                placeholder="e.g. Support level"
                className="w-full border border-gray-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300"
              />
            </div>
            <button
              onClick={handleAdd}
              className="px-4 py-1.5 bg-indigo-500 hover:bg-indigo-600 text-white text-sm font-medium rounded-lg transition-colors"
            >
              Add Alert
            </button>
          </div>
          {error && <p className="text-red-500 text-xs mt-2">{error}</p>}
        </div>

        {/* Alert list */}
        <div className="bg-white rounded-xl border border-gray-200">
          <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
            <span className="font-semibold text-gray-800 text-sm">{alerts.length} Alert{alerts.length !== 1 ? 's' : ''}</span>
            {triggeredCount > 0 && (
              <button onClick={handleClearTriggered} className="text-xs text-gray-400 hover:text-red-500 transition-colors">
                Clear {triggeredCount} triggered
              </button>
            )}
          </div>
          {alerts.length === 0 ? (
            <div className="px-5 py-8 text-center text-gray-400 text-sm">No alerts set. Add one above.</div>
          ) : (
            alerts.map((a) => (
              <div key={a.id} className={`flex items-center justify-between px-5 py-3 border-b border-gray-50 last:border-0 ${a.triggered ? 'bg-green-50/50' : ''}`}>
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full flex-shrink-0 ${a.triggered ? 'bg-green-500' : 'bg-gray-300'}`} />
                  <div>
                    <div className="text-sm text-gray-700 font-medium">{a.label}</div>
                    <div className="text-xs text-gray-400">
                      {a.type === 'above' ? 'Alert when price ≥' : 'Alert when price ≤'}{' '}
                      <span className="font-mono text-gray-600">${a.price.toFixed(2)}</span>
                      {a.triggered && <span className="ml-2 text-green-600 font-semibold">✓ Triggered</span>}
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => handleDelete(a.id)}
                  className="text-gray-300 hover:text-red-400 transition-colors text-lg leading-none"
                  title="Delete alert"
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>

        <p className="text-xs text-gray-400 text-center">Alerts are saved in your browser (localStorage). They reset if you clear browser data.</p>
      </div>
    </div>
  );
}
