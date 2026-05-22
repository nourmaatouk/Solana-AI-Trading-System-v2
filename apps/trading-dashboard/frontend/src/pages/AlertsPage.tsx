import { useState, useEffect, useRef } from 'react';
import { useTradingStore } from '../store';
import { DEFAULT_CONFIDENCE_GATE } from '../config';
import { parseMarketSymbol } from '../utils/symbol';

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

export function AlertsPage() {
  const ticker = useTradingStore((s) => s.ticker);
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);
  const { base, quote } = parseMarketSymbol(selectedSymbol);
  const [alerts, setAlerts] = useState<Alert[]>(loadAlerts);
  const [type, setType] = useState<'above' | 'below'>('above');
  const [priceInput, setPriceInput] = useState('');
  const [labelInput, setLabelInput] = useState('');
  const [error, setError] = useState('');
  const prevTriggeredRef = useRef<Set<string>>(new Set());

  const currentPrice = ticker ? parseFloat(ticker.price) : null;

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

  const alertsEnabled  = useTradingStore((s) => s.alertsEnabled);
  const toggleAlerts   = useTradingStore((s) => s.toggleAlerts);
  const triggeredCount = alerts.filter((a) => a.triggered).length;

  const inputStyle = {
    background: 'var(--bg-sub)',
    color: 'var(--text-1)',
    border: '1px solid var(--border)',
    borderRadius: '0.5rem',
    padding: '6px 12px',
    fontSize: '0.875rem',
    outline: 'none',
  } as React.CSSProperties;

  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: 'var(--bg-page)' }}>
      <div className="max-w-2xl mx-auto space-y-4">
        <div>
          <h1 className="text-xl font-bold mb-1" style={{ color: 'var(--text-1)' }}>Price Alerts</h1>
          <p className="text-sm" style={{ color: 'var(--text-3)' }}>
            Alerts are checked against the live {base}/{quote} price and stored in your browser.
            {currentPrice !== null && (
              <span className="ml-1 font-medium" style={{ color: 'var(--text-2)' }}>
                Current price: <span className="font-mono">${currentPrice.toFixed(2)}</span>
              </span>
            )}
          </p>
        </div>

        {/* AI signal notifications toggle */}
        <div className="rounded-xl p-4 flex items-center justify-between" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
          <div>
            <div className="text-sm font-semibold" style={{ color: 'var(--text-1)' }}>AI Signal Notifications</div>
            <div className="text-xs mt-0.5" style={{ color: 'var(--text-3)' }}>
              Browser notification when a BUY or SELL signal crosses the {Math.round(DEFAULT_CONFIDENCE_GATE * 100)}% confidence gate
            </div>
          </div>
          <button
            onClick={toggleAlerts}
            className="relative inline-flex h-5 w-9 flex-shrink-0 rounded-full border-2 border-transparent transition-colors"
            style={{ background: alertsEnabled ? 'var(--accent)' : 'var(--border)' }}
            title={alertsEnabled ? 'Disable AI signal alerts' : 'Enable AI signal alerts'}
          >
            <span
              className="inline-block h-4 w-4 rounded-full bg-white shadow transition-transform"
              style={{ transform: alertsEnabled ? 'translateX(16px)' : 'translateX(0)' }}
            />
          </button>
        </div>

        {/* Add alert form */}
        <div className="rounded-xl p-5" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
          <div className="font-semibold text-sm mb-4" style={{ color: 'var(--text-1)' }}>Add New Alert</div>
          <div className="flex flex-wrap gap-3 items-end">
            <div>
              <label className="block text-xs mb-1" style={{ color: 'var(--text-3)' }}>Type</label>
              <div className="flex rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
                {(['above', 'below'] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setType(t)}
                    className="px-3 py-1.5 text-xs font-medium transition-colors"
                    style={type === t
                      ? { background: 'var(--accent)', color: '#fff' }
                      : { background: 'var(--bg-sub)', color: 'var(--text-2)' }
                    }
                  >
                    Price {t === 'above' ? '≥' : '≤'}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="block text-xs mb-1" style={{ color: 'var(--text-3)' }}>Price (USD)</label>
              <input
                type="number"
                value={priceInput}
                onChange={(e) => setPriceInput(e.target.value)}
                placeholder={currentPrice ? currentPrice.toFixed(2) : '0.00'}
                className="w-32"
                style={inputStyle}
              />
            </div>
            <div className="flex-1 min-w-32">
              <label className="block text-xs mb-1" style={{ color: 'var(--text-3)' }}>Label (optional)</label>
              <input
                type="text"
                value={labelInput}
                onChange={(e) => setLabelInput(e.target.value)}
                placeholder="e.g. Support level"
                className="w-full"
                style={inputStyle}
              />
            </div>
            <button
              onClick={handleAdd}
              className="px-4 py-1.5 text-sm font-medium rounded-lg transition-colors text-white"
              style={{ background: 'var(--accent)' }}
            >
              Add Alert
            </button>
          </div>
          {error && <p className="text-xs mt-2" style={{ color: 'var(--red)' }}>{error}</p>}
        </div>

        {/* Alert list */}
        <div className="rounded-xl overflow-hidden" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
          <div className="px-5 py-3 flex items-center justify-between" style={{ borderBottom: '1px solid var(--border)' }}>
            <span className="font-semibold text-sm" style={{ color: 'var(--text-1)' }}>
              {alerts.length} Alert{alerts.length !== 1 ? 's' : ''}
            </span>
            {triggeredCount > 0 && (
              <button onClick={handleClearTriggered} className="text-xs transition-colors" style={{ color: 'var(--text-3)' }}>
                Clear {triggeredCount} triggered
              </button>
            )}
          </div>
          {alerts.length === 0 ? (
            <div className="px-5 py-8 text-center text-sm" style={{ color: 'var(--text-3)' }}>No alerts set. Add one above.</div>
          ) : (
            alerts.map((a) => (
              <div key={a.id} className="flex items-center justify-between px-5 py-3"
                style={{
                  borderBottom: '1px solid var(--border)',
                  background: a.triggered ? 'var(--green-bg)' : 'transparent',
                }}>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ background: a.triggered ? 'var(--green)' : 'var(--border)' }} />
                  <div>
                    <div className="text-sm font-medium" style={{ color: 'var(--text-1)' }}>{a.label}</div>
                    <div className="text-xs" style={{ color: 'var(--text-3)' }}>
                      {a.type === 'above' ? 'Alert when price ≥' : 'Alert when price ≤'}{' '}
                      <span className="font-mono" style={{ color: 'var(--text-2)' }}>${a.price.toFixed(2)}</span>
                      {a.triggered && <span className="ml-2 font-semibold" style={{ color: 'var(--green)' }}>✓ Triggered</span>}
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => handleDelete(a.id)}
                  className="text-lg leading-none transition-colors"
                  style={{ color: 'var(--text-3)' }}
                  title="Delete alert"
                >
                  ×
                </button>
              </div>
            ))
          )}
        </div>

        <p className="text-xs text-center" style={{ color: 'var(--text-3)' }}>Alerts are saved in your browser (localStorage). They reset if you clear browser data.</p>
      </div>
    </div>
  );
}
