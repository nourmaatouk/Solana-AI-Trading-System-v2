import { useState } from 'react';
import {
  loadSettings,
  SETTINGS_DEFAULTS,
  SETTINGS_KEY,
  type Settings,
} from '../hooks/useSettings';
import { MAX_TRADE_SIZE_SOL } from '../config';
import {
  enableAutoExecute,
  disableAutoExecute,
  setBackendUserRiskLevel,
  setBackendMaxTradeSize,
} from '../services/executionApi';

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl p-5" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
      <div className="font-semibold text-sm mb-4" style={{ color: 'var(--text-1)' }}>{title}</div>
      <div className="space-y-4">{children}</div>
    </div>
  );
}

function Row({ label, desc, children }: { label: string; desc?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <div className="text-sm" style={{ color: 'var(--text-1)' }}>{label}</div>
        {desc && <div className="text-xs mt-0.5" style={{ color: 'var(--text-3)' }}>{desc}</div>}
      </div>
      <div className="flex-shrink-0">{children}</div>
    </div>
  );
}

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className="w-10 h-5 rounded-full transition-colors"
      style={{ background: value ? 'var(--accent)' : 'var(--border)' }}
    >
      <div className={`w-4 h-4 rounded-full shadow transition-transform mx-0.5 ${value ? 'translate-x-5' : 'translate-x-0'}`}
        style={{ background: 'var(--bg-panel)' }} />
    </button>
  );
}

function Pills<T extends string>({ options, value, onChange, colorFn }: {
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  colorFn?: (v: T) => { color: string; bg: string; border: string };
}) {
  return (
    <div className="flex gap-1.5">
      {options.map((o) => {
        const active = value === o;
        const custom = active ? colorFn?.(o) : undefined;
        return (
          <button key={o} onClick={() => onChange(o)}
            className="px-2.5 py-1 rounded-lg text-xs font-medium transition-colors border"
            style={active
              ? custom
                ? { color: custom.color, background: custom.bg, borderColor: custom.border }
                : { color: 'var(--accent)', background: 'var(--accent-bg)', borderColor: 'var(--accent-bd)' }
              : { color: 'var(--text-2)', borderColor: 'var(--border)', background: 'transparent' }
            }
          >
            {o}
          </button>
        );
      })}
    </div>
  );
}

export function SettingsPage() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const update = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    setSaved(false);
    setSaveError(null);
  };

  const handleSave = async () => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    // Notify same-tab consumers (e.g. ExecutionPanel) immediately.
    window.dispatchEvent(new Event('solvanta:settings'));

    // Sync execution-layer state with the backend so the AutoExecutor enforces
    // the same trading mode, risk preference, and per-trade size the user
    // picked here.
    try {
      await Promise.all([
        setBackendUserRiskLevel(settings.riskLevel),
        setBackendMaxTradeSize(settings.maxTradeSizeSol),
      ]);
      if (settings.tradingMode === 'Auto') {
        await enableAutoExecute();
      } else {
        await disableAutoExecute();
      }
      setSaved(true);
      setSaveError(null);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      setSaveError(`Settings saved locally but backend sync failed: ${(err as Error).message}`);
    }
  };

  const handleReset = () => {
    setSettings(SETTINGS_DEFAULTS);
    localStorage.removeItem(SETTINGS_KEY);
    window.dispatchEvent(new Event('solvanta:settings'));
    setSaved(false);
    setSaveError(null);
  };

  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: 'var(--bg-page)' }}>
      <div className="max-w-xl mx-auto space-y-4">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-xl font-bold mb-1" style={{ color: 'var(--text-1)' }}>Settings</h1>
            <p className="text-sm" style={{ color: 'var(--text-3)' }}>Preferences saved to your browser.</p>
          </div>
          <div className="flex gap-2">
            <button onClick={handleReset}
              className="px-3 py-1.5 text-xs rounded-lg border transition-colors"
              style={{ borderColor: 'var(--border)', color: 'var(--text-2)', background: 'transparent' }}>
              Reset
            </button>
            <button onClick={handleSave}
              className="px-4 py-1.5 text-xs rounded-lg font-medium transition-colors text-white"
              style={{ background: saved ? '#16a34a' : 'var(--accent)' }}>
              {saved ? '✓ Saved' : 'Save'}
            </button>
          </div>
        </div>

        {saveError && (
          <div className="px-4 py-3 text-xs rounded-xl"
            style={{ background: 'var(--red-bg)', border: '1px solid var(--red-bd)', color: 'var(--red)' }}>
            {saveError}
          </div>
        )}

        <Section title="Execution">
          <Row
            label="Max Trade Size"
            desc="SOL amount cap per trade. Must be ≤ the Guardian's MAX_TRADE_SIZE_SOL env limit."
          >
            <div className="flex items-center gap-1.5">
              <input
                type="number"
                step={0.001}
                min={0.001}
                max={MAX_TRADE_SIZE_SOL}
                value={settings.maxTradeSizeSol}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  if (Number.isFinite(v) && v > 0) update('maxTradeSizeSol', v);
                }}
                className="w-24 px-2.5 py-1.5 rounded-lg text-sm font-mono border focus:outline-none"
                style={{ background: 'var(--bg-sub)', color: 'var(--text-1)', borderColor: 'var(--border)' }}
              />
              <span className="text-xs" style={{ color: 'var(--text-3)' }}>SOL</span>
            </div>
          </Row>
          <Row
            label="Trading Mode"
            desc="Auto fires trades automatically when the AI issues a BUY/SELL signal above the confidence threshold."
          >
            <Pills
              options={['Manual', 'Semi-Auto', 'Auto'] as const}
              value={settings.tradingMode}
              onChange={(v) => update('tradingMode', v)}
              colorFn={(v) =>
                v === 'Auto'
                  ? { color: 'var(--accent)', bg: 'var(--accent-bg)', border: 'var(--accent-bd)' }
                  : { color: 'var(--text-2)', bg: 'transparent', border: 'var(--border)' }
              }
            />
          </Row>
          {settings.tradingMode === 'Auto' && (
            <div className="px-3 py-2 rounded-lg text-xs"
              style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)', color: 'var(--accent)' }}>
              Auto mode: every BUY/SELL signal above the confidence threshold will execute automatically.
              Guardian rules and emergency stop remain active at all times.
            </div>
          )}
        </Section>

        <Section title="Trading Defaults">
          <Row label="Default Risk Level" desc="Pre-selected risk when opening the trade panel">
            <Pills
              options={['LOW', 'MEDIUM', 'HIGH'] as const}
              value={settings.riskLevel}
              onChange={(v) => update('riskLevel', v)}
              colorFn={(v) =>
                v === 'LOW'    ? { color: 'var(--green)',  bg: 'var(--green-bg)',  border: 'var(--green-bd)'  }
                : v === 'MEDIUM' ? { color: 'var(--accent)', bg: 'var(--accent-bg)', border: 'var(--accent-bd)' }
                :                  { color: 'var(--red)',    bg: 'var(--red-bg)',    border: 'var(--red-bd)'    }
              }
            />
          </Row>
        </Section>

        <Section title="Signal & Pipeline">
          <Row label="Signal Poll Interval" desc="How often the dashboard fetches the latest AI signal">
            <select
              value={settings.signalPollInterval}
              onChange={(e) => update('signalPollInterval', Number(e.target.value))}
              className="rounded-lg px-3 py-1.5 text-sm border focus:outline-none"
              style={{ background: 'var(--bg-sub)', color: 'var(--text-1)', borderColor: 'var(--border)' }}
            >
              <option value={10}>10 seconds</option>
              <option value={30}>30 seconds</option>
              <option value={60}>60 seconds</option>
              <option value={120}>2 minutes</option>
            </select>
          </Row>
          <Row label="Spoofing Alerts" desc="Show alert badge when TEN-GNN detects suspicious order book activity">
            <Toggle value={settings.showSpoofingAlerts} onChange={(v) => update('showSpoofingAlerts', v)} />
          </Row>
        </Section>

        <Section title="Display">
          <Row label="Price Decimal Places" desc="Number of decimal places shown for prices">
            <select
              value={settings.priceDisplayDecimals}
              onChange={(e) => update('priceDisplayDecimals', Number(e.target.value))}
              className="rounded-lg px-3 py-1.5 text-sm border focus:outline-none"
              style={{ background: 'var(--bg-sub)', color: 'var(--text-1)', borderColor: 'var(--border)' }}
            >
              <option value={2}>2 decimals ($163.45)</option>
              <option value={4}>4 decimals ($163.4521)</option>
            </select>
          </Row>
        </Section>

        <Section title="About">
          <div className="text-xs space-y-1.5" style={{ color: 'var(--text-2)' }}>
            {[
              ['System',      'Solvanta AI Trading Dashboard'],
              ['Stack',       'React · Node.js · FastAPI · XGBoost'],
              ['Data source', 'Binance WebSocket'],
              ['AI models',   'XGBoost + FinBERT + TEN-GNN'],
              ['Execution',   'Guardian + Jupiter v6'],
            ].map(([name, value]) => (
              <div key={name} className="flex justify-between">
                <span style={{ color: 'var(--text-3)' }}>{name}</span>
                <span className="font-mono" style={{ color: 'var(--text-1)' }}>{value}</span>
              </div>
            ))}
          </div>
        </Section>

        <p className="text-xs text-center" style={{ color: 'var(--text-3)' }}>
          Settings are stored in localStorage. Save syncs the trading mode with the backend.
        </p>
      </div>
    </div>
  );
}
