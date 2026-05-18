import { useState } from 'react';

interface Settings {
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  tradingMode: 'Manual' | 'Semi-Auto' | 'Auto';
  signalPollInterval: number;
  showSpoofingAlerts: boolean;
  priceDisplayDecimals: number;
}

const DEFAULTS: Settings = {
  riskLevel: 'MEDIUM',
  tradingMode: 'Manual',
  signalPollInterval: 60,
  showSpoofingAlerts: true,
  priceDisplayDecimals: 2,
};

const STORAGE_KEY = 'solvanta_settings';

function loadSettings(): Settings {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}') } as Settings;
  } catch {
    return DEFAULTS;
  }
}

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
          <button
            key={o}
            onClick={() => onChange(o)}
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

  const update = <K extends keyof Settings>(key: K, value: Settings[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    setSaved(false);
  };

  const handleSave = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    setSettings(DEFAULTS);
    localStorage.removeItem(STORAGE_KEY);
    setSaved(false);
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
          <Row label="Default Trading Mode" desc="Manual requires you to confirm every trade">
            <Pills
              options={['Manual', 'Semi-Auto', 'Auto'] as const}
              value={settings.tradingMode}
              onChange={(v) => update('tradingMode', v)}
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
              ['Execution',   'Paper trading only'],
            ].map(([name, value]) => (
              <div key={name} className="flex justify-between">
                <span style={{ color: 'var(--text-3)' }}>{name}</span>
                <span className="font-mono" style={{ color: name === 'Execution' ? 'var(--accent)' : 'var(--text-1)' }}>{value}</span>
              </div>
            ))}
          </div>
        </Section>

        <p className="text-xs text-center" style={{ color: 'var(--text-3)' }}>Settings are stored in localStorage and apply on next page load.</p>
      </div>
    </div>
  );
}
