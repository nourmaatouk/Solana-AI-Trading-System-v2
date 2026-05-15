import { useState } from 'react';

interface Settings {
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  tradingMode: 'Manual' | 'Semi-Auto' | 'Auto';
  signalPollInterval: number;
  showSpoofingAlerts: boolean;
  priceDisplayDecimals: number;
  theme: 'light';
}

const DEFAULTS: Settings = {
  riskLevel: 'MEDIUM',
  tradingMode: 'Manual',
  signalPollInterval: 60,
  showSpoofingAlerts: true,
  priceDisplayDecimals: 2,
  theme: 'light',
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
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <div className="font-semibold text-gray-800 text-sm mb-4">{title}</div>
      <div className="space-y-4">{children}</div>
    </div>
  );
}

function Row({ label, desc, children }: { label: string; desc?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <div className="text-sm text-gray-700">{label}</div>
        {desc && <div className="text-xs text-gray-400 mt-0.5">{desc}</div>}
      </div>
      <div className="flex-shrink-0">{children}</div>
    </div>
  );
}

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className={`w-10 h-5 rounded-full transition-colors ${value ? 'bg-indigo-500' : 'bg-gray-200'}`}
    >
      <div className={`w-4 h-4 bg-white rounded-full shadow transition-transform mx-0.5 ${value ? 'translate-x-5' : 'translate-x-0'}`} />
    </button>
  );
}

function Pills<T extends string>({ options, value, onChange, colorFn }: {
  options: T[];
  value: T;
  onChange: (v: T) => void;
  colorFn?: (v: T) => string;
}) {
  return (
    <div className="flex gap-1.5">
      {options.map((o) => (
        <button
          key={o}
          onClick={() => onChange(o)}
          className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors border ${
            value === o ? (colorFn?.(o) ?? 'bg-indigo-50 text-indigo-600 border-indigo-200') : 'text-gray-500 border-gray-200 hover:bg-gray-50'
          }`}
        >
          {o}
        </button>
      ))}
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
    <div className="flex-1 overflow-y-auto p-6" style={{ background: '#f8f9fd' }}>
      <div className="max-w-xl mx-auto space-y-4">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-800 mb-1">Settings</h1>
            <p className="text-gray-400 text-sm">Preferences saved to your browser.</p>
          </div>
          <div className="flex gap-2">
            <button onClick={handleReset} className="px-3 py-1.5 text-xs rounded-lg border border-gray-200 text-gray-500 hover:bg-gray-100 transition-colors">
              Reset
            </button>
            <button onClick={handleSave} className={`px-4 py-1.5 text-xs rounded-lg font-medium transition-colors ${saved ? 'bg-green-500 text-white' : 'bg-indigo-500 hover:bg-indigo-600 text-white'}`}>
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
              colorFn={(v) => v === 'LOW' ? 'bg-green-50 text-green-600 border-green-200' : v === 'MEDIUM' ? 'bg-yellow-50 text-yellow-600 border-yellow-200' : 'bg-red-50 text-red-500 border-red-200'}
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
              className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-300"
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
              className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-300"
            >
              <option value={2}>2 decimals ($163.45)</option>
              <option value={4}>4 decimals ($163.4521)</option>
            </select>
          </Row>
        </Section>

        <Section title="About">
          <div className="text-xs text-gray-500 space-y-1.5">
            <div className="flex justify-between"><span>System</span><span className="font-mono">Solvanta AI Trading Dashboard</span></div>
            <div className="flex justify-between"><span>Stack</span><span className="font-mono">React · Node.js · FastAPI · XGBoost</span></div>
            <div className="flex justify-between"><span>Data source</span><span className="font-mono">Binance WebSocket</span></div>
            <div className="flex justify-between"><span>AI models</span><span className="font-mono">XGBoost + FinBERT + TEN-GNN</span></div>
            <div className="flex justify-between"><span>Execution</span><span className="font-mono text-amber-600">Paper trading only</span></div>
          </div>
        </Section>

        <p className="text-xs text-gray-400 text-center">Settings are stored in localStorage and apply on next page load.</p>
      </div>
    </div>
  );
}
