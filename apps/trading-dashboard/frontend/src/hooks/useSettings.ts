import { useEffect, useState } from 'react';
import { EXECUTION_TRADE_DEFAULT_SOL } from '../config';

export type TradingMode = 'Manual' | 'Semi-Auto' | 'Auto';

export interface Settings {
  // Display
  priceDisplayDecimals: number;
  // Signal
  signalPollInterval: number;
  showSpoofingAlerts: boolean;
  // Trading
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  tradingMode: TradingMode;
  // Execution
  maxTradeSizeSol: number;
}

export const SETTINGS_KEY = 'solvanta_settings';

export const SETTINGS_DEFAULTS: Settings = {
  priceDisplayDecimals: 2,
  signalPollInterval: 60,
  showSpoofingAlerts: true,
  riskLevel: 'MEDIUM',
  tradingMode: 'Manual',
  maxTradeSizeSol: EXECUTION_TRADE_DEFAULT_SOL,
};

export function loadSettings(): Settings {
  try {
    return { ...SETTINGS_DEFAULTS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}') };
  } catch {
    return SETTINGS_DEFAULTS;
  }
}

/** Read settings from localStorage. Stays in sync across SPA navigation via
 *  storage events (cross-tab) and a custom 'solvanta:settings' event
 *  (same-tab saves from SettingsPage). */
export function useSettings(): Settings {
  const [settings, setSettings] = useState<Settings>(loadSettings);

  useEffect(() => {
    const reload = () => setSettings(loadSettings());
    window.addEventListener('storage', reload);
    window.addEventListener('solvanta:settings', reload);
    return () => {
      window.removeEventListener('storage', reload);
      window.removeEventListener('solvanta:settings', reload);
    };
  }, []);

  return settings;
}
