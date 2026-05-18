import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchFusion } from '../services/api';
import { useTradingStore } from '../store';
import type { AISignalData } from '../types';
import { SIGNAL_POLL_INTERVAL_MS } from '../config';

function _notify(action: string, confidence: number) {
  if (typeof window === 'undefined' || !('Notification' in window)) return;
  const body = `Confidence: ${(confidence * 100).toFixed(1)}% — check the dashboard.`;
  const title = action === 'BUY' ? '📈 BUY signal fired' : '📉 SELL signal fired';

  if (Notification.permission === 'granted') {
    new Notification(title, { body, tag: 'ai-signal', icon: '/favicon.ico' });
  } else if (Notification.permission === 'default') {
    Notification.requestPermission().then((perm) => {
      if (perm === 'granted') new Notification(title, { body, tag: 'ai-signal', icon: '/favicon.ico' });
    });
  }
}

export function useAISignal(): {
  signal: AISignalData | null;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  refresh: () => void;
} {
  const wsSignal      = useTradingStore((s) => s.signal);
  const setSignal     = useTradingStore((s) => s.setSignal);
  const alertsEnabled = useTradingStore((s) => s.alertsEnabled);

  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const prevActionRef                 = useRef<string | null>(null);
  const initializedRef                = useRef(false);

  const maybeNotify = useCallback((action: string, confidence: number) => {
    if (!alertsEnabled || !initializedRef.current) return;
    if (action === prevActionRef.current) return;
    prevActionRef.current = action;
    if (action === 'BUY' || action === 'SELL') _notify(action, confidence);
  }, [alertsEnabled]);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchFusion();
      setSignal(data);
      setLastUpdated(new Date());
      const action = data.action ?? data.signal ?? 'HOLD';
      if (!initializedRef.current) {
        // Seed prevActionRef with the current signal so the first fetch
        // never triggers a notification for a pre-existing stale signal.
        prevActionRef.current = action;
        initializedRef.current = true;
      } else {
        maybeNotify(action, data.confidence ?? 0);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch signal');
    } finally {
      setLoading(false);
    }
  }, [setSignal, maybeNotify]);

  // Keep prevActionRef in sync with WS-pushed signals so REST polling
  // doesn't re-fire a notification for a signal already delivered by WS.
  useEffect(() => {
    if (!wsSignal) return;
    const action = wsSignal.action ?? wsSignal.signal ?? 'HOLD';
    if (initializedRef.current) {
      maybeNotify(action, wsSignal.confidence ?? 0);
    }
  }, [wsSignal, maybeNotify]);

  useEffect(() => {
    fetchData();
    const timer = setInterval(fetchData, SIGNAL_POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [fetchData]);

  return { signal: wsSignal, loading, error, lastUpdated, refresh: fetchData };
}
