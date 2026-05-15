import { useState, useEffect, useCallback } from 'react';
import { fetchFusion } from '../services/api';
import { useTradingStore } from '../store';
import type { AISignalData } from '../types';

const POLL_INTERVAL_MS = 10_000;

export function useAISignal(): {
  signal: AISignalData | null;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  refresh: () => void;
} {
  const wsSignal = useTradingStore((s) => s.signal);
  const setSignal = useTradingStore((s) => s.setSignal);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchFusion();
      setSignal(data);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch signal');
    } finally {
      setLoading(false);
    }
  }, [setSignal]);

  // Initial fetch + polling fallback
  useEffect(() => {
    fetchData();
    const timer = setInterval(fetchData, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [fetchData]);

  return {
    signal: wsSignal,
    loading,
    error,
    lastUpdated,
    refresh: fetchData,
  };
}
