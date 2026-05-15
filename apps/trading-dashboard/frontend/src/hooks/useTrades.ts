import { useMemo } from 'react';
import { useTradingStore } from '../store';
import type { TradeData } from '../types';

export function useTrades(limit = 50): TradeData[] {
  const trades = useTradingStore((s) => s.trades);
  return useMemo(() => trades.slice(0, limit), [trades, limit]);
}
