import { useMemo } from 'react';
import { useTradingStore } from '../store';
import { NEUTRAL_IMBALANCE_PCT, ORDERBOOK_WALL_MULTIPLIER } from '../config';

export interface OrderBookLevel {
  price: number;
  size: number;
  total: number;
  depthPct: number;
  isWall: boolean;
}

export interface OrderBookData {
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  spread: number;
  spreadPct: number;
  bidTotalVolume: number;
  askTotalVolume: number;
  imbalancePct: number;
}

export function useOrderBook(levels = 15): OrderBookData {
  const depth = useTradingStore((s) => s.depth);

  return useMemo(() => {
    if (!depth) {
      return {
        bids: [],
        asks: [],
        spread: 0,
        spreadPct: 0,
        bidTotalVolume: 0,
        askTotalVolume: 0,
        imbalancePct: NEUTRAL_IMBALANCE_PCT,
      };
    }

    // Sort bids descending, asks ascending
    const rawBids = [...depth.bids]
      .map(([p, s]) => ({ price: parseFloat(p), size: parseFloat(s) }))
      .sort((a, b) => b.price - a.price)
      .slice(0, levels);

    const rawAsks = [...depth.asks]
      .map(([p, s]) => ({ price: parseFloat(p), size: parseFloat(s) }))
      .sort((a, b) => a.price - b.price)
      .slice(0, levels);

    // Median size for liquidity wall detection.
    const medianOf = (arr: number[]) => {
      if (arr.length === 0) return 0;
      const sorted = [...arr].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
    };
    const medianBidSize = medianOf(rawBids.map((b) => b.size));
    const medianAskSize = medianOf(rawAsks.map((a) => a.size));

    // Running totals
    let bidRunning = 0;
    const bidsWithTotal = rawBids.map((b) => {
      bidRunning += b.size;
      return { ...b, total: bidRunning };
    });

    let askRunning = 0;
    const asksWithTotal = rawAsks.map((a) => {
      askRunning += a.size;
      return { ...a, total: askRunning };
    });

    const maxBidTotal = bidsWithTotal[bidsWithTotal.length - 1]?.total ?? 1;
    const maxAskTotal = asksWithTotal[asksWithTotal.length - 1]?.total ?? 1;
    const maxTotal = Math.max(maxBidTotal, maxAskTotal);

    const bids: OrderBookLevel[] = bidsWithTotal.map((b) => ({
      ...b,
      depthPct: (b.total / maxTotal) * 100,
      isWall: medianBidSize > 0 && b.size > ORDERBOOK_WALL_MULTIPLIER * medianBidSize,
    }));

    const asks: OrderBookLevel[] = asksWithTotal.map((a) => ({
      ...a,
      depthPct: (a.total / maxTotal) * 100,
      isWall: medianAskSize > 0 && a.size > ORDERBOOK_WALL_MULTIPLIER * medianAskSize,
    }));

    const bestBid = bids[0]?.price ?? 0;
    const bestAsk = asks[0]?.price ?? 0;
    const spread = bestAsk - bestBid;
    const spreadPct = bestBid > 0 ? (spread / bestBid) * 100 : 0;

    const bidTotalVolume = depth.bids.reduce((sum, [, s]) => sum + parseFloat(s), 0);
    const askTotalVolume = depth.asks.reduce((sum, [, s]) => sum + parseFloat(s), 0);
    const totalVolume = bidTotalVolume + askTotalVolume;
    const imbalancePct = totalVolume > 0 ? (bidTotalVolume / totalVolume) * 100 : NEUTRAL_IMBALANCE_PCT;

    return { bids, asks, spread, spreadPct, bidTotalVolume, askTotalVolume, imbalancePct };
  }, [depth, levels]);
}
