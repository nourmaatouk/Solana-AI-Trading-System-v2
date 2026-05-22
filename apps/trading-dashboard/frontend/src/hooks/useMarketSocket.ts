import { useEffect } from 'react';
import { socketService } from '../services/socket';
import { useTradingStore, type ExecutionHistoryEntry } from '../store';
import type { TickerData, KlineData, TradeData, DepthData, WatchlistItem, AISignalData } from '../types';

interface TickerMsg { type: 'ticker'; symbol: string; data: TickerData }
interface KlineMsg { type: 'kline'; symbol: string; interval: string; data: KlineData }
interface KlinesSnapshotMsg { type: 'klines_snapshot'; symbol: string; data: KlineData[] }
interface TradeMsg { type: 'trade'; symbol: string; data: TradeData }
interface TradesSnapshotMsg { type: 'trades_snapshot'; symbol: string; data: TradeData[] }
interface DepthMsg { type: 'depth'; symbol: string; data: DepthData }
interface WatchlistMsg { type: 'watchlist'; data: WatchlistItem[] }
interface ConnectionMsg { type: 'connection'; status: 'connected' | 'reconnecting' | 'disconnected'; exchange: string }
interface SignalMsg { type: 'signal'; data: AISignalData }
interface ExecutionHistorySnapshotMsg { type: 'execution_history_snapshot'; data: ExecutionHistoryEntry[] }
interface ExecutionUpdateMsg { type: 'execution_update'; data: ExecutionHistoryEntry }

// selectedSymbol is the only reactive value this hook needs — it drives the
// useEffect re-registration when the user changes timeframe/symbol.
// All action calls go through getState() so they never trigger re-renders.
export function useMarketSocket(): void {
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);

  useEffect(() => {
    socketService.connect();

    const onTicker = (raw: unknown) => {
      const msg = raw as TickerMsg;
      const { selectedSymbol: sym, setTicker } = useTradingStore.getState();
      if (msg.symbol === sym) setTicker(msg.data);
    };

    const onKline = (raw: unknown) => {
      const msg = raw as KlineMsg;
      const { selectedSymbol: sym, addKline } = useTradingStore.getState();
      if (msg.symbol === sym) addKline(msg.data);
    };

    const onKlinesSnapshot = (raw: unknown) => {
      const msg = raw as KlinesSnapshotMsg;
      const { selectedSymbol: sym, setKlines } = useTradingStore.getState();
      if (msg.symbol === sym) setKlines(msg.data);
    };

    const onTrade = (raw: unknown) => {
      const msg = raw as TradeMsg;
      const { selectedSymbol: sym, addTrade } = useTradingStore.getState();
      if (msg.symbol === sym) addTrade(msg.data);
    };

    const onTradesSnapshot = (raw: unknown) => {
      const msg = raw as TradesSnapshotMsg;
      const { selectedSymbol: sym, setTrades } = useTradingStore.getState();
      if (msg.symbol === sym) setTrades(msg.data);
    };

    const onDepth = (raw: unknown) => {
      const msg = raw as DepthMsg;
      const { selectedSymbol: sym, setDepth } = useTradingStore.getState();
      if (msg.symbol === sym) setDepth(msg.data);
    };

    const onWatchlist = (raw: unknown) => {
      const msg = raw as WatchlistMsg;
      useTradingStore.getState().setWatchlist(msg.data);
    };

    const onConnection = (raw: unknown) => {
      const msg = raw as ConnectionMsg;
      useTradingStore.getState().setConnectionStatus(msg.status, msg.exchange);
    };

    const onSignal = (raw: unknown) => {
      const msg = raw as SignalMsg;
      useTradingStore.getState().setSignal(msg.data);
    };

    const onExecutionHistorySnapshot = (raw: unknown) => {
      const msg = raw as ExecutionHistorySnapshotMsg;
      useTradingStore.getState().setExecutionHistory(msg.data);
    };

    const onExecutionUpdate = (raw: unknown) => {
      const msg = raw as ExecutionUpdateMsg;
      useTradingStore.getState().prependExecutionEntry(msg.data);
    };

    socketService.on('ticker', onTicker);
    socketService.on('kline', onKline);
    socketService.on('klines_snapshot', onKlinesSnapshot);
    socketService.on('trade', onTrade);
    socketService.on('trades_snapshot', onTradesSnapshot);
    socketService.on('depth', onDepth);
    socketService.on('watchlist', onWatchlist);
    socketService.on('connection', onConnection);
    socketService.on('signal', onSignal);
    socketService.on('execution_history_snapshot', onExecutionHistorySnapshot);
    socketService.on('execution_update', onExecutionUpdate);

    return () => {
      socketService.off('ticker', onTicker);
      socketService.off('kline', onKline);
      socketService.off('klines_snapshot', onKlinesSnapshot);
      socketService.off('trade', onTrade);
      socketService.off('trades_snapshot', onTradesSnapshot);
      socketService.off('depth', onDepth);
      socketService.off('watchlist', onWatchlist);
      socketService.off('connection', onConnection);
      socketService.off('signal', onSignal);
      socketService.off('execution_history_snapshot', onExecutionHistorySnapshot);
      socketService.off('execution_update', onExecutionUpdate);
    };
  }, [selectedSymbol]);
}
