import { useTradingStore } from '../store';

export function ErrorBanner() {
  const connectionStatus = useTradingStore((s) => s.connectionStatus);
  const exchange = useTradingStore((s) => s.exchange);
  const ticker = useTradingStore((s) => s.ticker);

  if (connectionStatus === 'connected') return null;

  if (connectionStatus === 'reconnecting') {
    return (
      <div className="flex items-center gap-2 px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 text-amber-600 text-xs">
        <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse inline-block" />
        <span>
          {ticker ? `Reconnecting to ${exchange.toUpperCase()} — last price: $${parseFloat(ticker.price).toFixed(2)}` : `Connecting to ${exchange.toUpperCase()} market data…`}
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 px-4 py-2 bg-red-500/10 border-b border-red-500/20 text-red-500 text-xs">
      <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
      <span>
        Market data offline — all WebSocket sources unavailable.
        {ticker ? ` Last known price: $${parseFloat(ticker.price).toFixed(2)} (stale)` : ' Check your internet connection.'}
      </span>
    </div>
  );
}
