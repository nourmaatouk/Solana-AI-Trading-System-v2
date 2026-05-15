import { useTradingStore } from '../store';

function Skeleton({ w = 'w-16' }: { w?: string }) {
  return <div className={`h-4 ${w} bg-gray-200 rounded animate-pulse`} />;
}

function parseSymbol(raw: string) {
  if (raw.endsWith('USDT')) return { base: raw.slice(0, -4), quote: 'USDT' };
  if (raw.endsWith('BTC')) return { base: raw.slice(0, -3), quote: 'BTC' };
  return { base: raw, quote: '' };
}

export function MarketHeader() {
  const ticker = useTradingStore((s) => s.ticker);
  const connectionStatus = useTradingStore((s) => s.connectionStatus);
  const exchange = useTradingStore((s) => s.exchange);
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);
  const { base, quote } = parseSymbol(selectedSymbol);

  const priceColor = ticker
    ? parseFloat(ticker.changePct) >= 0
      ? 'text-[#0ecb81]'
      : 'text-[#f6465d]'
    : 'text-gray-800';

  const statusColor = {
    connected: 'bg-[#0ecb81]',
    reconnecting: 'bg-[#f0b90b]',
    disconnected: 'bg-[#f6465d]',
  }[connectionStatus];

  const statusLabel = {
    connected: 'Live',
    reconnecting: 'Reconnecting',
    disconnected: 'Disconnected',
  }[connectionStatus];

  return (
    <header className="bg-white border-b border-gray-200 px-4 py-2 flex items-center justify-between">
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <div className="w-6 h-6 rounded bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white font-bold text-xs">
            S
          </div>
          <span className="font-bold text-gray-800 text-sm tracking-wide">Solvanta AI</span>
        </div>
        <div className="h-4 w-px bg-gray-200" />
        <div className="flex items-center gap-1">
          <span className="text-gray-800 font-semibold text-sm">{base}</span>
          <span className="text-gray-400 text-sm">/{quote}</span>
        </div>
      </div>

      {/* Price info */}
      <div className="flex items-center gap-6">
        {/* Current price */}
        <div>
          {ticker ? (
            <span className={`text-xl font-bold font-mono ${priceColor}`}>
              ${parseFloat(ticker.price).toFixed(2)}
            </span>
          ) : (
            <Skeleton w="w-24" />
          )}
        </div>

        {/* 24h change */}
        <div className="flex flex-col items-center">
          <span className="text-gray-400 text-[10px]">24h Change</span>
          {ticker ? (
            <span className={`font-mono text-xs font-medium ${priceColor}`}>
              {parseFloat(ticker.changePct) >= 0 ? '+' : ''}{parseFloat(ticker.changePct).toFixed(2)}%
            </span>
          ) : (
            <Skeleton />
          )}
        </div>

        {/* 24h High */}
        <div className="flex flex-col items-center">
          <span className="text-gray-400 text-[10px]">24h High</span>
          {ticker ? (
            <span className="font-mono text-xs text-gray-700">${parseFloat(ticker.high).toFixed(2)}</span>
          ) : (
            <Skeleton />
          )}
        </div>

        {/* 24h Low */}
        <div className="flex flex-col items-center">
          <span className="text-gray-400 text-[10px]">24h Low</span>
          {ticker ? (
            <span className="font-mono text-xs text-gray-700">${parseFloat(ticker.low).toFixed(2)}</span>
          ) : (
            <Skeleton />
          )}
        </div>

        {/* Volume */}
        <div className="flex flex-col items-center">
          <span className="text-gray-400 text-[10px]">24h Volume</span>
          {ticker ? (
            <span className="font-mono text-xs text-gray-700">
              {parseFloat(ticker.volume).toLocaleString(undefined, { maximumFractionDigits: 0 })} SOL
            </span>
          ) : (
            <Skeleton />
          )}
        </div>

        {/* Quote volume */}
        <div className="flex flex-col items-center">
          <span className="text-gray-400 text-[10px]">24h Quote Vol</span>
          {ticker ? (
            <span className="font-mono text-xs text-gray-700">
              ${(parseFloat(ticker.quoteVolume) / 1_000_000).toFixed(1)}M
            </span>
          ) : (
            <Skeleton />
          )}
        </div>
      </div>

      {/* Right badges */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-gray-50 border border-gray-200">
          <div className={`w-1.5 h-1.5 rounded-full ${statusColor} animate-pulse-dot`} />
          <span className="text-gray-500 text-[10px]">{statusLabel}</span>
          <span className="text-gray-700 text-[10px] uppercase">{exchange}</span>
        </div>
        <div className="px-2 py-1 rounded bg-indigo-50 border border-indigo-200">
          <span className="text-indigo-600 text-[10px] font-medium">Paper Trading</span>
        </div>
      </div>
    </header>
  );
}
