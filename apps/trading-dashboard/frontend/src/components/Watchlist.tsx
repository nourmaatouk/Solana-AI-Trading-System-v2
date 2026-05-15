import { useTradingStore } from '../store';

const WATCHLIST_ORDER = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT'];
// Only SOL has full chart/orderbook streaming; others are price-only tickers
const CHART_SUPPORTED = new Set(['SOLUSDT']);

function formatSymbol(raw: string): { base: string; quote: string } {
  if (raw.endsWith('USDT')) {
    return { base: raw.slice(0, -4), quote: 'USDT' };
  }
  return { base: raw, quote: '' };
}

export function Watchlist() {
  const watchlist = useTradingStore((s) => s.watchlist);
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);
  const setSelectedSymbol = useTradingStore((s) => s.setSelectedSymbol);

  const map = new Map(watchlist.map((w) => [w.symbol, w]));
  const items = WATCHLIST_ORDER.map((sym) => map.get(sym) ?? { symbol: sym, price: '—', change: '0', changePct: '0', volume: '0' });

  return (
    <div className="bg-white border-r border-gray-200 flex flex-col h-full">
      <div className="px-3 py-2 border-b border-gray-200">
        <span className="text-gray-500 text-xs font-medium uppercase tracking-wider">Watchlist</span>
      </div>

      <div className="flex flex-col overflow-y-auto flex-1">
        {items.map((item) => {
          const pct = parseFloat(item.changePct ?? '0');
          const isPositive = pct >= 0;
          const isSelected = item.symbol === selectedSymbol;
          const { base, quote } = formatSymbol(item.symbol);

          const chartSupported = CHART_SUPPORTED.has(item.symbol);
          return (
            <button
              key={item.symbol}
              title={chartSupported ? undefined : 'Chart & order book available for SOL/USDT only'}
              className={`w-full text-left px-3 py-2.5 border-b border-gray-100 transition-colors ${
                chartSupported ? 'hover:bg-gray-50 cursor-pointer' : 'cursor-default opacity-70'
              } ${isSelected ? 'border-l-2 border-l-indigo-500 bg-indigo-50/50' : ''}`}
              onClick={() => { if (chartSupported) setSelectedSymbol(item.symbol); }}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-gray-800 text-xs font-semibold">
                    {base}
                    <span className="text-gray-400 font-normal">/{quote}</span>
                  </div>
                  <div className="text-gray-400 text-[10px] mt-0.5">
                    Vol: {item.volume !== '0' ? parseFloat(item.volume).toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-gray-800 text-xs font-mono">
                    {item.price !== '—' ? `$${parseFloat(item.price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: item.price && parseFloat(item.price) < 10 ? 4 : 2 })}` : '—'}
                  </div>
                  <div className={`text-[10px] font-mono ${isPositive ? 'text-[#0ecb81]' : 'text-[#f6465d]'}`}>
                    {item.changePct !== '0' ? `${isPositive ? '+' : ''}${pct.toFixed(2)}%` : '—'}
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      <div className="px-3 py-2 border-t border-gray-200">
        <div className="text-gray-400 text-[10px]">AI signals: SOL/USDT only</div>
      </div>
    </div>
  );
}
