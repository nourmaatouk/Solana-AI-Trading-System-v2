import { useTradingStore } from '../store';

const WATCHLIST_ORDER = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT'];
const CHART_SUPPORTED = new Set(['SOLUSDT']);

function formatSymbol(raw: string) {
  if (raw.endsWith('USDT')) return { base: raw.slice(0, -4), quote: 'USDT' };
  return { base: raw, quote: '' };
}

export function Watchlist() {
  const watchlist = useTradingStore((s) => s.watchlist);
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);
  const setSelectedSymbol = useTradingStore((s) => s.setSelectedSymbol);

  const map = new Map(watchlist.map((w) => [w.symbol, w]));
  const items = WATCHLIST_ORDER.map((sym) => map.get(sym) ?? { symbol: sym, price: '—', change: '0', changePct: '0', volume: '0' });

  return (
    <div className="flex flex-col h-full" style={{ background: 'var(--bg-panel)', borderRight: '1px solid var(--border)' }}>
      <div className="px-3 py-2" style={{ borderBottom: '1px solid var(--border)' }}>
        <span className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--text-3)' }}>Watchlist</span>
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
              className={`w-full text-left px-3 py-2.5 transition-colors ${chartSupported ? 'cursor-pointer' : 'cursor-default opacity-60'}`}
              style={{
                borderBottom: '1px solid var(--border)',
                borderLeft: isSelected ? '2px solid var(--accent)' : '2px solid transparent',
                background: isSelected ? 'var(--accent-bg)' : undefined,
              }}
              onMouseEnter={(e) => { if (chartSupported && !isSelected) (e.currentTarget as HTMLElement).style.background = 'var(--bg-sub)'; }}
              onMouseLeave={(e) => { if (!isSelected) (e.currentTarget as HTMLElement).style.background = 'transparent'; }}
              onClick={() => { if (chartSupported) setSelectedSymbol(item.symbol); }}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold" style={{ color: 'var(--text-1)' }}>
                    {base}<span style={{ color: 'var(--text-3)', fontWeight: 400 }}>/{quote}</span>
                  </div>
                  <div className="text-[10px] mt-0.5" style={{ color: 'var(--text-3)' }}>
                    Vol: {item.volume !== '0' ? parseFloat(item.volume).toLocaleString(undefined, { maximumFractionDigits: 0 }) : '—'}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs font-mono" style={{ color: 'var(--text-1)' }}>
                    {item.price !== '—' ? `$${parseFloat(item.price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: parseFloat(item.price) < 10 ? 4 : 2 })}` : '—'}
                  </div>
                  <div className="text-[10px] font-mono" style={{ color: isPositive ? 'var(--green)' : 'var(--red)' }}>
                    {item.changePct !== '0' ? `${isPositive ? '+' : ''}${pct.toFixed(2)}%` : '—'}
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>
      <div className="px-3 py-2" style={{ borderTop: '1px solid var(--border)' }}>
        <div className="text-[10px]" style={{ color: 'var(--text-3)' }}>AI signals: SOL/USDT only</div>
      </div>
    </div>
  );
}
