import { useOrderBook } from '../hooks/useOrderBook';
import { useTradingStore } from '../store';
import { ORDERBOOK_LEVELS } from '../config';
import { parseMarketSymbol } from '../utils/symbol';

function Row({ price, size, total, depthPct, side, isWall, baseAsset }: {
  price: number; size: number; total: number; depthPct: number; side: 'bid' | 'ask'; isWall: boolean; baseAsset: string;
}) {
  const isAsk = side === 'ask';
  const color = isAsk ? 'var(--red)' : 'var(--green)';
  const rawColor = isAsk ? '#f6465d' : '#0ecb81';

  return (
      <div className="relative flex items-center h-5 px-2 cursor-default"
        style={isWall ? { backgroundColor: isAsk ? '#f6465d10' : '#0ecb8110' } : undefined}
      title={isWall ? `Liquidity wall - ${size.toFixed(2)} ${baseAsset}` : undefined}>
      <div className="absolute top-0 bottom-0"
        style={{ [isAsk ? 'left' : 'right']: 0, width: `${depthPct}%`, backgroundColor: rawColor, opacity: isWall ? 0.22 : 0.12 }} />
      {isWall && (
        <div className="absolute top-0 bottom-0 left-0 right-0 border-l-2"
          style={{ borderColor: rawColor, opacity: 0.5, pointerEvents: 'none' }} />
      )}
      <span className="flex-1 font-mono text-[11px]" style={{ color, fontWeight: isWall ? 700 : 400 }}>
        {price.toFixed(2)}{isWall && <span className="ml-1 text-[9px] opacity-70">⚠</span>}
      </span>
      <span className="flex-1 text-center font-mono text-[11px]"
        style={{ color: isWall ? color : 'var(--text-1)', fontWeight: isWall ? 600 : 400 }}>
        {size.toFixed(2)}
      </span>
      <span className="flex-1 text-right font-mono text-[11px]" style={{ color: 'var(--text-3)' }}>
        {total.toFixed(2)}
      </span>
    </div>
  );
}

export function OrderBook() {
  const { bids, asks, spread, spreadPct } = useOrderBook(ORDERBOOK_LEVELS);
  const ticker = useTradingStore((s) => s.ticker);
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);
  const { base, quote } = parseMarketSymbol(selectedSymbol);
  const currentPrice = ticker ? parseFloat(ticker.price) : null;
  const priceColor = ticker
    ? parseFloat(ticker.changePct) >= 0 ? 'var(--green)' : 'var(--red)'
    : 'var(--text-1)';

  return (
    <div className="flex flex-col h-full" style={{ background: 'var(--bg-panel)', borderLeft: '1px solid var(--border)' }}>
      <div className="px-3 py-2" style={{ borderBottom: '1px solid var(--border)' }}>
        <span className="text-xs font-semibold" style={{ color: 'var(--text-1)' }}>Order Book</span>
      </div>
      <div className="flex items-center px-2 py-1" style={{ borderBottom: '1px solid var(--border)' }}>
        {[`Price (${quote || 'quote'})`, `Amount (${base})`, 'Total'].map((h, i) => (
          <span key={h} className={`flex-1 text-[10px] ${i === 1 ? 'text-center' : i === 2 ? 'text-right' : ''}`}
            style={{ color: 'var(--text-3)' }}>{h}</span>
        ))}
      </div>
      <div className="flex flex-col-reverse overflow-y-auto" style={{ height: '220px' }}>
        {asks.length === 0
          ? <div className="flex items-center justify-center h-full text-xs" style={{ color: 'var(--text-3)' }}>No data</div>
          : asks.map((ask) => <Row key={ask.price} {...ask} side="ask" isWall={ask.isWall} baseAsset={base} />)}
      </div>
      <div className="flex items-center justify-between px-3 py-1.5"
        style={{ background: 'var(--bg-sub)', borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)' }}>
        <span className="font-mono text-sm font-bold" style={{ color: priceColor }}>
          {currentPrice != null ? `$${currentPrice.toFixed(2)}` : '—'}
        </span>
        <span className="text-[10px]" style={{ color: 'var(--text-3)' }}>
          Spread: <span style={{ color: 'var(--text-2)' }}>${spread.toFixed(4)}</span>
          {' '}(<span style={{ color: 'var(--text-2)' }}>{spreadPct.toFixed(4)}%</span>)
        </span>
      </div>
      <div className="flex flex-col overflow-y-auto" style={{ height: '220px' }}>
        {bids.length === 0
          ? <div className="flex items-center justify-center h-full text-xs" style={{ color: 'var(--text-3)' }}>No data</div>
          : bids.map((bid) => <Row key={bid.price} {...bid} side="bid" isWall={bid.isWall} baseAsset={base} />)}
      </div>
    </div>
  );
}
