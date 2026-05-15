import { useOrderBook } from '../hooks/useOrderBook';
import { useTradingStore } from '../store';

function Row({
  price,
  size,
  total,
  depthPct,
  side,
  isWall,
}: {
  price: number;
  size: number;
  total: number;
  depthPct: number;
  side: 'bid' | 'ask';
  isWall: boolean;
}) {
  const isAsk = side === 'ask';
  const color = isAsk ? '#f6465d' : '#0ecb81';
  const bgColor = isAsk ? '#f6465d' : '#0ecb81';

  return (
    <div
      className="relative flex items-center h-5 px-2 cursor-default"
      style={isWall ? { backgroundColor: isAsk ? '#f6465d10' : '#0ecb8110' } : undefined}
      title={isWall ? `⚠ Liquidity wall — ${size.toFixed(2)} SOL` : undefined}
    >
      {/* Depth bar */}
      <div
        className="absolute top-0 bottom-0"
        style={{
          [isAsk ? 'left' : 'right']: 0,
          width: `${depthPct}%`,
          backgroundColor: bgColor,
          opacity: isWall ? 0.22 : 0.12,
        }}
      />
      {isWall && (
        <div
          className="absolute top-0 bottom-0 left-0 right-0 border-l-2"
          style={{ borderColor: bgColor, opacity: 0.5, pointerEvents: 'none' }}
        />
      )}
      <span className="flex-1 font-mono text-[11px]" style={{ color, fontWeight: isWall ? 700 : 400 }}>
        {price.toFixed(2)}
        {isWall && <span className="ml-1 text-[9px] opacity-70">⚠</span>}
      </span>
      <span className="flex-1 text-center font-mono text-[11px]" style={{ color: isWall ? color : '#374151', fontWeight: isWall ? 600 : 400 }}>
        {size.toFixed(2)}
      </span>
      <span className="flex-1 text-right font-mono text-[11px] text-gray-400">
        {total.toFixed(2)}
      </span>
    </div>
  );
}

export function OrderBook() {
  const { bids, asks, spread, spreadPct } = useOrderBook(15);
  const ticker = useTradingStore((s) => s.ticker);

  const currentPrice = ticker ? parseFloat(ticker.price) : null;
  const priceColor = ticker
    ? parseFloat(ticker.changePct) >= 0
      ? 'text-[#0ecb81]'
      : 'text-[#f6465d]'
    : 'text-gray-800';

  return (
    <div className="bg-white flex flex-col h-full border-l border-gray-200">
      {/* Header */}
      <div className="px-3 py-2 border-b border-gray-200">
        <span className="text-gray-800 text-xs font-semibold">Order Book</span>
      </div>

      {/* Column headers */}
      <div className="flex items-center px-2 py-1 border-b border-gray-100">
        <span className="flex-1 text-gray-400 text-[10px]">Price (USDT)</span>
        <span className="flex-1 text-center text-gray-400 text-[10px]">Amount</span>
        <span className="flex-1 text-right text-gray-400 text-[10px]">Total</span>
      </div>

      {/* Asks (sells) — reversed so lowest ask is at bottom */}
      <div className="flex flex-col-reverse overflow-hidden" style={{ height: '220px' }}>
        {asks.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-xs">No data</div>
        ) : (
          asks.map((ask) => (
            <Row key={ask.price} {...ask} side="ask" isWall={ask.isWall} />
          ))
        )}
      </div>

      {/* Spread */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-gray-50 border-y border-gray-200">
        <span className={`font-mono text-sm font-bold ${priceColor}`}>
          {currentPrice != null ? `$${currentPrice.toFixed(2)}` : '—'}
        </span>
        <span className="text-gray-400 text-[10px]">
          Spread: <span className="text-gray-700">${spread.toFixed(4)}</span>
          {' '}(<span className="text-gray-700">{spreadPct.toFixed(4)}%</span>)
        </span>
      </div>

      {/* Bids (buys) */}
      <div className="flex flex-col overflow-hidden" style={{ height: '220px' }}>
        {bids.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 text-xs">No data</div>
        ) : (
          bids.map((bid) => (
            <Row key={bid.price} {...bid} side="bid" isWall={bid.isWall} />
          ))
        )}
      </div>
    </div>
  );
}
