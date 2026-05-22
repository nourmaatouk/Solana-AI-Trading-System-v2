import { useTradingStore } from '../store';
import { parseMarketSymbol } from '../utils/symbol';

function Skeleton({ w = 'w-16' }: { w?: string }) {
  return <div className={`h-4 ${w} rounded animate-pulse`} style={{ background: 'var(--border)' }} />;
}

export function MarketHeader() {
  const ticker = useTradingStore((s) => s.ticker);
  const connectionStatus = useTradingStore((s) => s.connectionStatus);
  const exchange = useTradingStore((s) => s.exchange);
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);
  const { base, quote } = parseMarketSymbol(selectedSymbol);

  const priceColor = ticker
    ? parseFloat(ticker.changePct) >= 0 ? 'var(--green)' : 'var(--red)'
    : 'var(--text-1)';

  const statusDot = { connected: 'var(--green)', reconnecting: 'var(--accent)', disconnected: 'var(--red)' }[connectionStatus];
  const statusLabel = { connected: 'Live', reconnecting: 'Reconnecting', disconnected: 'Disconnected' }[connectionStatus];

  return (
    <header className="px-4 py-2 flex items-center justify-between flex-shrink-0"
      style={{ background: 'var(--bg-panel)', borderBottom: '1px solid var(--border)' }}>
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <div className="w-6 h-6 rounded flex items-center justify-center font-bold text-xs"
            style={{ background: 'var(--accent)', color: 'var(--bg-page)' }}>S</div>
          <span className="font-bold text-sm tracking-wide" style={{ color: 'var(--text-1)' }}>Solvanta AI</span>
        </div>
        <div className="h-4 w-px" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1">
          <span className="font-semibold text-sm" style={{ color: 'var(--text-1)' }}>{base}</span>
          <span className="text-sm" style={{ color: 'var(--text-3)' }}>/{quote}</span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div>{ticker
          ? <span className="text-xl font-bold font-mono" style={{ color: priceColor }}>${parseFloat(ticker.price).toFixed(2)}</span>
          : <Skeleton w="w-24" />}
        </div>
        {[
          { label: '24h Change', val: ticker ? `${parseFloat(ticker.changePct) >= 0 ? '+' : ''}${parseFloat(ticker.changePct).toFixed(2)}%` : null, color: priceColor },
          { label: '24h High',   val: ticker ? `$${parseFloat(ticker.high).toFixed(2)}`   : null, color: 'var(--text-1)' },
          { label: '24h Low',    val: ticker ? `$${parseFloat(ticker.low).toFixed(2)}`    : null, color: 'var(--text-1)' },
          { label: '24h Volume', val: ticker ? `${parseFloat(ticker.volume).toLocaleString(undefined, { maximumFractionDigits: 0 })} ${base}` : null, color: 'var(--text-1)' },
          { label: '24h Quote Vol', val: ticker ? `$${(parseFloat(ticker.quoteVolume) / 1_000_000).toFixed(1)}M` : null, color: 'var(--text-1)' },
        ].map(({ label, val, color }) => (
          <div key={label} className="flex flex-col items-center">
            <span className="text-[10px]" style={{ color: 'var(--text-3)' }}>{label}</span>
            {val ? <span className="font-mono text-xs font-medium" style={{ color }}>{val}</span> : <Skeleton />}
          </div>
        ))}
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 px-2 py-1 rounded"
          style={{ background: 'var(--bg-sub)', border: '1px solid var(--border)' }}>
          <div className="w-1.5 h-1.5 rounded-full animate-pulse-dot" style={{ background: statusDot }} />
          <span className="text-[10px]" style={{ color: 'var(--text-2)' }}>{statusLabel}</span>
          <span className="text-[10px] uppercase" style={{ color: 'var(--text-1)' }}>{exchange}</span>
        </div>
        <div className="px-2 py-1 rounded" style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)' }}>
          <span className="text-[10px] font-medium" style={{ color: 'var(--accent)' }}>Paper Trading</span>
        </div>
      </div>
    </header>
  );
}
