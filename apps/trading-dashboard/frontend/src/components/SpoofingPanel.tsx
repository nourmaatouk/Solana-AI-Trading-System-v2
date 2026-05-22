import { useOrderBook } from '../hooks/useOrderBook';
import { useTradingStore } from '../store';
import {
  IMBALANCE_BUY_PRESSURE_PCT,
  IMBALANCE_SELL_PRESSURE_PCT,
  SPOOFING_HIGH,
  SPOOFING_MEDIUM,
  SPOOFING_ORDERBOOK_LEVELS,
} from '../config';
import { parseMarketSymbol } from '../utils/symbol';

function Gauge({ pct, label, color }: { pct: number; label: string; color: string }) {
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-12 h-12">
        <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="var(--border)" strokeWidth="3" />
          <circle cx="18" cy="18" r="15.9" fill="none" stroke={color} strokeWidth="3"
            strokeDasharray={`${pct} ${100 - pct}`} strokeLinecap="round" />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-[10px] font-bold" style={{ color }}>{Math.round(pct)}%</span>
        </div>
      </div>
      <span className="text-[9px] mt-0.5" style={{ color: 'var(--text-3)' }}>{label}</span>
    </div>
  );
}

function AlertBadge({ status }: { status: 'NORMAL' | 'WARNING' | 'SUSPICIOUS' }) {
  const configs = {
    NORMAL:     { color: '#0ecb81', bg: '#0ecb8120', border: '#0ecb8140' },
    WARNING:    { color: '#8B5CF6', bg: '#8B5CF620', border: '#8B5CF640' },
    SUSPICIOUS: { color: '#f6465d', bg: '#f6465d20', border: '#f6465d40' },
  };
  const { color, bg, border } = configs[status];
  return (
    <div className="px-2 py-0.5 rounded text-[10px] font-bold" style={{ color, backgroundColor: bg, border: `1px solid ${border}` }}>
      {status}
    </div>
  );
}

export function SpoofingPanel() {
  const { bids, asks, spread, spreadPct, imbalancePct } = useOrderBook(SPOOFING_ORDERBOOK_LEVELS);
  const signal = useTradingStore((s) => s.signal);
  const ticker = useTradingStore((s) => s.ticker);
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);
  const { base } = parseMarketSymbol(selectedSymbol);

  const spoofingPct = (signal?.spoofing_confidence ?? 0) * 100;
  const spoofingHighPct = SPOOFING_HIGH * 100;
  const spoofingMediumPct = SPOOFING_MEDIUM * 100;
  let imbalanceLabel = 'Balanced', imbalanceColor = 'var(--text-2)';
  if (imbalancePct > IMBALANCE_BUY_PRESSURE_PCT) { imbalanceLabel = 'Buy Pressure';  imbalanceColor = 'var(--green)'; }
  else if (imbalancePct < IMBALANCE_SELL_PRESSURE_PCT) { imbalanceLabel = 'Sell Pressure'; imbalanceColor = 'var(--red)'; }

  const alertStatus: 'NORMAL' | 'WARNING' | 'SUSPICIOUS' =
    spoofingPct >= spoofingHighPct ? 'SUSPICIOUS' : spoofingPct >= spoofingMediumPct ? 'WARNING' : 'NORMAL';

  const largestBid = bids.reduce((m, b) => b.size > m.size ? b : m, { price: 0, size: 0, total: 0, depthPct: 0 });
  const largestAsk = asks.reduce((m, a) => a.size > m.size ? a : m, { price: 0, size: 0, total: 0, depthPct: 0 });
  const currentPrice = ticker ? parseFloat(ticker.price) : 0;
  const gnnColor = spoofingPct >= spoofingHighPct ? 'var(--red)' : spoofingPct >= spoofingMediumPct ? 'var(--accent)' : 'var(--green)';

  return (
    <div className="flex flex-col" style={{ background: 'var(--bg-panel)', borderLeft: '1px solid var(--border)', borderTop: '1px solid var(--border)' }}>
      <div className="px-3 py-2 flex items-center justify-between" style={{ borderBottom: '1px solid var(--border)' }}>
        <span className="text-xs font-semibold" style={{ color: 'var(--text-1)' }}>Spoofing Detection</span>
        <AlertBadge status={alertStatus} />
      </div>
      <div className="p-3 space-y-3">
        <div className="flex items-center justify-around">
          <Gauge pct={imbalancePct} label="Bid Imbalance" color={imbalanceColor} />
          <div className="flex flex-col items-center">
            <div className="text-[10px] font-medium" style={{ color: imbalanceColor }}>{imbalanceLabel}</div>
            <div className="text-[9px] mt-0.5" style={{ color: 'var(--text-3)' }}>{imbalancePct.toFixed(1)}% bids</div>
          </div>
          <Gauge pct={spoofingPct} label="TEN-GNN" color={gnnColor} />
        </div>

        <div style={{ borderTop: '1px solid var(--border)' }} />

        <div>
          <div className="text-[10px] mb-1.5 font-medium uppercase tracking-wider" style={{ color: 'var(--text-3)' }}>Order Walls</div>
          <div className="grid grid-cols-2 gap-2">
            {[{ label: 'Largest Bid', color: 'var(--green)', price: largestBid.price, size: largestBid.size },
              { label: 'Largest Ask', color: 'var(--red)',   price: largestAsk.price, size: largestAsk.size }].map(({ label, color, price, size }) => (
              <div key={label} className="rounded p-2" style={{ background: 'var(--bg-sub)' }}>
                <div className="text-[9px] font-medium mb-0.5" style={{ color }}>{label}</div>
                <div className="text-[10px] font-mono" style={{ color: 'var(--text-1)' }}>${price > 0 ? price.toFixed(2) : '—'}</div>
                <div className="text-[9px]" style={{ color: 'var(--text-3)' }}>{size > 0 ? size.toFixed(2) : '—'} {base}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ borderTop: '1px solid var(--border)' }} />

        <div className="flex items-center justify-between">
          <span className="text-[10px]" style={{ color: 'var(--text-3)' }}>Spread</span>
          <span className="text-[10px] font-mono" style={{ color: 'var(--text-2)' }}>${spread.toFixed(4)} ({spreadPct.toFixed(4)}%)</span>
        </div>
        {currentPrice > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-[10px]" style={{ color: 'var(--text-3)' }}>Mid Price</span>
            <span className="text-[10px] font-mono" style={{ color: 'var(--text-2)' }}>${currentPrice.toFixed(2)}</span>
          </div>
        )}
        <div className="rounded p-2" style={{ background: 'var(--bg-sub)' }}>
          <div className="text-[9px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
            Note: Rapid cancellation data not available from WebSocket stream. TEN-GNN confidence sourced from AI pipeline.
          </div>
        </div>
      </div>
    </div>
  );
}
