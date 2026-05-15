import { useOrderBook } from '../hooks/useOrderBook';
import { useTradingStore } from '../store';

function Gauge({ pct, label, color }: { pct: number; label: string; color: string }) {
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-12 h-12">
        <svg viewBox="0 0 36 36" className="w-full h-full -rotate-90">
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="#e5e7eb" strokeWidth="3" />
          <circle
            cx="18" cy="18" r="15.9" fill="none"
            stroke={color} strokeWidth="3"
            strokeDasharray={`${pct} ${100 - pct}`}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-[10px] font-bold" style={{ color }}>{Math.round(pct)}%</span>
        </div>
      </div>
      <span className="text-gray-400 text-[9px] mt-0.5">{label}</span>
    </div>
  );
}

function AlertBadge({ status }: { status: 'NORMAL' | 'WARNING' | 'SUSPICIOUS' }) {
  const configs = {
    NORMAL: { color: '#0ecb81', bg: '#0ecb8120', border: '#0ecb8140' },
    WARNING: { color: '#f0b90b', bg: '#f0b90b20', border: '#f0b90b40' },
    SUSPICIOUS: { color: '#f6465d', bg: '#f6465d20', border: '#f6465d40' },
  };
  const cfg = configs[status];
  return (
    <div
      className="px-2 py-0.5 rounded text-[10px] font-bold"
      style={{ color: cfg.color, backgroundColor: cfg.bg, border: `1px solid ${cfg.border}` }}
    >
      {status}
    </div>
  );
}

export function SpoofingPanel() {
  const { bids, asks, spread, spreadPct, imbalancePct } = useOrderBook(20);
  const signal = useTradingStore((s) => s.signal);
  const ticker = useTradingStore((s) => s.ticker);

  const spoofingConfidence = signal?.spoofing_confidence ?? 0;
  const spoofingPct = spoofingConfidence * 100;

  let imbalanceLabel = 'Balanced';
  let imbalanceColor = '#6b7280';
  if (imbalancePct > 60) { imbalanceLabel = 'Buy Pressure'; imbalanceColor = '#0ecb81'; }
  else if (imbalancePct < 40) { imbalanceLabel = 'Sell Pressure'; imbalanceColor = '#f6465d'; }

  let alertStatus: 'NORMAL' | 'WARNING' | 'SUSPICIOUS' = 'NORMAL';
  if (spoofingPct >= 70) alertStatus = 'SUSPICIOUS';
  else if (spoofingPct >= 30) alertStatus = 'WARNING';

  const largestBid = bids.reduce((max, b) => b.size > max.size ? b : max, { price: 0, size: 0, total: 0, depthPct: 0 });
  const largestAsk = asks.reduce((max, a) => a.size > max.size ? a : max, { price: 0, size: 0, total: 0, depthPct: 0 });

  const currentPrice = ticker ? parseFloat(ticker.price) : 0;

  return (
    <div className="bg-white flex flex-col border-l border-t border-gray-200">
      {/* Header */}
      <div className="px-3 py-2 border-b border-gray-200 flex items-center justify-between">
        <span className="text-gray-800 text-xs font-semibold">Spoofing Detection</span>
        <AlertBadge status={alertStatus} />
      </div>

      <div className="p-3 space-y-3">
        {/* Gauges */}
        <div className="flex items-center justify-around">
          <Gauge pct={imbalancePct} label="Bid Imbalance" color={imbalanceColor} />
          <div className="flex flex-col items-center">
            <div className="text-[10px] font-medium" style={{ color: imbalanceColor }}>{imbalanceLabel}</div>
            <div className="text-gray-400 text-[9px] mt-0.5">
              {imbalancePct.toFixed(1)}% bids
            </div>
          </div>
          <Gauge
            pct={spoofingPct}
            label="TEN-GNN"
            color={spoofingPct >= 70 ? '#f6465d' : spoofingPct >= 30 ? '#f0b90b' : '#0ecb81'}
          />
        </div>

        <div className="border-t border-gray-100" />

        {/* Walls */}
        <div>
          <div className="text-gray-400 text-[10px] mb-1.5 font-medium uppercase tracking-wider">Order Walls</div>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-gray-50 rounded p-2">
              <div className="text-[#0ecb81] text-[9px] font-medium mb-0.5">Largest Bid</div>
              <div className="text-gray-800 text-[10px] font-mono">
                ${largestBid.price > 0 ? largestBid.price.toFixed(2) : '—'}
              </div>
              <div className="text-gray-400 text-[9px]">
                {largestBid.size > 0 ? largestBid.size.toFixed(2) : '—'} SOL
              </div>
            </div>
            <div className="bg-gray-50 rounded p-2">
              <div className="text-[#f6465d] text-[9px] font-medium mb-0.5">Largest Ask</div>
              <div className="text-gray-800 text-[10px] font-mono">
                ${largestAsk.price > 0 ? largestAsk.price.toFixed(2) : '—'}
              </div>
              <div className="text-gray-400 text-[9px]">
                {largestAsk.size > 0 ? largestAsk.size.toFixed(2) : '—'} SOL
              </div>
            </div>
          </div>
        </div>

        <div className="border-t border-gray-100" />

        {/* Spread */}
        <div className="flex items-center justify-between">
          <span className="text-gray-400 text-[10px]">Spread</span>
          <span className="text-gray-700 text-[10px] font-mono">
            ${spread.toFixed(4)} ({spreadPct.toFixed(4)}%)
          </span>
        </div>

        {currentPrice > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-gray-400 text-[10px]">Mid Price</span>
            <span className="text-gray-700 text-[10px] font-mono">
              ${currentPrice.toFixed(2)}
            </span>
          </div>
        )}

        {/* Note */}
        <div className="bg-gray-50 rounded p-2">
          <div className="text-gray-400 text-[9px] leading-relaxed">
            Note: Rapid cancellation data not available from WebSocket stream. TEN-GNN confidence sourced from AI pipeline.
          </div>
        </div>
      </div>
    </div>
  );
}
