import { useTrades } from '../hooks/useTrades';
import { useTradingStore } from '../store';

function parseBase(sym: string) {
  if (sym.endsWith('USDT')) return sym.slice(0, -4);
  if (sym.endsWith('BTC')) return sym.slice(0, -3);
  return sym;
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

export function RecentTrades() {
  const trades = useTrades(30);
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);
  const base = parseBase(selectedSymbol);

  return (
    <div className="bg-white flex flex-col h-full border-t border-gray-200">
      {/* Header */}
      <div className="px-3 py-1.5 border-b border-gray-200 flex items-center justify-between">
        <span className="text-gray-800 text-xs font-semibold">Recent Trades</span>
        <div className="flex items-center gap-1">
          <div className="w-1.5 h-1.5 rounded-full bg-[#0ecb81] animate-pulse-dot" />
          <span className="text-gray-400 text-[10px]">Live</span>
        </div>
      </div>

      {/* Column headers */}
      <div className="flex items-center px-3 py-1 border-b border-gray-100">
        <span className="flex-1 text-gray-400 text-[10px]">Price (USDT)</span>
        <span className="flex-1 text-center text-gray-400 text-[10px]">Amount ({base})</span>
        <span className="flex-1 text-right text-gray-400 text-[10px]">Time</span>
      </div>

      {/* Trade rows */}
      <div className="flex flex-col overflow-y-auto flex-1">
        {trades.length === 0 ? (
          <div className="flex items-center justify-center h-16 text-gray-400 text-xs">
            Waiting for trades...
          </div>
        ) : (
          trades.map((trade) => {
            const isBuy = !trade.isBuyerMaker;
            const color = isBuy ? '#0ecb81' : '#f6465d';
            const price = parseFloat(trade.price);
            const qty = parseFloat(trade.qty);

            return (
              <div key={trade.id} className="flex items-center px-3 py-0.5 hover:bg-gray-50 cursor-default">
                <span className="flex-1 font-mono text-[11px]" style={{ color }}>
                  {price.toFixed(2)}
                </span>
                <span className="flex-1 text-center font-mono text-[11px] text-gray-700">
                  {qty.toFixed(3)}
                </span>
                <span className="flex-1 text-right font-mono text-[10px] text-gray-400">
                  {formatTime(trade.time)}
                </span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
