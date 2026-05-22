import { useTrades } from '../hooks/useTrades';
import { useTradingStore } from '../store';
import { RECENT_TRADES_VISIBLE } from '../config';
import { parseMarketSymbol } from '../utils/symbol';

function formatTime(ts: number): string {
  const d = new Date(ts);
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map((n) => String(n).padStart(2, '0')).join(':');
}

export function RecentTrades() {
  const trades = useTrades(RECENT_TRADES_VISIBLE);
  const selectedSymbol = useTradingStore((s) => s.selectedSymbol);
  const { base, quote } = parseMarketSymbol(selectedSymbol);

  return (
    <div className="flex flex-col h-full" style={{ background: 'var(--bg-panel)', borderTop: '1px solid var(--border)' }}>
      <div className="px-3 py-1.5 flex items-center justify-between" style={{ borderBottom: '1px solid var(--border)' }}>
        <span className="text-xs font-semibold" style={{ color: 'var(--text-1)' }}>Recent Trades</span>
        <div className="flex items-center gap-1">
          <div className="w-1.5 h-1.5 rounded-full animate-pulse-dot" style={{ background: 'var(--green)' }} />
          <span className="text-[10px]" style={{ color: 'var(--text-3)' }}>Live</span>
        </div>
      </div>
      <div className="flex items-center px-3 py-1" style={{ borderBottom: '1px solid var(--border)' }}>
        <span className="flex-1 text-[10px]" style={{ color: 'var(--text-3)' }}>Price ({quote || 'quote'})</span>
        <span className="flex-1 text-center text-[10px]" style={{ color: 'var(--text-3)' }}>Amount ({base})</span>
        <span className="flex-1 text-right text-[10px]" style={{ color: 'var(--text-3)' }}>Time</span>
      </div>
      <div className="flex flex-col overflow-y-auto flex-1">
        {trades.length === 0 ? (
          <div className="flex items-center justify-center h-16 text-xs" style={{ color: 'var(--text-3)' }}>
            Waiting for trades...
          </div>
        ) : trades.map((trade) => {
          const isBuy = !trade.isBuyerMaker;
          const color = isBuy ? 'var(--green)' : 'var(--red)';
          return (
            <div key={trade.id} className="flex items-center px-3 py-0.5 cursor-default"
              onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-sub)')}
              onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}>
              <span className="flex-1 font-mono text-[11px]" style={{ color }}>{parseFloat(trade.price).toFixed(2)}</span>
              <span className="flex-1 text-center font-mono text-[11px]" style={{ color: 'var(--text-1)' }}>{parseFloat(trade.qty).toFixed(3)}</span>
              <span className="flex-1 text-right font-mono text-[10px]" style={{ color: 'var(--text-3)' }}>{formatTime(trade.time)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
