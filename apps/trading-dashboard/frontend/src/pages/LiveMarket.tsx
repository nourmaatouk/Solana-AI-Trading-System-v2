import { MarketHeader } from '../components/MarketHeader';
import { PriceChart } from '../components/PriceChart';
import { OrderBook } from '../components/OrderBook';
import { RecentTrades } from '../components/RecentTrades';
import { Watchlist } from '../components/Watchlist';
import { AISignalPanel } from '../components/AISignalPanel';
import { SpoofingPanel } from '../components/SpoofingPanel';
export function LiveMarket() {
  return (
    <div className="flex flex-col h-full min-h-0" style={{ background: 'var(--bg-page)' }}>
      <MarketHeader />

      <div className="flex flex-1 min-h-0 overflow-hidden">
        <div className="flex flex-col flex-1 min-w-0">

          <div className="flex flex-1 min-h-0">
            <div className="flex-1 min-w-0">
              <PriceChart />
            </div>

            <div className="flex flex-col" style={{ width: 220, borderLeft: '1px solid var(--border)' }}>
              <div className="flex-1 min-h-0 overflow-hidden">
                <OrderBook />
              </div>
            </div>

            <div className="flex flex-col" style={{ width: 220, borderLeft: '1px solid var(--border)' }}>
              <AISignalPanel />
            </div>
          </div>

          <div className="flex flex-shrink-0" style={{ height: 280, borderTop: '1px solid var(--border)' }}>
            <div className="flex flex-col overflow-hidden" style={{ width: '38%', borderRight: '1px solid var(--border)' }}>
              <RecentTrades />
            </div>

            <div className="flex flex-col overflow-hidden" style={{ width: '31%', borderRight: '1px solid var(--border)' }}>
              <SpoofingPanel />
            </div>

            <div className="flex flex-col overflow-hidden" style={{ width: '31%' }}>
              <Watchlist />
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
