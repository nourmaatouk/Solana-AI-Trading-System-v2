import { MarketHeader } from '../components/MarketHeader';
import { PriceChart } from '../components/PriceChart';
import { OrderBook } from '../components/OrderBook';
import { RecentTrades } from '../components/RecentTrades';
import { Watchlist } from '../components/Watchlist';
import { AISignalPanel } from '../components/AISignalPanel';
import { SpoofingPanel } from '../components/SpoofingPanel';
import { TradingControls } from '../components/TradingControls';

export function LiveMarket() {
  return (
    <div className="flex flex-col h-full min-h-0 bg-white">
      {/* Top: market ticker bar */}
      <MarketHeader />

      {/* Main trading area */}
      <div className="flex flex-1 min-h-0 overflow-hidden">

        {/* Centre column: chart + bottom panels */}
        <div className="flex flex-col flex-1 min-w-0">

          {/* Chart + right panels row */}
          <div className="flex flex-1 min-h-0">

            {/* Candlestick chart */}
            <div className="flex-1 min-w-0">
              <PriceChart />
            </div>

            {/* Right column: order book stacked over AI signal */}
            <div className="flex flex-col border-l border-gray-200" style={{ width: 220 }}>
              <div className="flex-1 min-h-0 overflow-hidden">
                <OrderBook />
              </div>
            </div>

            {/* Far-right column: AI signal */}
            <div className="flex flex-col border-l border-gray-200" style={{ width: 220 }}>
              <AISignalPanel />
            </div>
          </div>

          {/* Bottom strip: trades | spoofing | watchlist */}
          <div className="flex border-t border-gray-200 flex-shrink-0" style={{ height: 280 }}>
            {/* Recent trades */}
            <div className="flex flex-col border-r border-gray-200 overflow-hidden" style={{ width: '38%' }}>
              <RecentTrades />
            </div>

            {/* Spoofing detection */}
            <div className="flex flex-col border-r border-gray-200 overflow-hidden" style={{ width: '31%' }}>
              <SpoofingPanel />
            </div>

            {/* Watchlist */}
            <div className="flex flex-col overflow-hidden" style={{ width: '31%' }}>
              <Watchlist />
            </div>
          </div>
        </div>
      </div>

      {/* Bottom: trading controls */}
      <TradingControls />
    </div>
  );
}
