import { useEffect, useState, useRef } from 'react';
import { useTradingStore } from '../store';
import type { WatchlistItem } from '../types';

function TickerChip({ item }: { item: WatchlistItem }) {
  const pct = parseFloat(item.changePct);
  const isPos = pct >= 0;
  return (
    <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-gray-50 border border-gray-100 flex-shrink-0">
      <span className="text-xs font-semibold text-gray-700">{item.symbol.replace('USDT', '')}</span>
      <span className="text-xs font-mono text-gray-800">${parseFloat(item.price).toLocaleString('en-US', { maximumFractionDigits: 4 })}</span>
      <span className={`text-xs font-medium ${isPos ? 'text-[#16a34a]' : 'text-[#dc2626]'}`}>
        {isPos ? '+' : ''}{pct.toFixed(2)}%
      </span>
    </div>
  );
}

function NotifPanel({ onClose, notifications }: { onClose: () => void; notifications: string[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    }
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  return (
    <div ref={ref} className="absolute right-0 top-10 w-72 bg-white border border-gray-200 rounded-xl shadow-lg z-50">
      <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
        <span className="text-sm font-semibold text-gray-800">Notifications</span>
        <span className="text-xs text-gray-400">{notifications.length} recent</span>
      </div>
      <div className="max-h-64 overflow-y-auto">
        {notifications.length === 0 ? (
          <div className="px-4 py-6 text-center text-gray-400 text-sm">No notifications yet</div>
        ) : (
          notifications.map((n, i) => (
            <div key={i} className="px-4 py-2.5 border-b border-gray-50 text-xs text-gray-600 last:border-0">
              {n}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export function TopBar() {
  const watchlist = useTradingStore((s) => s.watchlist);
  const ticker = useTradingStore((s) => s.ticker);
  const connectionStatus = useTradingStore((s) => s.connectionStatus);
  const exchange = useTradingStore((s) => s.exchange);
  const signal = useTradingStore((s) => s.signal);

  const [timeStr, setTimeStr] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [notifOpen, setNotifOpen] = useState(false);
  const [notifications, setNotifications] = useState<string[]>([]);
  const prevSignalRef = useRef<string | null>(null);

  // Ticking clock
  useEffect(() => {
    const tick = () => {
      setTimeStr(new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  // Add notification when signal changes
  useEffect(() => {
    if (!signal) return;
    const action = signal.action ?? (signal as unknown as Record<string, unknown>).signal as string ?? 'HOLD';
    if (action !== prevSignalRef.current) {
      prevSignalRef.current = action;
      const ts = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
      const conf = signal.confidence != null ? ` · ${(signal.confidence * 100).toFixed(0)}% confidence` : '';
      setNotifications((prev) => [`[${ts}] New signal: ${action}${conf}`, ...prev.slice(0, 19)]);
    }
  }, [signal]);

  // Add notification on connection change
  useEffect(() => {
    if (connectionStatus === 'reconnecting') {
      const ts = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
      setNotifications((prev) => [`[${ts}] Connection lost — reconnecting to ${exchange}`, ...prev.slice(0, 19)]);
    }
    if (connectionStatus === 'connected') {
      const ts = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
      setNotifications((prev) => [`[${ts}] Connected to ${exchange.toUpperCase()}`, ...prev.slice(0, 19)]);
    }
  }, [connectionStatus, exchange]);

  const mainTicker: WatchlistItem | null = ticker
    ? { symbol: ticker.symbol, price: ticker.price, change: ticker.change, changePct: ticker.changePct, volume: ticker.volume }
    : null;

  const allItems: WatchlistItem[] = mainTicker
    ? [mainTicker, ...watchlist.filter((w) => w.symbol !== mainTicker.symbol)]
    : watchlist;

  const filteredItems = searchQuery
    ? allItems.filter((i) => i.symbol.toLowerCase().includes(searchQuery.toLowerCase()))
    : allItems;

  const unreadCount = notifications.length;

  return (
    <header className="h-12 bg-white border-b border-gray-100 flex items-center px-4 gap-4 flex-shrink-0" style={{ boxShadow: '0 1px 4px rgba(0,0,0,0.04)' }}>
      {/* Search bar or ticker strip */}
      {searchOpen ? (
        <div className="flex-1 flex items-center gap-2 min-w-0">
          <input
            autoFocus
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search symbol… (e.g. BTC, SOL)"
            className="flex-1 text-sm border border-gray-200 rounded-lg px-3 py-1 focus:outline-none focus:ring-2 focus:ring-indigo-300"
          />
          <button
            onClick={() => { setSearchOpen(false); setSearchQuery(''); }}
            className="text-xs text-gray-400 hover:text-gray-600 px-2"
          >
            Cancel
          </button>
        </div>
      ) : (
        <div className="flex-1 flex items-center gap-2 overflow-x-auto scrollbar-hide min-w-0 pr-4">
          {allItems.length === 0 ? (
            <span className="text-xs text-gray-400 italic">Connecting to live feed…</span>
          ) : (
            filteredItems.map((item) => <TickerChip key={item.symbol} item={item} />)
          )}
        </div>
      )}

      {/* Right side controls */}
      <div className="flex items-center gap-3 flex-shrink-0">
        {/* Live status badge */}
        {connectionStatus === 'connected' && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#dcfce7] border border-[#bbf7d0]">
            <span className="w-1.5 h-1.5 rounded-full bg-[#15803d] animate-pulse-dot" />
            <span className="text-xs font-semibold text-[#15803d]">LIVE · {exchange.toUpperCase()}</span>
          </div>
        )}
        {connectionStatus === 'reconnecting' && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-50 border border-amber-200">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
            <span className="text-xs font-semibold text-amber-600">RECONNECTING</span>
          </div>
        )}
        {connectionStatus === 'disconnected' && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-50 border border-red-200">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
            <span className="text-xs font-semibold text-red-600">OFFLINE</span>
          </div>
        )}

        {/* Ticking clock */}
        <span className="text-xs text-gray-400 font-mono hidden sm:block">{timeStr}</span>

        {/* Search */}
        <button
          className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-gray-100 transition-colors"
          title="Search symbols"
          onClick={() => setSearchOpen((v) => !v)}
        >
          <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </button>

        {/* Notifications */}
        <div className="relative">
          <button
            className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-gray-100 transition-colors relative"
            title="Notifications"
            onClick={() => setNotifOpen((v) => !v)}
          >
            <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
            {unreadCount > 0 && (
              <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-indigo-500 text-white text-[9px] font-bold rounded-full flex items-center justify-center">
                {unreadCount > 9 ? '9+' : unreadCount}
              </span>
            )}
          </button>
          {notifOpen && <NotifPanel onClose={() => setNotifOpen(false)} notifications={notifications} />}
        </div>

        {/* Avatar */}
        <div
          className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white flex-shrink-0"
          style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}
          title="Nour Maatouk"
        >
          NM
        </div>
      </div>
    </header>
  );
}
