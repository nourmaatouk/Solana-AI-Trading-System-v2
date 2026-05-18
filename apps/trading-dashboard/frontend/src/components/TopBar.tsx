import { useEffect, useState, useRef } from 'react';
import { useTradingStore } from '../store';
import type { WatchlistItem } from '../types';

function TickerChip({ item }: { item: WatchlistItem }) {
  const pct = parseFloat(item.changePct);
  const isPos = pct >= 0;
  return (
    <div className="flex items-center gap-1.5 px-3 py-1 rounded-full flex-shrink-0"
      style={{ background: 'var(--bg-sub)', border: '1px solid var(--border)' }}>
      <span className="text-xs font-semibold" style={{ color: 'var(--text-1)' }}>{item.symbol.replace('USDT', '')}</span>
      <span className="text-xs font-mono" style={{ color: 'var(--text-1)' }}>${parseFloat(item.price).toLocaleString('en-US', { maximumFractionDigits: 4 })}</span>
      <span className="text-xs font-medium" style={{ color: isPos ? 'var(--green)' : 'var(--red)' }}>
        {isPos ? '+' : ''}{pct.toFixed(2)}%
      </span>
    </div>
  );
}

function ThemeToggle() {
  const theme = useTradingStore((s) => s.theme);
  const toggleTheme = useTradingStore((s) => s.toggleTheme);
  const isDark = theme === 'dark';

  return (
    <button
      onClick={toggleTheme}
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="w-8 h-8 flex items-center justify-center rounded-lg transition-colors"
      style={{ color: 'var(--text-2)' }}
      onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--border)')}
      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
    >
      {isDark ? (
        /* Sun icon — switch to light */
        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="5" />
          <path strokeLinecap="round" d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
        </svg>
      ) : (
        /* Moon icon — switch to dark */
        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
        </svg>
      )}
    </button>
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
    <div ref={ref} className="absolute right-0 top-10 w-72 rounded-xl shadow-lg z-50"
      style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
      <div className="px-4 py-3 flex items-center justify-between" style={{ borderBottom: '1px solid var(--border)' }}>
        <span className="text-sm font-semibold" style={{ color: 'var(--text-1)' }}>Notifications</span>
        <span className="text-xs" style={{ color: 'var(--text-3)' }}>{notifications.length} recent</span>
      </div>
      <div className="max-h-64 overflow-y-auto">
        {notifications.length === 0 ? (
          <div className="px-4 py-6 text-center text-sm" style={{ color: 'var(--text-3)' }}>No notifications yet</div>
        ) : (
          notifications.map((n, i) => (
            <div key={i} className="px-4 py-2.5 text-xs last:border-0"
              style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-2)' }}>
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

  useEffect(() => {
    const tick = () => setTimeStr(new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

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

  return (
    <header className="h-12 flex items-center px-4 gap-4 flex-shrink-0"
      style={{ background: 'var(--bg-panel)', borderBottom: '1px solid var(--border)' }}>

      {searchOpen ? (
        <div className="flex-1 flex items-center gap-2 min-w-0">
          <input
            autoFocus
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search symbol… (e.g. BTC, SOL)"
            className="flex-1 text-sm rounded-lg px-3 py-1 focus:outline-none"
            style={{ background: 'var(--bg-sub)', border: '1px solid var(--border)', color: 'var(--text-1)' }}
          />
          <button onClick={() => { setSearchOpen(false); setSearchQuery(''); }}
            className="text-xs px-2" style={{ color: 'var(--text-2)' }}>
            Cancel
          </button>
        </div>
      ) : (
        <div className="flex-1 flex items-center gap-2 overflow-x-auto scrollbar-hide min-w-0 pr-4">
          {allItems.length === 0 ? (
            <span className="text-xs italic" style={{ color: 'var(--text-3)' }}>Connecting to live feed…</span>
          ) : (
            filteredItems.map((item) => <TickerChip key={item.symbol} item={item} />)
          )}
        </div>
      )}

      <div className="flex items-center gap-3 flex-shrink-0">
        {connectionStatus === 'connected' && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full"
            style={{ background: 'var(--green-bg)', border: '1px solid var(--green-bd)' }}>
            <span className="w-1.5 h-1.5 rounded-full animate-pulse-dot" style={{ background: 'var(--green)' }} />
            <span className="text-xs font-semibold" style={{ color: 'var(--green)' }}>LIVE · {exchange.toUpperCase()}</span>
          </div>
        )}
        {connectionStatus === 'reconnecting' && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full"
            style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)' }}>
            <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: 'var(--accent)' }} />
            <span className="text-xs font-semibold" style={{ color: 'var(--accent)' }}>RECONNECTING</span>
          </div>
        )}
        {connectionStatus === 'disconnected' && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full"
            style={{ background: 'var(--red-bg)', border: '1px solid var(--red-bd)' }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--red)' }} />
            <span className="text-xs font-semibold" style={{ color: 'var(--red)' }}>OFFLINE</span>
          </div>
        )}

        <span className="text-xs font-mono hidden sm:block" style={{ color: 'var(--text-3)' }}>{timeStr}</span>

        {/* Theme toggle */}
        <ThemeToggle />

        {/* Search */}
        <button
          className="w-8 h-8 flex items-center justify-center rounded-lg transition-colors"
          title="Search symbols"
          onClick={() => setSearchOpen((v) => !v)}
          style={{ color: 'var(--text-2)' }}
          onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--border)')}
          onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </button>

        {/* Notifications */}
        <div className="relative">
          <button
            className="w-8 h-8 flex items-center justify-center rounded-lg transition-colors relative"
            title="Notifications"
            onClick={() => setNotifOpen((v) => !v)}
            style={{ color: 'var(--text-2)' }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--border)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
            {notifications.length > 0 && (
              <span className="absolute -top-0.5 -right-0.5 w-4 h-4 text-[9px] font-bold rounded-full flex items-center justify-center"
                style={{ background: 'var(--accent)', color: 'var(--bg-page)' }}>
                {notifications.length > 9 ? '9+' : notifications.length}
              </span>
            )}
          </button>
          {notifOpen && <NotifPanel onClose={() => setNotifOpen(false)} notifications={notifications} />}
        </div>

        {/* Avatar */}
        <div
          className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
          style={{ background: 'var(--accent)', color: 'var(--bg-page)' }}
          title="Nour Maatouk"
        >
          NM
        </div>
      </div>
    </header>
  );
}
