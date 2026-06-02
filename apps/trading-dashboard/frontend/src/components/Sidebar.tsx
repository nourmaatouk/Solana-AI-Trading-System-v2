import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { useTradingStore } from '../store';
import logoImg from '../assets/logo.png';

// `mode` gates an item to a single trading mode. Items without `mode` always show.
//   Paper Trading → paper (Explore) only.  Execution → devnet (Connect Wallet) only.
interface NavItem { label: string; path: string; icon: string; mode?: 'paper' | 'devnet'; }

const NAV_ITEMS: NavItem[] = [
  { label: 'Dashboard',        path: '/dashboard', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6' },
  { label: 'Live Market',      path: '/market',  icon: 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6' },
  { label: 'Why This Signal?', path: '/why',     icon: 'M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' },
  { label: 'Sentiment & News', path: '/sentiment', icon: 'M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z' },
  { label: 'Alerts',           path: '/alerts',  icon: 'M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9' },
  { label: 'Paper Trading',    path: '/trade',   mode: 'paper',  icon: 'M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z' },
  { label: 'Backtesting',      path: '/backtest',icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
  { label: 'Execution',        path: '/execution', mode: 'devnet', icon: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z' },
  { label: 'System Status',    path: '/status',  icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z' },
  { label: 'Settings',         path: '/settings',icon: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z' },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const mode = useTradingStore((s) => s.mode);
  const items = NAV_ITEMS.filter((item) => !item.mode || item.mode === mode);

  return (
    <aside
      className="flex flex-col h-full transition-all duration-200"
      style={{ width: collapsed ? 56 : 200, background: 'var(--bg-panel)', borderRight: '1px solid var(--border)', flexShrink: 0 }}
    >
      {/* Logo */}
      <div className={`flex items-center gap-2 ${collapsed ? 'justify-center px-0 py-4' : 'px-4 py-4'}`}
        style={{ borderBottom: '1px solid var(--border)' }}>
        <img src={logoImg} alt="Solvanta AI" className="flex-shrink-0" style={{ width: 36, height: 36, objectFit: 'contain' }} />
        {!collapsed && <span className="font-bold text-sm tracking-tight" style={{ color: 'var(--text-1)' }}>Solvanta AI</span>}
      </div>

      {/* Nav items */}
      <nav className="flex-1 py-4 px-2 overflow-y-auto">
        <div className="space-y-0.5">
          {items.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === '/'}
              title={collapsed ? item.label : undefined}
              className={`flex items-center gap-2.5 rounded-lg text-xs font-medium transition-colors duration-150 ${
                collapsed ? 'justify-center px-0 py-2.5' : 'px-3 py-2'
              }`}
              style={({ isActive }) =>
                isActive
                  ? { backgroundColor: 'var(--accent)', color: 'var(--bg-page)' }
                  : { color: 'var(--text-2)' }
              }
              onMouseEnter={(e) => {
                if (!(e.currentTarget as HTMLAnchorElement).classList.contains('active')) {
                  (e.currentTarget as HTMLElement).style.background = 'var(--border)';
                }
              }}
              onMouseLeave={(e) => {
                if (!(e.currentTarget as HTMLAnchorElement).classList.contains('active')) {
                  (e.currentTarget as HTMLElement).style.background = 'transparent';
                }
              }}
            >
              {({ isActive }) => (
                <>
                  <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth={isActive ? 2.5 : 2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d={item.icon} />
                  </svg>
                  {!collapsed && <span className="truncate">{item.label}</span>}
                </>
              )}
            </NavLink>
          ))}
        </div>
      </nav>

      {/* Collapse toggle */}
      <div className="px-2 py-4" style={{ borderTop: '1px solid var(--border)' }}>
        <button
          onClick={() => setCollapsed((v) => !v)}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className={`flex items-center gap-2 rounded-lg text-xs transition-colors cursor-pointer w-full ${
            collapsed ? 'justify-center px-0 py-2.5' : 'px-3 py-2'
          }`}
          style={{ color: 'var(--text-2)' }}
          onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--border)')}
          onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
        >
          <svg className={`w-4 h-4 transition-transform duration-200 ${collapsed ? 'rotate-180' : ''}`}
            fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}
