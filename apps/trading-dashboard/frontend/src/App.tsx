import { BrowserRouter, Routes, Route, Outlet, useLocation } from 'react-router-dom';
import { useEffect } from 'react';
import { useMarketSocket } from './hooks/useMarketSocket';
import { useTradingStore } from './store';
import { LandingPage } from './pages/LandingPage';
import { Dashboard } from './pages/Dashboard';
import { LiveMarket } from './pages/LiveMarket';
import { WhySignal } from './pages/WhySignal';
import { AlertsPage } from './pages/AlertsPage';
import { TradePage } from './pages/TradePage';
import { HowItWorks } from './pages/HowItWorks';
import { SystemStatus } from './pages/SystemStatus';
import { SettingsPage } from './pages/SettingsPage';
import { BacktestPage } from './pages/BacktestPage';
import { ExecutionPanel } from './pages/ExecutionPanel';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { ErrorBanner } from './components/ErrorBanner';

/**
 * Layout for the authenticated/app surface: sidebar + top bar chrome wrapping
 * the active page (rendered via <Outlet/>). The public landing page at `/`
 * lives outside this layout so it renders full-screen.
 */
function AppLayout() {
  useMarketSocket();
  const { pathname } = useLocation();
  const isMarket = pathname === '/market';
  const theme = useTradingStore((s) => s.theme);

  // Keep data-theme in sync with store (also applied on toggle; this covers initial load)
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg-page)' }}>
      <Sidebar />

      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        {!isMarket && <TopBar />}
        {!isMarket && <ErrorBanner />}
        <main className="flex-1 overflow-hidden flex flex-col min-h-0">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public landing page — full screen, no app chrome */}
        <Route path="/" element={<LandingPage />} />

        {/* Dashboard app — wrapped in the sidebar/top-bar layout */}
        <Route element={<AppLayout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/market" element={<LiveMarket />} />
          <Route path="/why" element={<WhySignal />} />
          <Route path="/alerts" element={<AlertsPage />} />
          <Route path="/trade" element={<TradePage />} />
          <Route path="/how" element={<HowItWorks />} />
          <Route path="/status" element={<SystemStatus />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/backtest" element={<BacktestPage />} />
          <Route path="/execution" element={<ExecutionPanel />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
