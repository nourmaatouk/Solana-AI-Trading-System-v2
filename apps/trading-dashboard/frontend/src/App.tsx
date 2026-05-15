import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { useMarketSocket } from './hooks/useMarketSocket';
import { Dashboard } from './pages/Dashboard';
import { LiveMarket } from './pages/LiveMarket';
import { WhySignal } from './pages/WhySignal';
import { AlertsPage } from './pages/AlertsPage';
import { TradePage } from './pages/TradePage';
import { HowItWorks } from './pages/HowItWorks';
import { SystemStatus } from './pages/SystemStatus';
import { SettingsPage } from './pages/SettingsPage';
import { BacktestPage } from './pages/BacktestPage';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { ErrorBanner } from './components/ErrorBanner';

function AppLayout() {
  useMarketSocket();
  const { pathname } = useLocation();
  const isMarket = pathname === '/market';

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: '#f8f9fd' }}>
      <Sidebar />

      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        {!isMarket && <TopBar />}
        {!isMarket && <ErrorBanner />}
        <main className="flex-1 overflow-hidden flex flex-col min-h-0">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/market" element={<LiveMarket />} />
            <Route path="/why" element={<WhySignal />} />
            <Route path="/alerts" element={<AlertsPage />} />
            <Route path="/trade" element={<TradePage />} />
            <Route path="/how" element={<HowItWorks />} />
            <Route path="/status" element={<SystemStatus />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/backtest" element={<BacktestPage />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}

export default App;
