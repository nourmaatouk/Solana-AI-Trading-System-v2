import { BrowserRouter, Routes, Route, Outlet, Navigate, useLocation } from 'react-router-dom';
import { useEffect } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { useMarketSocket } from './hooks/useMarketSocket';
import { useSystemWallet } from './hooks/useSystemWallet';
import { useTradingStore } from './store';
import { enableAutoExecute, disableAutoExecute } from './services/executionApi';
import { loadSettings, SETTINGS_KEY } from './hooks/useSettings';
import { LandingPage } from './pages/LandingPage';
import { Dashboard } from './pages/Dashboard';
import { DevnetDashboard } from './pages/DevnetDashboard';
import { LiveMarket } from './pages/LiveMarket';
import { WhySignal } from './pages/WhySignal';
import { SentimentNews } from './pages/SentimentNews';
import { PaperTrading } from './pages/PaperTrading';
import { SystemStatus } from './pages/SystemStatus';
import { SettingsPage } from './pages/SettingsPage';
import { BacktestPage } from './pages/BacktestPage';
import { Correlation } from './pages/Correlation';
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
  const mode = useTradingStore((s) => s.mode);
  const setMode = useTradingStore((s) => s.setMode);
  const { connected: phantomConnected } = useWallet();
  const { connected: systemConnected } = useSystemWallet();
  // Devnet is accessible when EITHER the Phantom browser wallet OR the backend
  // system wallet (Dev Wallet button) is connected.
  const canAccessDevnet = phantomConnected || systemConnected;

  // Keep data-theme in sync with store (also applied on toggle; this covers initial load)
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Any wallet connected → upgrade to devnet automatically.
  useEffect(() => {
    if (canAccessDevnet && mode === 'paper') setMode('devnet');
  }, [canAccessDevnet, mode, setMode]);

  // All wallets disconnected → fall back to paper immediately.
  // Also fires on initial load when localStorage held a stale 'devnet' value.
  useEffect(() => {
    if (!canAccessDevnet && mode === 'devnet') setMode('paper');
  }, [canAccessDevnet, mode, setMode]);

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
  const mode = useTradingStore((s) => s.mode);

  // Auto-enable execution when entering devnet, auto-disable when leaving.
  useEffect(() => {
    if (mode === 'devnet') {
      // 1. Tell the backend AutoExecutor to start firing on BUY/SELL signals.
      enableAutoExecute().catch(() => {});
      // 2. Sync frontend Settings so ExecutionPanel + useAutoExecution hook
      //    both see tradingMode = 'Auto' without needing a manual Settings save.
      const current = loadSettings();
      if (current.tradingMode !== 'Auto') {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...current, tradingMode: 'Auto' }));
        window.dispatchEvent(new Event('solvanta:settings'));
      }
    } else {
      // Switching back to paper — disable the backend AutoExecutor so it
      // doesn't keep trying to execute devnet trades in the background.
      disableAutoExecute().catch(() => {});
      const current = loadSettings();
      if (current.tradingMode === 'Auto') {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...current, tradingMode: 'Manual' }));
        window.dispatchEvent(new Event('solvanta:settings'));
      }
    }
  }, [mode]);

  return (
    <BrowserRouter>
      <Routes>
        {/* Public landing page — full screen, no app chrome */}
        <Route path="/" element={<LandingPage />} />

        {/* Dashboard app — wrapped in the sidebar/top-bar layout */}
        <Route element={<AppLayout />}>
          <Route path="/dashboard" element={mode === 'devnet' ? <DevnetDashboard /> : <Dashboard />} />
          <Route path="/market" element={<LiveMarket />} />
          <Route path="/why" element={<WhySignal />} />
          <Route path="/sentiment" element={<SentimentNews />} />
          <Route path="/trade" element={mode === 'devnet' ? <Navigate to="/execution" replace /> : <PaperTrading />} />
          <Route path="/status" element={<SystemStatus />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/backtest" element={<BacktestPage />} />
          <Route path="/correlation" element={<Correlation />} />
          <Route path="/execution" element={mode === 'paper' ? <Navigate to="/trade" replace /> : <ExecutionPanel />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
