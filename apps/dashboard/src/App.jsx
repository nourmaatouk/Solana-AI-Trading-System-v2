import Header          from './components/Header.jsx'
import SystemHealth    from './components/SystemHealth.jsx'
import MarketAnalysis  from './components/MarketAnalysis.jsx'
import TradingDecision from './components/TradingDecision.jsx'
import PaperTrading    from './components/PaperTrading.jsx'
import Preferences     from './components/Preferences.jsx'
import EquityChart     from './components/EquityChart.jsx'
import SignalHistory   from './components/SignalHistory.jsx'
export default function App() {
  return (
    <div className="app">
      <Header />

      <main className="main-content">
        <SystemHealth />
        <MarketAnalysis />
        <TradingDecision />
        <PaperTrading />
        <Preferences />
        <EquityChart />
        <SignalHistory />
      </main>

      <footer style={{
        textAlign: 'center',
        padding: '12px 0 24px',
        fontSize: 11,
        color: '#94a3b8',
      }}>
        Solvanta AI · Paper Trading Simulation · Not financial advice · No real funds at risk
      </footer>
    </div>
  )
}
