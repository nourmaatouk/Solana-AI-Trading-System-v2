import { usePoll } from '../hooks/usePoll.js'
import { api } from '../api.js'

function BackendStatus({ data, loading, error }) {
  if (loading) return <span className="badge badge-gray"><span className="dot dot-gray" /> Connecting…</span>
  if (error)   return <span className="badge badge-red"><span className="dot dot-red" /> Offline</span>
  const ok = data?.status === 'ok'
  return ok
    ? <span className="badge badge-green"><span className="dot dot-green" /> Live</span>
    : <span className="badge badge-yellow"><span className="dot dot-yellow" /> Degraded</span>
}

export default function Header() {
  const health = usePoll(api.health, 15_000)

  return (
    <header style={{
      background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 30%, #4338ca 65%, #6366f1 100%)',
      color: '#fff',
      position: 'sticky',
      top: 0,
      zIndex: 100,
      boxShadow: '0 4px 24px rgba(67,56,202,.25)',
    }}>
      <div style={{
        maxWidth: 1280,
        margin: '0 auto',
        height: 60,
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <span style={{
            fontSize: 20,
            fontWeight: 800,
            letterSpacing: '-.02em',
            color: '#fff',
          }}>
            Solvanta AI
          </span>
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 5,
            padding: '3px 11px',
            borderRadius: 99,
            fontSize: 11,
            fontWeight: 600,
            background: 'rgba(255,255,255,.15)',
            border: '1px solid rgba(255,255,255,.25)',
            color: '#fff',
          }}>
            📊 Paper Trading Mode
          </span>
        </div>

        {/* Right */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 11, color: 'rgba(255,255,255,.55)' }}>
            No real funds at risk
          </span>
          <BackendStatus {...health} />
        </div>
      </div>

      <div style={{
        maxWidth: 1280,
        margin: '0 auto',
        padding: '0 24px 8px',
        fontSize: 11,
        color: 'rgba(255,255,255,.42)',
        letterSpacing: '.03em',
      }}>
        SOL/USDT · XGBoost + FinBERT + TEN-GNN · Simulation only · Not connected to wallet
      </div>
    </header>
  )
}
