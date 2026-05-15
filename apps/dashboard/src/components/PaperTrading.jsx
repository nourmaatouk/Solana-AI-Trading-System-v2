import { usePoll } from '../hooks/usePoll.js'
import { api } from '../api.js'

function fmt(n, decimals = 4) {
  if (n == null) return '—'
  return Number(n).toFixed(decimals)
}

function pct(n, decimals = 2) {
  if (n == null) return '—'
  return (Number(n) * 100).toFixed(decimals) + '%'
}

export default function PaperTrading() {
  const { data, loading, error } = usePoll(api.paperTrading, 30_000)

  const m = data?.metrics ?? {}
  const equity        = m.equity != null ? m.equity.toFixed(4) : '—'
  const totalReturn   = m.total_return != null ? ((m.total_return) * 100).toFixed(2) + '%' : '—'
  const winRate       = m.win_rate != null ? pct(m.win_rate) : '—'
  const sharpe        = m.sharpe != null ? fmt(m.sharpe, 3) : '—'
  const maxDrawdown   = m.max_drawdown != null ? pct(m.max_drawdown) : '—'
  const nTrades       = m.n_trades ?? 0
  const pos           = data?.open_position
  const returnColor   = m.total_return >= 0 ? '#16a34a' : '#dc2626'

  return (
    <div className="card full-width">
      <div className="card-header">
        <span className="card-title">Paper Trading Simulation</span>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span className="badge badge-purple">Simulation Only</span>
          <span className="badge badge-gray">No real funds</span>
        </div>
      </div>
      <div className="card-body">
        <div className="alert alert-info">
          📊 This is a paper trading simulation. No real SOL or USDT is used. Results are based on historical price data.
        </div>

        {loading && !data && <div className="spinner" />}
        {error && <div className="alert alert-error">Paper trading log unavailable: {error}</div>}

        {data && (
          <>
            <div className="metrics-grid" style={{ marginBottom: 16 }}>
              <div className="metric-box">
                <div className="metric-label">Equity</div>
                <div className="metric-value">{equity}</div>
                <div className="metric-sub">starting at 1.0</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Total Return</div>
                <div className="metric-value" style={{ color: returnColor, fontSize: 18 }}>
                  {totalReturn}
                </div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Win Rate</div>
                <div className="metric-value">{winRate}</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Sharpe Ratio</div>
                <div className="metric-value">{sharpe}</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Max Drawdown</div>
                <div className="metric-value" style={{ color: '#dc2626' }}>{maxDrawdown}</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Trades</div>
                <div className="metric-value">{nTrades}</div>
              </div>
            </div>

            <hr className="sep" />

            <div style={{ marginBottom: 12 }}>
              <div className="metric-label" style={{ marginBottom: 6 }}>Open Position</div>
              {pos ? (
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span className="badge badge-blue">Direction: {pos.direction ?? '—'}</span>
                  <span className="badge badge-gray">Entry: ${fmt(pos.entry_price, 2)}</span>
                  <span className="badge badge-gray">Size: {fmt(pos.size, 4)}</span>
                  <span className="badge badge-gray">TP: ${fmt(pos.take_profit, 2)}</span>
                  <span className="badge badge-red">SL: ${fmt(pos.stop_loss, 2)}</span>
                </div>
              ) : (
                <span className="badge badge-gray">No open position</span>
              )}
            </div>

            {data.last_run_utc && (
              <div className="ts">Last updated: {data.last_run_utc} · Source: {data.signal_source ?? '—'}</div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
