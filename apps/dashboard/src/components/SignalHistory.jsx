import { usePoll } from '../hooks/usePoll.js'
import { api } from '../api.js'

function actionBadge(a) {
  if (a === 'BUY')  return 'badge-green'
  if (a === 'SELL') return 'badge-red'
  return 'badge-gray'
}

function riskBadge(r) {
  if (r === 'LOW')    return 'badge-green'
  if (r === 'MEDIUM') return 'badge-yellow'
  if (r === 'HIGH')   return 'badge-red'
  return 'badge-gray'
}

function fmt(v, d = 4) {
  return v != null ? Number(v).toFixed(d) : '—'
}

function pct(v) {
  return v != null ? `${(Number(v) * 100).toFixed(2)}%` : '—'
}

export default function SignalHistory() {
  const { data: history, loading, error } = usePoll(api.history, 30_000)

  const entries = Array.isArray(history) ? [...history].reverse().slice(0, 20) : []

  return (
    <div className="card full-width">
      <div className="card-header">
        <span className="card-title">Signal History</span>
        <span className="badge badge-gray">Last 20 runs</span>
      </div>
      <div className="card-body" style={{ padding: 0 }}>
        {loading && !history && <div className="spinner" style={{ margin: 20 }} />}
        {error && <div className="alert alert-error" style={{ margin: 16 }}>History unavailable: {error}</div>}

        {!loading && entries.length === 0 && (
          <div className="alert alert-info" style={{ margin: 16 }}>
            No history yet. History is recorded after each pipeline run.
          </div>
        )}

        {entries.length > 0 && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: 12,
            }}>
              <thead>
                <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                  {['Time', 'Signal', 'Action', 'Score', 'Risk', 'Equity', 'Return', 'Price'].map(h => (
                    <th key={h} style={{
                      padding: '8px 14px',
                      textAlign: 'left',
                      fontWeight: 600,
                      color: '#64748b',
                      whiteSpace: 'nowrap',
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {entries.map((e, i) => {
                  const ret = e.total_return
                  const retColor = ret != null ? (ret >= 0 ? '#16a34a' : '#dc2626') : '#94a3b8'
                  return (
                    <tr key={i} style={{
                      borderBottom: '1px solid #f1f5f9',
                      background: i % 2 === 0 ? '#fff' : '#fafafa',
                    }}>
                      <td style={{ padding: '7px 14px', color: '#94a3b8', whiteSpace: 'nowrap' }}>
                        {e.run_utc
                          ? new Date(e.run_utc).toLocaleString([], {
                              month: '2-digit', day: '2-digit',
                              hour: '2-digit', minute: '2-digit',
                            })
                          : '—'}
                      </td>
                      <td style={{ padding: '7px 14px' }}>
                        <span className="badge badge-gray" style={{ fontSize: 10 }}>
                          {e.signal ?? '—'}
                        </span>
                      </td>
                      <td style={{ padding: '7px 14px' }}>
                        <span className={`badge ${actionBadge(e.action)}`} style={{ fontSize: 10 }}>
                          {e.action ?? '—'}
                        </span>
                      </td>
                      <td style={{ padding: '7px 14px', fontFamily: 'monospace', color: '#334155' }}>
                        {e.fused_score != null ? Number(e.fused_score).toFixed(4) : '—'}
                      </td>
                      <td style={{ padding: '7px 14px' }}>
                        <span className={`badge ${riskBadge(e.risk_level)}`} style={{ fontSize: 10 }}>
                          {e.risk_level ?? '—'}
                        </span>
                      </td>
                      <td style={{ padding: '7px 14px', fontFamily: 'monospace' }}>
                        {fmt(e.equity)}
                      </td>
                      <td style={{ padding: '7px 14px', fontFamily: 'monospace', color: retColor, fontWeight: 600 }}>
                        {pct(e.total_return)}
                      </td>
                      <td style={{ padding: '7px 14px', color: '#475569' }}>
                        {e.latest_close != null ? `$${Number(e.latest_close).toFixed(2)}` : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
