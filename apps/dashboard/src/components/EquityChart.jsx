import { usePoll } from '../hooks/usePoll.js'
import { api } from '../api.js'

const W = 600
const H = 180
const PAD = { top: 16, right: 16, bottom: 32, left: 52 }
const IW = W - PAD.left - PAD.right
const IH = H - PAD.top  - PAD.bottom

function scaleX(i, n)     { return PAD.left + (i / Math.max(n - 1, 1)) * IW }
function scaleY(v, lo, hi) {
  const range = hi - lo || 0.001
  return PAD.top + IH - ((v - lo) / range) * IH
}

function yTicks(lo, hi, n = 4) {
  const step = (hi - lo) / n
  return Array.from({ length: n + 1 }, (_, i) => lo + i * step)
}

function actionColor(a) {
  if (a === 'BUY')  return '#22c55e'
  if (a === 'SELL') return '#ef4444'
  return '#94a3b8'
}

export default function EquityChart() {
  const { data: history, loading, error } = usePoll(api.history, 30_000)

  const entries = Array.isArray(history) ? history.filter(e => e.equity != null) : []
  const hasData = entries.length >= 2

  // Y domain with a bit of padding
  const equities = entries.map(e => e.equity)
  const rawLo    = Math.min(...equities, 1.0)
  const rawHi    = Math.max(...equities, 1.0)
  const pad      = Math.max((rawHi - rawLo) * 0.15, 0.005)
  const lo       = rawLo - pad
  const hi       = rawHi + pad

  const points = entries.map((e, i) =>
    `${scaleX(i, entries.length).toFixed(1)},${scaleY(e.equity, lo, hi).toFixed(1)}`
  ).join(' ')

  const latest  = entries[entries.length - 1]
  const ret     = latest?.total_return
  const retPct  = ret != null ? (ret * 100).toFixed(2) : null
  const retColor = ret >= 0 ? '#16a34a' : '#dc2626'

  const ticks = yTicks(lo, hi)

  return (
    <div className="card full-width">
      <div className="card-header">
        <span className="card-title">Equity Curve</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {retPct != null && (
            <span className={`badge ${ret >= 0 ? 'badge-green' : 'badge-red'}`}
                  style={{ color: retColor }}>
              {ret >= 0 ? '+' : ''}{retPct}% total return
            </span>
          )}
          <span className="badge badge-gray">{entries.length} data points</span>
        </div>
      </div>
      <div className="card-body">
        {loading && !history && <div className="spinner" />}
        {error && <div className="alert alert-error">History unavailable: {error}</div>}

        {!hasData && !loading && (
          <div className="alert alert-info">
            Equity curve appears after 2+ pipeline runs. The scheduler runs every 15 minutes.
          </div>
        )}

        {hasData && (
          <svg
            viewBox={`0 0 ${W} ${H}`}
            style={{ width: '100%', height: 'auto', overflow: 'visible' }}
          >
            {/* Grid lines */}
            {ticks.map((t, i) => {
              const y = scaleY(t, lo, hi)
              return (
                <g key={i}>
                  <line
                    x1={PAD.left} y1={y} x2={W - PAD.right} y2={y}
                    stroke="#e2e8f0" strokeWidth="1"
                  />
                  <text
                    x={PAD.left - 6} y={y + 4}
                    textAnchor="end" fontSize="10" fill="#94a3b8"
                  >
                    {t.toFixed(3)}
                  </text>
                </g>
              )
            })}

            {/* Baseline at 1.0 */}
            {lo < 1.0 && hi > 1.0 && (
              <line
                x1={PAD.left} y1={scaleY(1.0, lo, hi)}
                x2={W - PAD.right} y2={scaleY(1.0, lo, hi)}
                stroke="#94a3b8" strokeWidth="1" strokeDasharray="4 3"
              />
            )}

            {/* Area fill */}
            <defs>
              <linearGradient id="eq-grad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#3b82f6" stopOpacity="0.18" />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.01" />
              </linearGradient>
            </defs>
            <polygon
              points={[
                `${PAD.left},${scaleY(lo, lo, hi)}`,
                ...entries.map((e, i) =>
                  `${scaleX(i, entries.length).toFixed(1)},${scaleY(e.equity, lo, hi).toFixed(1)}`
                ),
                `${scaleX(entries.length - 1, entries.length).toFixed(1)},${scaleY(lo, lo, hi)}`,
              ].join(' ')}
              fill="url(#eq-grad)"
            />

            {/* Equity line */}
            <polyline
              points={points}
              fill="none"
              stroke="#2563eb"
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
            />

            {/* Action dots */}
            {entries.map((e, i) => (
              <circle
                key={i}
                cx={scaleX(i, entries.length)}
                cy={scaleY(e.equity, lo, hi)}
                r="3.5"
                fill={actionColor(e.action)}
                stroke="#fff"
                strokeWidth="1.5"
              >
                <title>{new Date(e.run_utc).toLocaleString()} | {e.action} | equity {e.equity?.toFixed(4)}</title>
              </circle>
            ))}

            {/* X-axis labels (first and last) */}
            {entries.length > 0 && (
              <>
                <text
                  x={PAD.left} y={H - 6}
                  fontSize="10" fill="#94a3b8" textAnchor="middle"
                >
                  {new Date(entries[0].run_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </text>
                <text
                  x={W - PAD.right} y={H - 6}
                  fontSize="10" fill="#94a3b8" textAnchor="middle"
                >
                  {new Date(entries[entries.length - 1].run_utc).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </text>
              </>
            )}
          </svg>
        )}

        {/* Legend */}
        <div style={{ display: 'flex', gap: 14, marginTop: 8, fontSize: 11, color: '#94a3b8' }}>
          {[['BUY', '#22c55e'], ['SELL', '#ef4444'], ['HOLD', '#94a3b8']].map(([label, color]) => (
            <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <svg width="10" height="10">
                <circle cx="5" cy="5" r="4" fill={color} />
              </svg>
              {label}
            </span>
          ))}
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <svg width="20" height="10">
              <line x1="0" y1="5" x2="20" y2="5" stroke="#94a3b8" strokeWidth="1" strokeDasharray="4 3" />
            </svg>
            Baseline (1.0)
          </span>
        </div>
      </div>
    </div>
  )
}
