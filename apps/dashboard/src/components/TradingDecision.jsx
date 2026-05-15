import { usePoll } from '../hooks/usePoll.js'
import { api } from '../api.js'

const ACTION_LABELS = { BUY: 'BUY', SELL: 'SELL', HOLD: 'HOLD' }
const ACTION_COLORS = {
  BUY:  { box: 'decision-buy',  text: '#16a34a' },
  SELL: { box: 'decision-sell', text: '#dc2626' },
  HOLD: { box: 'decision-hold', text: '#4338ca' },
}

function riskBadge(level) {
  if (!level) return 'badge-gray'
  if (level === 'LOW') return 'badge-green'
  if (level === 'MEDIUM') return 'badge-yellow'
  return 'badge-red'
}

export default function TradingDecision() {
  const { data, loading, error } = usePoll(api.signal, 30_000)

  const action = data?.action ?? 'HOLD'
  const colors = ACTION_COLORS[action] ?? ACTION_COLORS.HOLD
  const conf   = data?.confidence != null ? (data.confidence * 100).toFixed(1) : '—'
  const score  = data?.fused_score != null ? data.fused_score.toFixed(4) : '—'
  const risk   = data?.risk_level ?? '—'

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Trading Decision</span>
        <span className="badge badge-blue">Paper Trading</span>
      </div>
      <div className="card-body">
        {loading && !data && <div className="spinner" />}
        {error && <div className="alert alert-error">Decision unavailable: {error}</div>}

        {data && (
          <>
            <div className={`decision-box ${colors.box}`}>
              <div className="decision-action">{ACTION_LABELS[action] ?? action}</div>
              <div style={{ fontSize: 13, color: colors.text, fontWeight: 600 }}>
                Confidence: {conf}%
              </div>
            </div>

            <div className="metrics-grid">
              <div className="metric-box">
                <div className="metric-label">Fused Score</div>
                <div className="metric-value" style={{ fontSize: 16 }}>{score}</div>
                <div className="metric-sub">range [−1, +1]</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Risk Level</div>
                <div className="metric-value" style={{ fontSize: 15 }}>
                  <span className={`badge ${riskBadge(risk)}`}>{risk}</span>
                </div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Alignment</div>
                <div className="metric-value" style={{ fontSize: 16 }}>
                  {data.alignment_score != null
                    ? (data.alignment_score * 100).toFixed(0) + '%'
                    : '—'}
                </div>
                <div className="metric-sub">signal agreement</div>
              </div>
            </div>

            {data.reason && (
              <div className="reason-text">
                ℹ {data.reason}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
