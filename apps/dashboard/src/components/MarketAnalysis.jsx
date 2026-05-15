import { usePoll } from '../hooks/usePoll.js'
import { api } from '../api.js'

function ProbBar({ label, value, className }) {
  const pct = Math.round((value || 0) * 100)
  return (
    <div className="prob-bar">
      <div className="prob-bar-label">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="prob-track">
        <div className={`prob-fill ${className}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function spoofingBadge(conf) {
  if (conf == null) return ['badge-gray', 'N/A']
  if (conf < 0.3)   return ['badge-green', 'Low']
  if (conf < 0.6)   return ['badge-yellow', 'Medium']
  return ['badge-red', 'High']
}

export default function MarketAnalysis() {
  const { data, loading, error } = usePoll(api.signal, 30_000)

  const age = data?._age_minutes
  const stale = age != null && age > 10

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">AI Market Analysis</span>
        <span className="badge badge-blue">Live Signal</span>
      </div>
      <div className="card-body">
        {loading && !data && <div className="spinner" />}

        {error && (
          <div className="alert alert-error">Signal unavailable: {error}</div>
        )}

        {stale && (
          <div className="alert alert-warn">
            ⚠ Signal is {age?.toFixed(1)} min old — run the pipeline to refresh.
          </div>
        )}

        {data && (
          <>
            <div className="metrics-grid" style={{ marginBottom: 14 }}>
              <div className="metric-box">
                <div className="metric-label">Asset</div>
                <div className="metric-value" style={{ fontSize: 16 }}>{data.asset || 'SOL/USDT'}</div>
              </div>
              <div className="metric-box">
                <div className="metric-label">Price</div>
                <div className="metric-value" style={{ fontSize: 18 }}>
                  ${data.latest_close != null ? Number(data.latest_close).toFixed(2) : '—'}
                </div>
              </div>
              <div className="metric-box">
                <div className="metric-label">XGBoost Signal</div>
                <div className="metric-value" style={{ fontSize: 15 }}>
                  {data.signal ?? '—'}
                </div>
              </div>
            </div>

            <ProbBar
              label="Probability UP"
              value={data.classification_probability_up}
              className="prob-up"
            />
            <ProbBar
              label="Probability DOWN"
              value={data.classification_probability_down}
              className="prob-down"
            />

            <hr className="sep" />

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
              <div>
                <div className="metric-label" style={{ marginBottom: 4 }}>Sentiment</div>
                <span className="badge badge-gray">
                  {data.reasoning?.sentiment
                    ? data.reasoning.sentiment.split(',')[0]
                    : 'N/A'}
                </span>
              </div>
              <div>
                <div className="metric-label" style={{ marginBottom: 4 }}>Spoofing Risk</div>
                {(() => {
                  const [cls, label] = spoofingBadge(data.spoofing_confidence)
                  return (
                    <span className={`badge ${cls}`}>
                      {label} ({data.spoofing_confidence != null
                        ? (data.spoofing_confidence * 100).toFixed(0) + '%'
                        : 'N/A'})
                    </span>
                  )
                })()}
              </div>
              <div>
                <div className="metric-label" style={{ marginBottom: 4 }}>Sentiment Relevance</div>
                <span className="badge badge-gray">
                  {data.sentiment_relevance != null
                    ? (data.sentiment_relevance * 100).toFixed(0) + '%'
                    : 'N/A'}
                </span>
              </div>
            </div>

            <div className="ts">
              Signal timestamp: {data.signal_timestamp ?? '—'} ·{' '}
              {age != null ? `${age.toFixed(1)} min ago` : ''}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
