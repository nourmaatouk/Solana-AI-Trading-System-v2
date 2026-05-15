import { useState } from 'react'
import { usePoll } from '../hooks/usePoll.js'
import { api } from '../api.js'

function statusClass(s) {
  if (!s || s === 'missing' || s === 'not_started') return 'dot-gray'
  if (s === 'ok' || s === 'running' || s === 'idle') return 'dot-green'
  if (s === 'warning' || s === 'unknown') return 'dot-yellow'
  return 'dot-red'
}

function statusBadge(s) {
  if (!s || s === 'missing') return ['badge-gray', 'Missing']
  if (s === 'not_started')   return ['badge-gray', 'Not started']
  if (s === 'ok' || s === 'running') return ['badge-green', s === 'running' ? 'Running' : 'OK']
  if (s === 'idle')          return ['badge-green', 'Idle (scheduled)']
  if (s === 'warning' || s === 'unknown') return ['badge-yellow', 'Warning']
  if (s === 'stopped')       return ['badge-gray', 'Stopped']
  return ['badge-red', 'Error']
}

function HealthRow({ label, status, age, extra }) {
  const [cls, text] = statusBadge(status)
  return (
    <div className="health-row">
      <div>
        <div className="health-label">{label}</div>
        {extra && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{extra}</div>}
      </div>
      <div className="health-right">
        {age != null && <span className="health-age">{age.toFixed(1)} min ago</span>}
        <span className={`badge ${cls}`}>
          <span className={`dot ${statusClass(status)}`} />
          {text}
        </span>
      </div>
    </div>
  )
}

export default function SystemHealth() {
  const { data, loading, error } = usePoll(api.system, 20_000)
  const { data: status }         = usePoll(api.status, 30_000)
  const [triggering, setTriggering] = useState(false)
  const [triggerMsg, setTriggerMsg] = useState(null)

  const pipelineStatus = status?.status ?? null
  const pipelineAge    = status?.finished_at_utc
    ? (Date.now() - new Date(status.finished_at_utc)) / 60000
    : null

  // Scheduler extras
  const sched = data?.scheduler ?? {}
  const schedExtra = sched.next_run_utc
    ? `Next run: ${new Date(sched.next_run_utc).toLocaleTimeString()} · ${sched.n_runs ?? 0} runs`
    : sched.interval_seconds
      ? `Interval: ${sched.interval_seconds / 60} min`
      : null

  async function handleTrigger() {
    setTriggering(true)
    setTriggerMsg(null)
    try {
      const res = await fetch('/scheduler/run', { method: 'POST' })
      const body = await res.json()
      setTriggerMsg(body.message ?? 'Pipeline triggered.')
    } catch {
      setTriggerMsg('Could not trigger pipeline — is the API running?')
    } finally {
      setTriggering(false)
      setTimeout(() => setTriggerMsg(null), 8000)
    }
  }

  return (
    <div className="card full-width">
      <div className="card-header">
        <span className="card-title">System Health</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {loading && <span className="badge badge-gray">Refreshing…</span>}
          {error   && <span className="badge badge-red">API error</span>}
          <button
            className="btn btn-primary btn-sm"
            onClick={handleTrigger}
            disabled={triggering}
          >
            {triggering ? '⟳ Running…' : '▶ Refresh Signal'}
          </button>
        </div>
      </div>
      <div className="card-body">
        {error && (
          <div className="alert alert-error">
            Cannot reach backend: {error}. Make sure the API server is running on port 8000.
          </div>
        )}
        {triggerMsg && (
          <div className="alert alert-info">{triggerMsg}</div>
        )}

        <HealthRow
          label="Scheduler"
          status={sched.status ?? 'not_started'}
          age={data?.scheduler?.age_minutes}
          extra={schedExtra}
        />
        <HealthRow
          label="Pipeline (last run)"
          status={pipelineStatus}
          age={pipelineAge}
        />
        <HealthRow
          label="Live Signal"
          status={data?.fused_decision?.status ?? data?.signal?.status}
          age={data?.fused_decision?.age_minutes ?? data?.signal?.age_minutes}
        />
        <HealthRow
          label="Orderbook Collector"
          status={data?.orderbook?.status}
          age={data?.orderbook?.age_minutes}
        />
        <HealthRow
          label="Sentiment Daemon"
          status={data?.sentiment?.status}
          age={data?.sentiment?.age_minutes}
        />
        <HealthRow
          label="Paper Trading"
          status={data?.paper_trading?.status === 'unknown' ? 'ok' : data?.paper_trading?.status}
          age={data?.paper_trading?.age_minutes}
        />
      </div>
    </div>
  )
}
