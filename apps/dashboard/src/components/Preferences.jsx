import { useState, useEffect } from 'react'
import { api } from '../api.js'

export default function Preferences() {
  const [prefs, setPrefs]     = useState(null)
  const [saving, setSaving]   = useState(false)
  const [saved, setSaved]     = useState(false)
  const [error, setError]     = useState(null)

  useEffect(() => {
    api.preferences().then(setPrefs).catch((e) => setError(e.message))
  }, [])

  function handleChange(key, value) {
    setPrefs((p) => ({ ...p, [key]: value }))
    setSaved(false)
  }

  async function handleSave() {
    setSaving(true)
    setError(null)
    try {
      const result = await api.savePreferences(prefs)
      setPrefs(result)
      setSaved(true)
      setTimeout(() => setSaved(false), 2500)
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Preferences</span>
        {saved && <span className="badge badge-green">✓ Saved</span>}
      </div>
      <div className="card-body">
        {error && <div className="alert alert-error">{error}</div>}
        {!prefs && !error && <div className="spinner" />}

        {prefs && (
          <>
            <div className="form-row">
              <label className="form-label">Asset</label>
              <input
                className="form-input"
                value={prefs.asset ?? ''}
                onChange={(e) => handleChange('asset', e.target.value)}
              />
            </div>

            <div className="form-row">
              <label className="form-label">Risk Level</label>
              <select
                className="form-select"
                value={prefs.risk_level ?? 'medium'}
                onChange={(e) => handleChange('risk_level', e.target.value)}
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>

            <div className="form-row">
              <label className="form-label">Trading Mode</label>
              <select
                className="form-select"
                value={prefs.trading_mode ?? 'paper'}
                onChange={(e) => handleChange('trading_mode', e.target.value)}
              >
                <option value="paper">Paper (Simulation)</option>
                <option value="live" disabled>Live (Not available)</option>
              </select>
            </div>

            <div className="form-row">
              <label className="form-label">
                Confidence Threshold
                <span style={{ color: '#94a3b8', fontWeight: 400, marginLeft: 6 }}>
                  ({((prefs.confidence_threshold ?? 0.69) * 100).toFixed(0)}%)
                </span>
              </label>
              <input
                className="form-input"
                type="range"
                min="0.5" max="0.95" step="0.01"
                value={prefs.confidence_threshold ?? 0.69}
                onChange={(e) => handleChange('confidence_threshold', parseFloat(e.target.value))}
                style={{ padding: '4px 0' }}
              />
            </div>

            <div className="form-row">
              <label className="form-label">Max Position Size</label>
              <input
                className="form-input"
                type="number"
                min="0.1" max="1" step="0.1"
                value={prefs.max_position_size ?? 1.0}
                onChange={(e) => handleChange('max_position_size', parseFloat(e.target.value))}
              />
            </div>

            <button
              className="btn btn-primary"
              onClick={handleSave}
              disabled={saving}
              style={{ width: '100%', justifyContent: 'center', marginTop: 4 }}
            >
              {saving ? 'Saving…' : 'Save Preferences'}
            </button>
          </>
        )}
      </div>
    </div>
  )
}
