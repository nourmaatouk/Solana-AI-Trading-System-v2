// In dev (npm run dev) Vite proxies /health, /signal, etc. to localhost:8000.
// In production (served from FastAPI at /app/) the base is empty — same origin.
// Override with VITE_API_BASE_URL if the backend lives on a different host.
const BASE = import.meta.env.VITE_API_BASE_URL ?? ''

async function fetchJSON(endpoint, options) {
  const res = await fetch(`${BASE}${endpoint}`, { signal: options?.signal })
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`)
  return res.json()
}

export const api = {
  health:       (opts) => fetchJSON('/health', opts),
  signal:       (opts) => fetchJSON('/signal', opts),
  status:       (opts) => fetchJSON('/status', opts),
  paperTrading: (opts) => fetchJSON('/paper-trading', opts),
  preferences:  (opts) => fetchJSON('/preferences', opts),
  system:       (opts) => fetchJSON('/system', opts),
  scheduler:    (opts) => fetchJSON('/scheduler', opts),
  history:      (opts) => fetchJSON('/history?last=100', opts),
  savePreferences: (body) =>
    fetch(`${BASE}/preferences`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) => r.json()),
  triggerRun: () =>
    fetch(`${BASE}/scheduler/run`, { method: 'POST' }).then((r) => r.json()),
}
