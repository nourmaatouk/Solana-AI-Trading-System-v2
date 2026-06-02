import { Router, Request, Response } from 'express';
import { ACTIVE_TIMEFRAME, PYTHON_API, AUTH_HEADERS, TIMEOUT_DEFAULT_MS, TIMEOUT_LONG_MS } from '../config';
import { proxyGet, fetchJsonFromPython } from '../utils/pythonApi';

export const aiRouter = Router();

// ── Pure proxies to FastAPI ──────────────────────────────────────────────────
// FastAPI owns every realtime JSON file. The Node backend has no filesystem
// knowledge of services/models/classification/data/realtime/* anymore.

aiRouter.get('/health',         async (_req, res) => proxyGet(`${PYTHON_API}/health`, res));
aiRouter.get('/status',         async (_req, res) => proxyGet(`${PYTHON_API}/status`, res));
aiRouter.get('/scheduler',      async (_req, res) => proxyGet(`${PYTHON_API}/scheduler`, res));
aiRouter.get('/breakdown',      async (_req, res) => proxyGet(`${PYTHON_API}/signal/breakdown`, res));
aiRouter.get('/news',           async (req, res)  => proxyGet(`${PYTHON_API}/news${req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : ''}`, res));
aiRouter.get('/signal',         async (_req, res) => proxyGet(`${PYTHON_API}/signal`, res));
aiRouter.get('/fusion',         async (_req, res) => proxyGet(`${PYTHON_API}/signal`, res));
aiRouter.get('/history',        async (_req, res) => proxyGet(`${PYTHON_API}/history`, res));
aiRouter.get('/system',         async (_req, res) => proxyGet(`${PYTHON_API}/system`, res));
aiRouter.get('/backtest',       async (_req, res) => proxyGet(`${PYTHON_API}/backtest`, res));
aiRouter.get('/walkforward',    async (_req, res) => proxyGet(`${PYTHON_API}/walkforward`, res));
aiRouter.get('/accuracy',       async (_req, res) => proxyGet(`${PYTHON_API}/accuracy`, res));
aiRouter.get('/accuracy/log',   async (_req, res) => proxyGet(`${PYTHON_API}/accuracy/log`, res));
aiRouter.get('/drift',          async (_req, res) => proxyGet(`${PYTHON_API}/drift`, res));
aiRouter.get('/rl-signal',      async (_req, res) => proxyGet(`${PYTHON_API}/rl-signal`, res));
aiRouter.get('/retrain/status', async (_req, res) => proxyGet(`${PYTHON_API}/retrain/status`, res));

// ── Shape-preserving fetches ─────────────────────────────────────────────────
// These used to read JSON files from disk. Data now comes from FastAPI; the
// reshape is kept here only to preserve the existing frontend contract.
// TODO (Phase 4 / API consolidation): enrich /signal/breakdown to surface
// event_type and high_impact_event, then collapse these into pure proxies.

interface FusedDecisionPayload {
  signal?: string;
  action?: string;
  confidence?: number;
  classification_probability_up?: number;
  classification_probability_down?: number;
  horizon?: string;
  high_impact_event?: boolean;
  event_type?: string | null;
  reasoning?: { classification?: string; sentiment?: string; spoofing?: string };
}

interface BreakdownPayload {
  action?: string;
  spoofing?: { confidence: number; is_suspicious: boolean; reasoning?: string };
  sentiment?: { label: string; score: number | null; reasoning: string };
}

aiRouter.get('/prediction', async (_req: Request, res: Response) => {
  try {
    const { status, data } = await fetchJsonFromPython<FusedDecisionPayload>(`${PYTHON_API}/signal`);
    if (!data) {
      res.status(status === 200 ? 404 : status).json({ error: 'No prediction data found. Run the pipeline first.' });
      return;
    }
    res.json({
      signal: data.signal ?? data.action,
      confidence: data.confidence,
      classification_probability_up: data.classification_probability_up,
      classification_probability_down: data.classification_probability_down,
      horizon: data.horizon ?? ACTIVE_TIMEFRAME,
    });
  } catch (err) {
    res.status(503).json({ error: 'Python API unavailable', detail: String(err) });
  }
});

aiRouter.get('/spoofing', async (_req: Request, res: Response) => {
  try {
    const { status, data } = await fetchJsonFromPython<BreakdownPayload>(`${PYTHON_API}/signal/breakdown`);
    if (!data?.spoofing) {
      res.status(status === 200 ? 404 : status).json({ error: 'No spoofing data found.' });
      return;
    }
    const { confidence, is_suspicious } = data.spoofing;
    res.json({
      confidence,
      severity: is_suspicious ? 'HIGH' : 'LOW',
      is_spoofing: is_suspicious,
      label: is_suspicious ? 'SPOOFING DETECTED' : 'NORMAL',
    });
  } catch (err) {
    res.status(503).json({ error: 'Python API unavailable', detail: String(err) });
  }
});

aiRouter.get('/sentiment', async (_req: Request, res: Response) => {
  // Source: last_fused_decision.json via FastAPI /signal. The fusion aggregator
  // now writes structured event_type + high_impact_event fields; we prefer those
  // and fall back to parsing the reasoning string for older JSON payloads.
  try {
    const { status, data } = await fetchJsonFromPython<FusedDecisionPayload>(`${PYTHON_API}/signal`);
    if (!data) {
      res.status(status === 200 ? 404 : status).json({ error: 'No sentiment data found. Run the pipeline first.' });
      return;
    }
    const raw = data.reasoning?.sentiment ?? '';
    const upper = raw.toUpperCase();
    const label = upper.includes('POSITIVE') ? 'POSITIVE'
      : upper.includes('NEGATIVE') ? 'NEGATIVE'
      : upper.includes('BULLISH')  ? 'BULLISH'
      : upper.includes('BEARISH')  ? 'BEARISH'
      : 'NEUTRAL';
    const scoreMatch = raw.match(/\(([-\d.]+)\)/);
    const score = scoreMatch ? parseFloat(scoreMatch[1]) : null;
    const eventMatch = raw.match(/event=(\S+)/);
    res.json({
      label,
      score,
      event_type: data.event_type ?? (eventMatch ? eventMatch[1] : null),
      high_impact: data.high_impact_event ?? false,
    });
  } catch (err) {
    res.status(503).json({ error: 'Python API unavailable', detail: String(err) });
  }
});

// ── Stop / trigger ────────────────────────────────────────────────────────────

aiRouter.post('/retrain/trigger', async (_req: Request, res: Response) => {
  try {
    const { default: fetch } = await import('node-fetch');
    const r = await fetch(`${PYTHON_API}/retrain/trigger`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...AUTH_HEADERS },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(TIMEOUT_DEFAULT_MS),
    });
    const data = await r.json() as Record<string, unknown>;
    res.status(r.status).json(data);
  } catch (err) {
    res.status(503).json({ error: 'Python API unavailable', detail: String(err) });
  }
});

aiRouter.post('/stop', async (_req: Request, res: Response) => {
  try {
    const { default: fetch } = await import('node-fetch');
    await fetch(`${PYTHON_API}/scheduler/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...AUTH_HEADERS },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(TIMEOUT_DEFAULT_MS),
    }).catch(() => null);
  } catch { /* scheduler may not be running */ }
  res.json({ stopped: true, message: 'Stop signal sent. Pipeline scheduler will pause after the current run completes.' });
});

aiRouter.post('/resume', async (_req: Request, res: Response) => {
  try {
    const { default: fetch } = await import('node-fetch');
    const r = await fetch(`${PYTHON_API}/scheduler/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...AUTH_HEADERS },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(TIMEOUT_LONG_MS),
    });
    const data = await r.json() as Record<string, unknown>;
    res.status(r.status).json({ ...data, resumed: true });
  } catch (err) {
    res.status(503).json({ error: 'Python API unavailable', detail: String(err) });
  }
});

aiRouter.post('/trigger', async (_req: Request, res: Response) => {
  try {
    const { default: fetch } = await import('node-fetch');
    const r = await fetch(`${PYTHON_API}/scheduler/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...AUTH_HEADERS },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(TIMEOUT_LONG_MS),
    });
    const data = await r.json() as Record<string, unknown>;
    res.status(r.status).json(data);
  } catch (err) {
    res.status(503).json({ error: 'Python API unavailable', detail: String(err) });
  }
});
