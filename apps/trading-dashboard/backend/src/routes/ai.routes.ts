import { Router, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';

export const aiRouter = Router();

const PYTHON_API = process.env.PYTHON_API ?? 'http://localhost:8000';
const REALTIME_DIR = process.env.REALTIME_DIR ?? '/home/nourmaatouk/Solana-AI-Trading-System/services/models/classification/data/realtime';

function readJsonFile(filename: string): Record<string, unknown> | null {
  try {
    const filePath = path.join(REALTIME_DIR, filename);
    const raw = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return null;
  }
}

async function proxyGet(url: string, res: Response): Promise<void> {
  try {
    const { default: fetch } = await import('node-fetch');
    const r = await fetch(url, { signal: AbortSignal.timeout(5000) });
    const data = await r.json() as Record<string, unknown>;
    res.status(r.status).json(data);
  } catch (err) {
    res.status(503).json({ error: 'Python API unavailable', detail: String(err) });
  }
}

// Proxy routes ────────────────────────────────────────────────────────────────

aiRouter.get('/health', async (_req: Request, res: Response) => {
  await proxyGet(`${PYTHON_API}/health`, res);
});

aiRouter.get('/status', async (_req: Request, res: Response) => {
  await proxyGet(`${PYTHON_API}/status`, res);
});

aiRouter.get('/scheduler', async (_req: Request, res: Response) => {
  await proxyGet(`${PYTHON_API}/scheduler`, res);
});

aiRouter.get('/breakdown', async (_req: Request, res: Response) => {
  await proxyGet(`${PYTHON_API}/signal/breakdown`, res);
});

aiRouter.get('/signal', async (_req: Request, res: Response) => {
  await proxyGet(`${PYTHON_API}/signal`, res);
});

aiRouter.get('/history', async (_req: Request, res: Response) => {
  await proxyGet(`${PYTHON_API}/history`, res);
});

// File-based routes ───────────────────────────────────────────────────────────

aiRouter.get('/prediction', (_req: Request, res: Response) => {
  const data = readJsonFile('last_fused_decision.json');
  if (!data) {
    res.status(404).json({ error: 'No prediction data found. Run the pipeline first.' });
    return;
  }
  res.json({
    signal: data.signal ?? data.action,
    confidence: data.confidence,
    classification_probability_up: data.classification_probability_up,
    classification_probability_down: data.classification_probability_down,
    horizon: data.horizon ?? '15m',
  });
});

aiRouter.get('/spoofing', (_req: Request, res: Response) => {
  const data = readJsonFile('spoofing_score.json');
  if (!data) {
    res.status(404).json({ error: 'No spoofing data found.' });
    return;
  }
  res.json({
    confidence: data.spoofing_probability ?? data.confidence ?? 0,
    severity: data.is_suspicious ? 'HIGH' : 'LOW',
    is_spoofing: data.is_suspicious ?? false,
    label: data.is_suspicious ? 'SPOOFING DETECTED' : 'NORMAL',
  });
});

aiRouter.get('/sentiment', (_req: Request, res: Response) => {
  const data = readJsonFile('sentiment_status.json');
  if (!data) {
    res.status(404).json({ error: 'No sentiment data found.' });
    return;
  }
  res.json({
    label: data.label ?? data.status ?? 'UNKNOWN',
    score: data.score ?? data.sentiment_score ?? null,
    event_type: data.event_type ?? null,
    high_impact: data.high_impact ?? false,
  });
});

aiRouter.get('/fusion', (_req: Request, res: Response) => {
  const data = readJsonFile('last_fused_decision.json');
  if (!data) {
    res.status(404).json({ error: 'No fusion data found. Run the pipeline first.' });
    return;
  }
  res.json(data);
});

aiRouter.get('/system', async (_req: Request, res: Response) => {
  await proxyGet(`${PYTHON_API}/system`, res);
});

aiRouter.get('/backtest', async (_req: Request, res: Response) => {
  await proxyGet(`${PYTHON_API}/backtest`, res);
});

aiRouter.get('/walkforward', async (_req: Request, res: Response) => {
  await proxyGet(`${PYTHON_API}/walkforward`, res);
});

// Stop / trigger ──────────────────────────────────────────────────────────────

aiRouter.post('/stop', async (_req: Request, res: Response) => {
  try {
    const { default: fetch } = await import('node-fetch');
    // Best-effort: tell Python scheduler to stop; ignore if not running
    await fetch(`${PYTHON_API}/scheduler/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(5000),
    }).catch(() => null);
  } catch { /* scheduler may not be running */ }
  res.json({ stopped: true, message: 'Stop signal sent. Pipeline scheduler will pause after the current run completes.' });
});

aiRouter.post('/resume', async (_req: Request, res: Response) => {
  try {
    const { default: fetch } = await import('node-fetch');
    const r = await fetch(`${PYTHON_API}/scheduler/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(10000),
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
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(10000),
    });
    const data = await r.json() as Record<string, unknown>;
    res.status(r.status).json(data);
  } catch (err) {
    res.status(503).json({ error: 'Python API unavailable', detail: String(err) });
  }
});
