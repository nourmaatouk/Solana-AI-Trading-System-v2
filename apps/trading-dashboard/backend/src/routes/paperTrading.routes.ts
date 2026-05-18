import { Router, Request, Response } from 'express';
import { PYTHON_API, AUTH_HEADERS, TIMEOUT_PREFS_MS, TIMEOUT_PIPELINE_MS } from '../config';
import { proxyGet } from '../utils/pythonApi';

export const paperTradingRouter = Router();

// GET /api/paper-trading — proxies to Python /paper-trading
paperTradingRouter.get('/', async (_req: Request, res: Response) => {
  await proxyGet(`${PYTHON_API}/paper-trading`, res);
});

// POST /api/paper-trading/execute — saves risk_level preference then triggers a pipeline run
paperTradingRouter.post('/execute', async (req: Request, res: Response) => {
  try {
    const { default: fetch } = await import('node-fetch');
    const body = req.body as Record<string, unknown>;

    // Persist risk_level so the paper trader uses it on the next run
    if (body.risk_level) {
      await fetch(`${PYTHON_API}/preferences`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...AUTH_HEADERS },
        body: JSON.stringify({ risk_level: String(body.risk_level).toLowerCase() }),
        signal: AbortSignal.timeout(TIMEOUT_PREFS_MS),
      }).catch(() => null);
    }

    const r = await fetch(`${PYTHON_API}/scheduler/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...AUTH_HEADERS },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(TIMEOUT_PIPELINE_MS),
    });
    const data = await r.json() as Record<string, unknown>;
    res.status(r.status).json(data);
  } catch (err) {
    res.status(503).json({ error: 'Python API unavailable', detail: String(err) });
  }
});
