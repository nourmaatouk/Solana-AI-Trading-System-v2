import { Router, Request, Response } from 'express';

export const paperTradingRouter = Router();

const PYTHON_API = process.env.PYTHON_API ?? 'http://localhost:8000';

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

// GET /api/paper-trading — proxies to Python /paper-trading
paperTradingRouter.get('/', async (_req: Request, res: Response) => {
  await proxyGet(`${PYTHON_API}/paper-trading`, res);
});

// POST /api/paper-trading/execute — triggers a pipeline run (simulated trade)
paperTradingRouter.post('/execute', async (_req: Request, res: Response) => {
  try {
    const { default: fetch } = await import('node-fetch');
    const r = await fetch(`${PYTHON_API}/scheduler/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(15000),
    });
    const data = await r.json() as Record<string, unknown>;
    res.status(r.status).json(data);
  } catch (err) {
    res.status(503).json({ error: 'Python API unavailable', detail: String(err) });
  }
});
