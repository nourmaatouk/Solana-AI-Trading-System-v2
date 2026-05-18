import type { Response } from 'express';
import { AUTH_HEADERS, TIMEOUT_DEFAULT_MS } from '../config';

export async function proxyGet(url: string, res: Response): Promise<void> {
  try {
    const { default: fetch } = await import('node-fetch');
    const r = await fetch(url, { headers: AUTH_HEADERS, signal: AbortSignal.timeout(TIMEOUT_DEFAULT_MS) });
    const data = await r.json() as Record<string, unknown>;
    res.status(r.status).json(data);
  } catch (err) {
    res.status(503).json({ error: 'Python API unavailable', detail: String(err) });
  }
}
