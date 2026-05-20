import type { Response } from 'express';
import { AUTH_HEADERS, TIMEOUT_DEFAULT_MS } from '../config';

export interface PythonResult<T> {
  status: number;
  data: T | null;
}

/**
 * Single underlying call to FastAPI. All consumers (proxy or shape-preserving)
 * go through this so the headers, timeout, and JSON parsing exist in one place.
 *
 * Returns the parsed body regardless of status — FastAPI's error bodies are
 * still JSON and the caller may want to surface them.
 */
export async function fetchJsonFromPython<T = Record<string, unknown>>(url: string): Promise<PythonResult<T>> {
  const { default: fetch } = await import('node-fetch');
  const r = await fetch(url, { headers: AUTH_HEADERS, signal: AbortSignal.timeout(TIMEOUT_DEFAULT_MS) });
  let data: T | null = null;
  try { data = await r.json() as T; } catch { /* non-JSON body — leave data null */ }
  return { status: r.status, data };
}

/**
 * Forward a GET request to FastAPI and write the parsed JSON straight to the
 * Express response. Use this when the Node side has nothing to add or reshape.
 */
export async function proxyGet(url: string, res: Response): Promise<void> {
  try {
    const { status, data } = await fetchJsonFromPython(url);
    res.status(status).json(data ?? { error: 'Python API returned no body' });
  } catch (err) {
    res.status(503).json({ error: 'Python API unavailable', detail: String(err) });
  }
}
