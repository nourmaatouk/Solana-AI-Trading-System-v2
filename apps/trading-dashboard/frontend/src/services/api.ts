import type { AISignalData } from '../types';
import { API_TIMEOUT_GET_MS, API_TIMEOUT_POST_MS } from '../config';

const BASE = '/api';

async function get<T>(path: string, timeoutMs = API_TIMEOUT_GET_MS): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { signal: AbortSignal.timeout(timeoutMs) });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText })) as { error?: string };
    throw new Error(err.error ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body?: unknown, timeoutMs = API_TIMEOUT_POST_MS): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText })) as { error?: string };
    throw new Error(err.error ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export const fetchSignal = (): Promise<AISignalData> =>
  get<AISignalData>('/ai/signal');

export const fetchSpoofing = (): Promise<{ confidence: number; severity: string; is_spoofing: boolean; label: string }> =>
  get('/ai/spoofing');

export const fetchSentiment = (): Promise<{ label: string; score: number | null; event_type: string | null; high_impact: boolean }> =>
  get('/ai/sentiment');

export const fetchFusion = (): Promise<AISignalData> =>
  get<AISignalData>('/ai/fusion');

export const fetchStatus = (): Promise<Record<string, unknown>> =>
  get('/ai/status');

export const fetchHistory = (): Promise<unknown[]> =>
  get('/ai/history');

export const fetchPaperTrading = (): Promise<Record<string, unknown>> =>
  get('/paper-trading');

export const executePaperTrade = (params?: unknown): Promise<Record<string, unknown>> =>
  post('/paper-trading/execute', params);

export const triggerSignal = (): Promise<Record<string, unknown>> =>
  post('/ai/trigger');

export const stopScheduler = (): Promise<Record<string, unknown>> =>
  post('/ai/stop');

export const fetchScheduler = (): Promise<Record<string, unknown>> =>
  get('/ai/scheduler');

export const fetchSystem = (): Promise<Record<string, unknown>> =>
  get('/ai/system');

export const fetchBacktest = (): Promise<Record<string, unknown>> =>
  get('/ai/backtest');

export const fetchWalkForward = (): Promise<Record<string, unknown>> =>
  get('/ai/walkforward');

export const resumeScheduler = (): Promise<Record<string, unknown>> =>
  post('/ai/resume');

export const fetchAccuracy = (): Promise<Record<string, unknown>> =>
  get('/ai/accuracy');

export const fetchDrift = (): Promise<Record<string, unknown>> =>
  get('/ai/drift');

export const fetchRetrainStatus = (): Promise<Record<string, unknown>> =>
  get('/ai/retrain/status');

export const triggerRetrain = (): Promise<Record<string, unknown>> =>
  post('/ai/retrain/trigger');

export const fetchRLSignal = (): Promise<Record<string, unknown>> =>
  get('/ai/rl-signal');
