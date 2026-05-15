// Paper trading state management — proxies to Python backend

const PYTHON_API = 'http://localhost:8000';

export interface PaperTradeResult {
  success: boolean;
  message?: string;
  data?: Record<string, unknown>;
  error?: string;
}

export async function fetchPaperTradingLog(): Promise<Record<string, unknown> | null> {
  try {
    const { default: fetch } = await import('node-fetch');
    const r = await fetch(`${PYTHON_API}/paper-trading`, { signal: AbortSignal.timeout(5000) });
    return await r.json() as Record<string, unknown>;
  } catch {
    return null;
  }
}

export async function triggerPipelineRun(): Promise<PaperTradeResult> {
  try {
    const { default: fetch } = await import('node-fetch');
    const r = await fetch(`${PYTHON_API}/scheduler/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(15000),
    });
    const data = await r.json() as Record<string, unknown>;
    return { success: r.ok, data };
  } catch (err) {
    return { success: false, error: String(err) };
  }
}
