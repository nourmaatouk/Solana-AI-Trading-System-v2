/**
 * HTTP client for the FastAPI signal endpoint.
 *
 * The Python backend already exposes the fused decision at GET /signal. This
 * module fetches it, normalises the response shape (the upstream returns
 * either the `fused_decision` shape or the legacy `xgboost_only` shape), and
 * returns a clean GuardianInput-compatible object.
 */
import { loadEnv } from './config/env';
import { log } from './logger';
import type { RiskLevel, TradeAction } from './guardian/types';

export interface NormalisedSignal {
  asset: string;
  action: TradeAction;
  confidence: number;
  fused_score: number;
  risk_level: RiskLevel;
  spoofing_confidence: number;
  sentiment_relevance?: number;
  classification_confidence?: number;
  classification_probability_up?: number;
  signal_generated_at_utc: string | null;
  raw: Record<string, unknown>;
}

export interface SignalClientOptions {
  baseUrl?: string;
  apiKey?: string;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}

export class SignalClient {
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly fetchImpl: typeof fetch;
  private readonly timeoutMs: number;

  constructor(opts: SignalClientOptions = {}) {
    const env = loadEnv();
    this.baseUrl = (opts.baseUrl ?? env.PYTHON_API).replace(/\/$/, '');
    this.apiKey = opts.apiKey ?? env.PYTHON_API_KEY;
    this.fetchImpl = opts.fetchImpl ?? fetch;
    this.timeoutMs = opts.timeoutMs ?? 5_000;
  }

  async getLatestSignal(): Promise<NormalisedSignal> {
    const url = `${this.baseUrl}/signal`;
    const headers: Record<string, string> = { Accept: 'application/json' };
    if (this.apiKey) headers['X-API-Key'] = this.apiKey;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await this.fetchImpl(url, { headers, signal: controller.signal });
      if (!res.ok) {
        throw new Error(`Signal API returned ${res.status}: ${await res.text()}`);
      }
      const raw = (await res.json()) as Record<string, unknown>;
      return this.normalise(raw);
    } finally {
      clearTimeout(timer);
    }
  }

  private normalise(raw: Record<string, unknown>): NormalisedSignal {
    const action = String(raw.signal ?? raw.action ?? 'HOLD').toUpperCase();
    const normalisedAction: TradeAction =
      action === 'BUY' || action === 'UP' ? 'BUY'
        : action === 'SELL' || action === 'DOWN' ? 'SELL'
        : 'HOLD';

    const generatedAt =
      typeof raw.generated_at_utc === 'string' ? raw.generated_at_utc
        : typeof raw.signal_generated_at_utc === 'string' ? raw.signal_generated_at_utc
        : null;

    const riskRaw = String(raw.risk_level ?? 'UNKNOWN').toUpperCase();
    const risk_level: RiskLevel =
      riskRaw === 'LOW' || riskRaw === 'MEDIUM' || riskRaw === 'HIGH'
        ? riskRaw
        : 'UNKNOWN';

    const asset = String(raw.asset ?? 'SOL/USDT');

    const num = (v: unknown, fallback = 0): number => {
      const n = typeof v === 'number' ? v : Number(v);
      return Number.isFinite(n) ? n : fallback;
    };

    const result: NormalisedSignal = {
      asset,
      action: normalisedAction,
      confidence: num(raw.confidence),
      fused_score: num(raw.fused_score),
      risk_level,
      spoofing_confidence: num(raw.spoofing_confidence),
      sentiment_relevance: num(raw.sentiment_relevance, 0),
      classification_confidence: num(raw.classification_confidence, 0),
      classification_probability_up: num(raw.classification_probability_up, 0),
      signal_generated_at_utc: generatedAt,
      raw,
    };

    log.debug('signal.fetched', {
      source: raw._source,
      action: result.action,
      confidence: result.confidence,
      risk: result.risk_level,
      age_minutes: raw._age_minutes,
    });

    return result;
  }
}
