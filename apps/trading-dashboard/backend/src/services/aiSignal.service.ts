import fs from 'fs';
import path from 'path';
import { EventEmitter } from 'events';
import WebSocket, { WebSocketServer } from 'ws';

import { SIGNAL_POLL_INTERVAL_MS } from '../config';

// NOTE: This poller still reads last_fused_decision.json from disk for WS
// fan-out. The audit flagged this as a follow-up to Phase 2 — switching it
// to fetch from FastAPI removes the last bit of filesystem coupling in the
// Node backend. Not done here to keep Phase 2 scope tight.
const REALTIME_DIR = process.env.REALTIME_DIR
  ?? path.join(process.cwd(), 'services/models/classification/data/realtime');

// Mirror of the Guardian's freshness window. On the first poll after the
// backend starts, signals older than this are seeded into the dedup state
// but NOT emitted — so a long-lived on-disk JSON from a previous run cannot
// trigger a "fresh signal" event after restart.
const SIGNAL_MAX_AGE_SECONDS = parseInt(
  process.env.SIGNAL_MAX_AGE_SECONDS ?? '300',
  10,
);

export interface FusedDecision {
  status?: string;
  generated_at_utc?: string;
  asset?: string;
  signal_timestamp?: string;
  latest_close?: number;
  action?: string;
  signal?: string;
  fused_score?: number;
  decision_score?: number;
  confidence?: number;
  risk_level?: string;
  classification_confidence?: number;
  classification_probability_up?: number;
  classification_probability_down?: number;
  spoofing_confidence?: number;
  sentiment_relevance?: number;
  reason?: string;
  reasoning?: {
    classification?: string;
    sentiment?: string;
    spoofing?: string;
  };
}

function readJson<T>(filename: string): T | null {
  try {
    const raw = fs.readFileSync(path.join(REALTIME_DIR, filename), 'utf8');
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

/**
 * Stable identity key for a signal. Two signals are "the same" when all five
 * fields are identical. A new generated_at_utc or changed fused_score/action
 * means a genuinely new signal that may warrant auto-execution.
 */
export function computeSignalKey(s: FusedDecision): string {
  return [
    s.asset ?? '',
    s.action ?? '',
    s.generated_at_utc ?? s.signal_timestamp ?? '',
    String(s.fused_score ?? ''),
    String(s.confidence ?? ''),
  ].join('|');
}

/**
 * Emits:
 *   'signal_updated' (signal: FusedDecision) — fired when the signal file
 *   contains a signal with a different key from the last one seen. Multiple
 *   polls that return the same signal only fire the event once.
 */
export class SignalPoller extends EventEmitter {
  private wss: WebSocketServer;
  private interval: NodeJS.Timeout | null = null;
  // Cached last fused decision — delivered to new WS clients on connect
  // (market.socket.ts) so they see the latest signal without waiting for the
  // next poll cycle.
  private lastSignal: FusedDecision | null = null;
  private lastSignalKey: string | null = null;
  // First-poll guard: on startup we seed lastSignalKey from any pre-existing
  // signal file without emitting, so a stale signal left on disk by a previous
  // run cannot trigger auto-execution on restart.
  private hasPolledOnce = false;

  constructor(wss: WebSocketServer) {
    super();
    this.wss = wss;
  }

  start(): void {
    this.poll();
    this.interval = setInterval(() => this.poll(), SIGNAL_POLL_INTERVAL_MS);
    console.log('[signal-poller] Started, polling every', SIGNAL_POLL_INTERVAL_MS / 1000, 's');
  }

  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  private poll(): void {
    const signal = readJson<FusedDecision>('last_fused_decision.json');
    if (!signal) {
      this.hasPolledOnce = true;
      return;
    }

    this.lastSignal = signal;
    this.broadcast(signal);

    const key = computeSignalKey(signal);
    const keyChanged = key !== this.lastSignalKey;
    this.lastSignalKey = key;

    if (!keyChanged) {
      this.hasPolledOnce = true;
      return;
    }

    // On the very first poll after startup, suppress emission for signals
    // that are already stale — they were written by a previous backend run
    // and shouldn't trigger auto-execution as if they were brand new. The
    // key is still recorded so subsequent polls with the same content stay
    // silent until a genuinely new signal arrives.
    if (!this.hasPolledOnce) {
      this.hasPolledOnce = true;
      if (this.isStale(signal)) {
        console.log(
          `[signal-poller] Startup: on-disk signal is older than ${SIGNAL_MAX_AGE_SECONDS}s — seeded, not emitted.`,
        );
        return;
      }
    }

    this.emit('signal_updated', signal);
  }

  private isStale(signal: FusedDecision): boolean {
    const ts = signal.generated_at_utc ?? signal.signal_timestamp;
    if (!ts) return true; // missing timestamp → treat as stale, fail safe
    const ms = Date.parse(ts);
    if (!Number.isFinite(ms)) return true;
    const ageSec = (Date.now() - ms) / 1000;
    return ageSec > SIGNAL_MAX_AGE_SECONDS;
  }

  private broadcast(signal: FusedDecision): void {
    const msg = JSON.stringify({ type: 'signal', data: signal });
    this.wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(msg);
      }
    });
  }

  getLatestSignal(): FusedDecision | null {
    return this.lastSignal;
  }
}
