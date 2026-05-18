import fs from 'fs';
import path from 'path';
import WebSocket, { WebSocketServer } from 'ws';

import { SIGNAL_POLL_INTERVAL_MS } from '../config';

const REALTIME_DIR = process.env.REALTIME_DIR
  ?? path.join(process.cwd(), 'services/models/classification/data/realtime');

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

export class SignalPoller {
  private wss: WebSocketServer;
  private interval: NodeJS.Timeout | null = null;
  private lastSignal: FusedDecision | null = null;

  constructor(wss: WebSocketServer) {
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
    if (signal) {
      this.lastSignal = signal;
      this.broadcast(signal);
    }
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

  readFusion(): FusedDecision | null {
    return readJson<FusedDecision>('last_fused_decision.json');
  }

  readSpoofing(): Record<string, unknown> | null {
    return readJson<Record<string, unknown>>('spoofing_score.json');
  }

  readSentiment(): Record<string, unknown> | null {
    return readJson<Record<string, unknown>>('sentiment_status.json');
  }
}
