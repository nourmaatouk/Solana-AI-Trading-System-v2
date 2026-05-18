import WebSocket, { WebSocketServer } from 'ws';
import { SignalPoller, FusedDecision } from '../services/aiSignal.service';
import { SIGNAL_POLL_INTERVAL_MS } from '../config';

export interface AISignalMessage {
  type: 'signal';
  data: FusedDecision;
}

export function broadcastSignal(wss: WebSocketServer, signal: FusedDecision): void {
  const msg = JSON.stringify({ type: 'signal', data: signal });
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  });
}

export function startSignalBroadcast(wss: WebSocketServer, poller: SignalPoller): NodeJS.Timeout {
  return setInterval(() => {
    const signal = poller.readFusion();
    if (signal) {
      broadcastSignal(wss, signal);
    }
  }, SIGNAL_POLL_INTERVAL_MS);
}
