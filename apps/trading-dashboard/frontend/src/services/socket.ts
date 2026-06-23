import { WS_RECONNECT_BASE_MS, WS_RECONNECT_MAX_MS } from '../config';

type MessageHandler = (data: unknown) => void;

// Use the same host as the page so the Vite proxy (/ws) handles the upgrade in dev,
// and the production reverse-proxy handles it in prod — no hardcoded port needed.
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = `${protocol}//${window.location.host}/ws`;

class SocketService {
  private ws: WebSocket | null = null;
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private reconnectDelay = WS_RECONNECT_BASE_MS;
  private destroyed = false;

  connect(): void {
    // Reuse a socket that is OPEN *or still CONNECTING*. Guarding only on OPEN
    // lets React StrictMode (which mounts→unmounts→remounts effects in dev)
    // create a second WebSocket while the first is mid-handshake. The first is
    // then orphaned but stays connected and keeps delivering every message,
    // so each broadcast fires the handlers twice → duplicate history rows.
    if (
      this.ws &&
      (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    try {
      this.ws = new WebSocket(WS_URL);

      this.ws.onopen = () => {
        console.log('[socket] Connected to backend WS');
        this.reconnectDelay = WS_RECONNECT_BASE_MS;
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data as string) as { type: string };
          const handlers = this.handlers.get(msg.type);
          if (handlers) {
            handlers.forEach((h) => h(msg));
          }
          // Also fire '*' catch-all handlers
          const catchAll = this.handlers.get('*');
          if (catchAll) {
            catchAll.forEach((h) => h(msg));
          }
        } catch { /* ignore */ }
      };

      this.ws.onclose = () => {
        console.warn('[socket] Disconnected');
        if (!this.destroyed) this.scheduleReconnect();
      };

      this.ws.onerror = (err) => {
        console.error('[socket] Error', err);
      };
    } catch (err) {
      console.error('[socket] Failed to connect:', err);
      if (!this.destroyed) this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    setTimeout(() => {
      if (!this.destroyed) {
        console.log(`[socket] Reconnecting in ${this.reconnectDelay}ms...`);
        this.connect();
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, WS_RECONNECT_MAX_MS);
      }
    }, this.reconnectDelay);
  }

  on(type: string, handler: MessageHandler): void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
  }

  off(type: string, handler: MessageHandler): void {
    this.handlers.get(type)?.delete(handler);
  }

  send(payload: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  destroy(): void {
    this.destroyed = true;
    this.ws?.close();
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Singleton
export const socketService = new SocketService();
