/**
 * Singleton event emitter for execution lifecycle events.
 * Decouples executionState.service.ts (which writes files and emits events)
 * from server.ts (which owns the WebSocket server and forwards events to clients).
 */
import { EventEmitter } from 'events';

export interface ExecutionHistoryEntry {
  request_received_at_utc: string;
  decision: 'APPROVE' | 'BLOCK';
  blocked_by: string | null;
  reasons: string[];
  execution_status: string | null;
  mode: string | null;
  /** The fused model action the Guardian evaluated (BUY/SELL/HOLD). Unlike
   *  `side`, this survives a block so the history can show the intended trade. */
  signal_action: 'BUY' | 'SELL' | 'HOLD' | null;
  side: 'BUY' | 'SELL' | null;
  amount_sol: number | null;
  asset: string | null;
  signature: string | null;
  error: string | null;
  /** Base58 public key of the keypair that signed the on-chain transaction. */
  signed_by: string | null;
}

class ExecutionEventEmitter extends EventEmitter {}

export const executionEvents = new ExecutionEventEmitter();
