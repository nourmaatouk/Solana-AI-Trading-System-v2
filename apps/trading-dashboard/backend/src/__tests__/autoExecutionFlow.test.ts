/**
 * Integration test for the SignalPoller -> AutoExecutor wire.
 *
 * Closes the gap our pure-Guardian tests don't cover: that backend state
 * files on disk (auto_execute, user_risk_level, max_trade_size, daily_volume)
 * are actually read by the AutoExecutor and forwarded to
 * ExecutionWorkflow.execute, and that the persistent dedup log (set on
 * disk) prevents the same signal from re-firing after a restart.
 *
 * The ExecutionWorkflow is replaced by a vi.fn() — we're testing the
 * wiring, not the Guardian. The Guardian path is covered by the
 * integration tests in services/execution.
 */

// MUST run before any module imports below — autoExecutor.service.ts reads
// these at module load time and caches them in module-level consts.
process.env.WALLET_MODE = 'paper';
process.env.AUTO_EXECUTOR_POLL_MS = '3600000';        // disable poll fallback
process.env.SIGNAL_DEDUP_WINDOW_SECONDS = '300';
process.env.AUTO_EXECUTE_AMOUNT_SOL = '0.01';

import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { WebSocketServer } from 'ws';

let TMP_DIR: string;

function writeJson(filename: string, data: unknown): void {
  fs.writeFileSync(path.join(TMP_DIR, filename), JSON.stringify(data, null, 2));
}

function freshSignal(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  const now = new Date().toISOString();
  return {
    status: 'ok',
    generated_at_utc: now,
    asset: 'SOL/USDT',
    signal_timestamp: now,
    latest_close: 87,
    action: 'BUY',
    signal: 'BUY',
    fused_score: 0.78,
    decision_score: 0.78,
    confidence: 0.81,
    risk_level: 'LOW',
    spoofing_confidence: 0.04,
    sentiment_relevance: 1.0,
    ...overrides,
  };
}

function mockWorkflowExecuteResult(overrides: Record<string, unknown> = {}) {
  const now = new Date().toISOString();
  return {
    request_received_at_utc: now,
    request_completed_at_utc: now,
    guardian: {
      approved: true,
      decision: 'APPROVE',
      reasons: ['ok'],
      blocked_by: null,
      sanitized_action: {
        asset: 'SOL',
        side: 'BUY',
        amount_sol: 0.05,
        execution_mode: 'paper',
      },
      evaluated_at_utc: now,
    },
    execution: {
      status: 'simulated',
      mode: 'paper',
      signature: `PAPER-test-${Date.now()}`,
      block_height: null,
      asset: 'SOL',
      side: 'BUY',
      amount_sol: 0.05,
      rpc_url: null,
      submitted_at_utc: now,
      confirmed_at_utc: now,
      error: null,
    },
    narration: null,
    ...overrides,
  };
}

const stubWss = { clients: new Set() } as unknown as WebSocketServer;

async function importModules() {
  vi.resetModules();
  const aiSignalModule = await import('../services/aiSignal.service');
  const autoExecModule = await import('../services/autoExecutor.service');
  return { SignalPoller: aiSignalModule.SignalPoller, AutoExecutor: autoExecModule.AutoExecutor };
}

beforeEach(() => {
  TMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'autoexec-test-'));
  process.env.REALTIME_DIR = TMP_DIR;

  // Default state: auto-execute ON, no emergency stop, LOW risk, 0.05 SOL cap.
  writeJson('auto_execute.json', { enabled: true, updated_at_utc: new Date().toISOString() });
  writeJson('user_risk_level.json', { level: 'LOW', updated_at_utc: new Date().toISOString() });
  writeJson('max_trade_size_sol.json', {
    amount_sol: 0.05,
    updated_at_utc: new Date().toISOString(),
  });
  writeJson('emergency_stop.json', { enabled: false, updated_at_utc: new Date().toISOString(), reason: null });
});

afterEach(() => {
  fs.rmSync(TMP_DIR, { recursive: true, force: true });
});

describe('SignalPoller -> AutoExecutor', () => {
  it('forwards the persisted user_risk_level + max_trade_size to workflow.execute', async () => {
    writeJson('last_fused_decision.json', freshSignal());
    const { SignalPoller, AutoExecutor } = await importModules();

    const executeFn = vi.fn().mockResolvedValue(mockWorkflowExecuteResult());
    const poller = new SignalPoller(stubWss);
    const executor = new AutoExecutor({ execute: executeFn } as never, poller);

    poller.start();
    // Let the async _handleSignal chain settle.
    await new Promise((r) => setTimeout(r, 30));
    poller.stop();
    executor.stop();

    expect(executeFn).toHaveBeenCalledOnce();
    const arg = executeFn.mock.calls[0][0];
    expect(arg.amount_sol).toBe(0.05);          // from max_trade_size_sol.json
    expect(arg.user_risk_level).toBe('LOW');     // from user_risk_level.json
    expect(arg.daily_executed_sol).toBe(0);      // no daily_volume.json yet today
  });

  it('skips execution when auto_execute.json is disabled', async () => {
    writeJson('auto_execute.json', { enabled: false, updated_at_utc: new Date().toISOString() });
    writeJson('last_fused_decision.json', freshSignal());
    const { SignalPoller, AutoExecutor } = await importModules();

    const executeFn = vi.fn().mockResolvedValue(mockWorkflowExecuteResult());
    const poller = new SignalPoller(stubWss);
    const executor = new AutoExecutor({ execute: executeFn } as never, poller);

    poller.start();
    await new Promise((r) => setTimeout(r, 30));
    poller.stop();
    executor.stop();

    expect(executeFn).not.toHaveBeenCalled();
  });

  it('skips execution when emergency_stop is engaged', async () => {
    writeJson('emergency_stop.json', {
      enabled: true,
      updated_at_utc: new Date().toISOString(),
      reason: 'test',
    });
    writeJson('last_fused_decision.json', freshSignal());
    const { SignalPoller, AutoExecutor } = await importModules();

    const executeFn = vi.fn().mockResolvedValue(mockWorkflowExecuteResult());
    const poller = new SignalPoller(stubWss);
    const executor = new AutoExecutor({ execute: executeFn } as never, poller);

    poller.start();
    await new Promise((r) => setTimeout(r, 30));
    poller.stop();
    executor.stop();

    expect(executeFn).not.toHaveBeenCalled();
    expect(executor.getStatus().lastSkippedReason).toMatch(/emergency stop/i);
  });

  it('skips HOLD signals (only BUY/SELL trigger auto-execution)', async () => {
    writeJson('last_fused_decision.json', freshSignal({ action: 'HOLD', signal: 'HOLD' }));
    const { SignalPoller, AutoExecutor } = await importModules();

    const executeFn = vi.fn().mockResolvedValue(mockWorkflowExecuteResult());
    const poller = new SignalPoller(stubWss);
    const executor = new AutoExecutor({ execute: executeFn } as never, poller);

    poller.start();
    await new Promise((r) => setTimeout(r, 30));
    poller.stop();
    executor.stop();

    expect(executeFn).not.toHaveBeenCalled();
    expect(executor.getStatus().lastSkippedReason).toMatch(/HOLD/);
  });

  it('does NOT re-execute the same signal emitted twice (in-memory dedup)', async () => {
    writeJson('last_fused_decision.json', freshSignal());
    const { SignalPoller, AutoExecutor } = await importModules();

    const executeFn = vi.fn().mockResolvedValue(mockWorkflowExecuteResult());
    const poller = new SignalPoller(stubWss);
    const executor = new AutoExecutor({ execute: executeFn } as never, poller);

    // First emission: workflow runs.
    poller.start();
    await new Promise((r) => setTimeout(r, 30));

    // Bypass the SignalPoller's own key-equality short-circuit and emit the
    // event a second time directly, exercising AutoExecutor's SignalDedup.
    const cached = poller.getLatestSignal()!;
    poller.emit('signal_updated', cached);
    await new Promise((r) => setTimeout(r, 30));

    poller.stop();
    executor.stop();

    expect(executeFn).toHaveBeenCalledOnce();
    expect(executor.getStatus().lastSkippedReason).toMatch(/duplicate signal key/i);
  });

  it('re-executes a genuinely new signal (different key)', async () => {
    writeJson('last_fused_decision.json', freshSignal());
    const { SignalPoller, AutoExecutor } = await importModules();

    const executeFn = vi.fn().mockResolvedValue(mockWorkflowExecuteResult());
    const poller = new SignalPoller(stubWss);
    const executor = new AutoExecutor({ execute: executeFn } as never, poller);

    poller.start();
    await new Promise((r) => setTimeout(r, 30));
    poller.stop();

    // Write a new signal file with a different timestamp (different key) and
    // emit. AutoExecutor should treat this as fresh.
    const second = freshSignal({
      generated_at_utc: new Date(Date.now() + 1).toISOString(),
      fused_score: 0.42,
    });
    writeJson('last_fused_decision.json', second);
    poller.emit('signal_updated', second);
    await new Promise((r) => setTimeout(r, 30));
    executor.stop();

    expect(executeFn).toHaveBeenCalledTimes(2);
  });

  it('persistent dedup: pre-seeded executed_signal_keys.json blocks re-execution after "restart"', async () => {
    const signal = freshSignal();
    writeJson('last_fused_decision.json', signal);

    // Seed the persistent dedup log with this exact signal's key, as if the
    // backend had already executed it before crashing and restarting.
    const key = [
      signal.asset ?? '',
      signal.action ?? '',
      signal.generated_at_utc ?? signal.signal_timestamp ?? '',
      String(signal.fused_score ?? ''),
      String(signal.confidence ?? ''),
    ].join('|');
    writeJson('executed_signal_keys.json', {
      entries: [
        {
          signal_key: key,
          executed_at_utc: new Date().toISOString(), // within dedup window
          action: 'BUY',
          amount_sol: 0.05,
        },
      ],
    });

    const { SignalPoller, AutoExecutor } = await importModules();
    const executeFn = vi.fn().mockResolvedValue(mockWorkflowExecuteResult());
    const poller = new SignalPoller(stubWss);
    const executor = new AutoExecutor({ execute: executeFn } as never, poller);

    poller.start();
    await new Promise((r) => setTimeout(r, 30));
    poller.stop();
    executor.stop();

    expect(executeFn).not.toHaveBeenCalled();
    expect(executor.getStatus().lastSkippedReason).toMatch(/duplicate signal key/i);
  });

  it('persistent dedup: writes executed_signal_keys.json after a successful trade', async () => {
    writeJson('last_fused_decision.json', freshSignal());
    const { SignalPoller, AutoExecutor } = await importModules();

    const executeFn = vi.fn().mockResolvedValue(mockWorkflowExecuteResult());
    const poller = new SignalPoller(stubWss);
    const executor = new AutoExecutor({ execute: executeFn } as never, poller);

    poller.start();
    await new Promise((r) => setTimeout(r, 30));
    poller.stop();
    executor.stop();

    const persisted = JSON.parse(
      fs.readFileSync(path.join(TMP_DIR, 'executed_signal_keys.json'), 'utf8'),
    );
    expect(persisted.entries).toHaveLength(1);
    expect(persisted.entries[0]).toMatchObject({ action: 'BUY', amount_sol: 0.05 });
  });

  it('does not emit signal_updated for a stale signal on first poll after startup', async () => {
    // Signal generated 10 minutes ago — outside the 300s freshness window.
    writeJson('last_fused_decision.json', freshSignal({
      generated_at_utc: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
      signal_timestamp: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
    }));
    const { SignalPoller, AutoExecutor } = await importModules();

    const executeFn = vi.fn().mockResolvedValue(mockWorkflowExecuteResult());
    const poller = new SignalPoller(stubWss);
    const executor = new AutoExecutor({ execute: executeFn } as never, poller);

    poller.start();
    await new Promise((r) => setTimeout(r, 30));
    poller.stop();
    executor.stop();

    // SignalPoller suppressed the emission, so AutoExecutor never saw it.
    expect(executeFn).not.toHaveBeenCalled();
  });
});
