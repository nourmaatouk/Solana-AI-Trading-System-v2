/**
 * Execution routes — the HTTP boundary for the agentic execution layer.
 *
 * Endpoints (all under /api/execution):
 *   POST /validate              → Guardian only, never touches Solana
 *   POST /execute               → Full workflow: signal → guardian → execute
 *   GET  /status                → mode, wallet pubkey, e-stop, last execution
 *   GET  /history               → last 100 execution entries (newest first)
 *   POST /emergency-stop        → enable e-stop (immediate effect)
 *   POST /emergency-resume      → disable e-stop (requires {confirm:true})
 *   POST /auto-execute/enable   → start auto-execution poller
 *   POST /auto-execute/disable  → stop auto-execution poller
 *   GET  /auto-execute/status   → current auto-execute state
 */
import { Router, type Request, type Response } from 'express';
import { z } from 'zod';
import {
  ExecutionWorkflow,
  defaultWalletService,
  loadEnv,
  type ExecutionRequest,
  type PrepareTransactionResult,
} from '@trading/execution';
import {
  readAutoExecute,
  readDailyVolume,
  readEmergencyStop,
  readExecutionHistory,
  readLastExecution,
  readUserMaxTradeSize,
  readUserRiskLevel,
  recordDailyVolume,
  recordLastExecution,
  writeAutoExecute,
  writeEmergencyStop,
  writeUserMaxTradeSize,
  writeUserRiskLevel,
} from '../services/executionState.service';
import type { AutoExecutorStatus } from '../services/autoExecutor.service';

export const executionRouter = Router();

// ── Live AutoExecutor runtime status ────────────────────────────────────────
// server.ts registers the running AutoExecutor instance here so the
// /auto-execute/status route can surface its runtime state (last signal,
// last skip reason, cooldown window) without importing the class (avoids a
// circular import). Null until registration completes at startup.
let _autoExecutorStatus: (() => AutoExecutorStatus) | null = null;
export function registerAutoExecutor(getStatus: () => AutoExecutorStatus): void {
  _autoExecutorStatus = getStatus;
}

// ── Shared workflow instance ────────────────────────────────────────────────
// One instance per process. The Guardian inside reads env at validate-time
// so a runtime EMERGENCY_STOP change applies on the very next request.
const workflow = new ExecutionWorkflow();

// ── Input validation ────────────────────────────────────────────────────────
const ExecutionRequestSchema = z.object({
  amount_sol: z.number().positive(),
  execution_mode: z.enum(['paper', 'devnet', 'mainnet']).optional(),
  narration_request: z.string().max(500).optional(),
  // Required when WALLET_MODE=phantom — validated by Guardian rule #12.
  phantom_public_key: z.string().optional(),
  // User's risk-tolerance preference from the frontend Settings page.
  user_risk_level: z.enum(['LOW', 'MEDIUM', 'HIGH']).optional(),
  signal: z
    .object({
      confidence: z.number().optional(),
      fused_score: z.number().optional(),
      risk_level: z.enum(['LOW', 'MEDIUM', 'HIGH', 'UNKNOWN']).optional(),
      spoofing_confidence: z.number().optional(),
    })
    .optional(),
});

// Body fields validated from the HTTP request. `daily_executed_sol` is always
// injected server-side (never trusted from the client) so it is intentionally
// absent from the schema and added explicitly at each call site.
type ParsedRequest = z.infer<typeof ExecutionRequestSchema>;

function parseRequest(req: Request, res: Response): ParsedRequest | null {
  const parsed = ExecutionRequestSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({
      error: 'Invalid request body',
      issues: parsed.error.issues.map((i) => ({
        path: i.path.join('.'),
        message: i.message,
      })),
    });
    return null;
  }
  return parsed.data;
}

// ── POST /validate ──────────────────────────────────────────────────────────
executionRouter.post('/validate', async (req, res) => {
  const parsed = parseRequest(req, res);
  if (!parsed) return;
  try {
    const guardian = await workflow.validate(parsed);
    res.json({ guardian });
  } catch (err) {
    res.status(503).json({ error: 'validation_failed', detail: String((err as Error).message) });
  }
});

// ── POST /execute ───────────────────────────────────────────────────────────
executionRouter.post('/execute', async (req, res) => {
  const parsed = parseRequest(req, res);
  if (!parsed) return;

  // Access-control: devnet execution requires a connected wallet public key.
  // The Guardian also enforces this (rule #12 for phantom mode), but rejecting
  // here gives a clear 401 before any execution logic is touched.
  const env = loadEnv();
  if (
    parsed.execution_mode === 'devnet' &&
    env.WALLET_MODE === 'phantom' &&
    !parsed.phantom_public_key
  ) {
    res.status(401).json({
      error: 'wallet_required',
      detail: 'Devnet execution requires a connected Solana wallet. Connect your wallet and retry.',
    });
    return;
  }

  try {
    const requestWithDaily = { ...parsed, daily_executed_sol: readDailyVolume() };
    const result = await workflow.execute(requestWithDaily);
    recordLastExecution(result);
    if (result.execution?.status === 'confirmed' || result.execution?.status === 'simulated') {
      recordDailyVolume(result.execution.amount_sol);
    }
    res.json(result);
  } catch (err) {
    res.status(503).json({ error: 'execution_failed', detail: String((err as Error).message) });
  }
});

// ── GET /status ─────────────────────────────────────────────────────────────
executionRouter.get('/status', async (_req, res) => {
  const env = loadEnv();
  const wallet = defaultWalletService();
  const walletInfo = wallet.getInfo();
  const balanceSol = await wallet.getBalanceSol();
  const eStop = readEmergencyStop();
  res.json({
    execution_mode: env.EXECUTION_MODE,
    allow_mainnet: env.ALLOW_MAINNET_EXECUTION,
    require_guardian_approval: env.REQUIRE_GUARDIAN_APPROVAL,
    emergency_stop: eStop,
    wallet: {
      public_key: walletInfo.public_key,
      cluster: walletInfo.cluster,
      rpc_url: walletInfo.rpc_url,
      signing_capable: walletInfo.signing_capable,
      balance_sol: balanceSol,
    },
    last_execution: readLastExecution(),
  });
});

// ── GET /history ─────────────────────────────────────────────────────────────
executionRouter.get('/history', (_req, res) => {
  res.json({ history: readExecutionHistory() });
});

// ── POST /prepare-tx ────────────────────────────────────────────────────────
// Guardian-validate then return an unsigned VersionedTransaction for
// client-side (Phantom) signing. Nothing is signed or submitted server-side.
executionRouter.post('/prepare-tx', async (req, res) => {
  const parsed = parseRequest(req, res);
  if (!parsed) return;
  try {
    const result: PrepareTransactionResult = await workflow.prepareTransaction(parsed);
    res.json(result);
  } catch (err) {
    res.status(503).json({ error: 'prepare_failed', detail: String((err as Error).message) });
  }
});

// ── POST /emergency-stop ────────────────────────────────────────────────────
executionRouter.post('/emergency-stop', (req, res) => {
  const reason = typeof req.body?.reason === 'string' ? req.body.reason : 'manual stop';
  const state = writeEmergencyStop(true, reason);
  res.json({ ok: true, emergency_stop: state });
});

// ── POST /emergency-resume ──────────────────────────────────────────────────
executionRouter.post('/emergency-resume', (req, res) => {
  if (req.body?.confirm !== true) {
    res.status(400).json({
      error: 'confirmation_required',
      detail: 'Resume requires {"confirm": true} in the body.',
    });
    return;
  }
  const state = writeEmergencyStop(false, null);
  res.json({ ok: true, emergency_stop: state });
});

// ── POST /auto-execute/enable ───────────────────────────────────────────────
executionRouter.post('/auto-execute/enable', (_req, res) => {
  const state = writeAutoExecute(true);
  res.json({ ok: true, auto_execute: state });
});

// ── POST /auto-execute/disable ──────────────────────────────────────────────
executionRouter.post('/auto-execute/disable', (_req, res) => {
  const state = writeAutoExecute(false);
  res.json({ ok: true, auto_execute: state });
});

// ── GET /auto-execute/status ─────────────────────────────────────────────────
executionRouter.get('/auto-execute/status', (_req, res) => {
  const runtime = _autoExecutorStatus?.() ?? null;
  res.json({
    enabled: readAutoExecute(),
    user_risk_level: readUserRiskLevel(),
    max_trade_size_sol: readUserMaxTradeSize(),
    // Live runtime state from the running AutoExecutor (null if not yet registered).
    wallet_mode:          runtime?.walletMode ?? null,
    last_signal_key:      runtime?.lastSignalKey ?? null,
    last_executed_at:     runtime?.lastExecutedAt ?? null,
    last_skipped_reason:  runtime?.lastSkippedReason ?? null,
    dedup_window_seconds: runtime?.dedupWindowSeconds ?? null,
    poll_interval_seconds: runtime?.pollIntervalSeconds ?? null,
  });
});

// ── POST /user-risk-level ───────────────────────────────────────────────────
// Persists the user's risk-tolerance preference so the backend AutoExecutor
// (paper/system wallet modes) can enforce it via the Guardian. The frontend
// Settings page posts here on save.
const UserRiskLevelSchema = z.object({
  level: z.enum(['LOW', 'MEDIUM', 'HIGH']),
});

executionRouter.post('/user-risk-level', (req, res) => {
  const parsed = UserRiskLevelSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({
      error: 'Invalid request body',
      issues: parsed.error.issues.map((i) => ({ path: i.path.join('.'), message: i.message })),
    });
    return;
  }
  const state = writeUserRiskLevel(parsed.data.level);
  res.json({ ok: true, user_risk_level: state });
});

executionRouter.get('/user-risk-level', (_req, res) => {
  res.json({ level: readUserRiskLevel() });
});

// ── POST /max-trade-size ────────────────────────────────────────────────────
// Persists the user's per-trade size cap so the backend AutoExecutor uses
// the same value the user picked in Settings. Guardian's MAX_TRADE_SIZE_SOL
// remains the absolute server-side ceiling — submitting a value above it
// still gets BLOCKed by checkTradeSize.
const MaxTradeSizeSchema = z.object({
  amount_sol: z.number().positive().max(10),
});

executionRouter.post('/max-trade-size', (req, res) => {
  const parsed = MaxTradeSizeSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({
      error: 'Invalid request body',
      issues: parsed.error.issues.map((i) => ({ path: i.path.join('.'), message: i.message })),
    });
    return;
  }
  const state = writeUserMaxTradeSize(parsed.data.amount_sol);
  res.json({ ok: true, max_trade_size: state });
});

executionRouter.get('/max-trade-size', (_req, res) => {
  res.json({ amount_sol: readUserMaxTradeSize() });
});
