/**
 * Typed wrappers for the /api/execution and /api/guardian endpoints.
 *
 * Type shapes mirror the public types exported from @trading/execution. We
 * keep them inline rather than importing because the frontend doesn't share
 * a tsconfig with the execution package.
 */
import { API_TIMEOUT_GET_MS, API_TIMEOUT_POST_MS } from '../config';

const BASE = '/api';

export type ExecutionMode = 'paper' | 'devnet' | 'mainnet';
export type GuardianDecision = 'APPROVE' | 'BLOCK';

export interface GuardianResultDto {
  approved: boolean;
  decision: GuardianDecision;
  reasons: string[];
  blocked_by: string | null;
  sanitized_action: {
    asset: string;
    side: 'BUY' | 'SELL';
    amount_sol: number;
    execution_mode: ExecutionMode;
  } | null;
  evaluated_at_utc: string;
}

export interface ExecutionResultDto {
  status: 'simulated' | 'submitted' | 'confirmed' | 'failed';
  mode: ExecutionMode;
  signature: string | null;
  block_height: number | null;
  asset: string;
  side: 'BUY' | 'SELL';
  amount_sol: number;
  rpc_url: string | null;
  submitted_at_utc: string;
  confirmed_at_utc: string | null;
  error: string | null;
}

export interface ExecutionWorkflowDto {
  request_received_at_utc: string;
  request_completed_at_utc: string;
  guardian: GuardianResultDto;
  execution: ExecutionResultDto | null;
  narration: string | null;
}

export interface ExecutionStatusDto {
  execution_mode: ExecutionMode;
  allow_mainnet: boolean;
  require_guardian_approval: boolean;
  emergency_stop: {
    enabled: boolean;
    updated_at_utc: string;
    reason: string | null;
  };
  wallet: {
    public_key: string | null;
    cluster: string;
    rpc_url: string;
    signing_capable: boolean;
    balance_sol: number | null;
  };
  last_execution: ExecutionWorkflowDto | null;
}

export interface GuardianPolicyDto {
  policy: {
    confidenceThreshold: number;
    maxSpoofingConfidence: number;
    maxTradeSizeSol: number;
    signalMaxAgeSeconds: number;
    allowedAssets: string[];
    allowMainnetExecution: boolean;
    emergencyStop: boolean;
  };
  notes: string[];
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { signal: AbortSignal.timeout(API_TIMEOUT_GET_MS) });
  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(err.error ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(API_TIMEOUT_POST_MS),
  });
  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(err.error ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export interface ExecuteRequest {
  amount_sol: number;
  execution_mode?: ExecutionMode;
  narration_request?: string;
  /** Connected Phantom wallet public key (required when WALLET_MODE=phantom). */
  phantom_public_key?: string;
  /** User's risk-tolerance preference (LOW | MEDIUM | HIGH). Guardian falls
   *  back to DEFAULT_USER_RISK_LEVEL from env when omitted. */
  user_risk_level?: 'LOW' | 'MEDIUM' | 'HIGH';
}

// ── Wallet API DTOs ──────────────────────────────────────────────────────────

export interface WalletSessionDto {
  publicKey: string;
  cluster: string;
  timestamp: string;
}

export interface WalletBalanceDto {
  publicKey: string;
  balance_sol: number;
  cluster: string;
  rpc_url: string;
  timestamp: string;
}

export interface WalletAirdropDto {
  signature: string;
  amount_sol: number;
  cluster: string;
}

export interface PreparedTxDto {
  guardian: GuardianResultDto;
  /** base64 VersionedTransaction for Phantom signing. Null = Guardian blocked or paper mode. */
  transaction_base64: string | null;
  execution_mode: ExecutionMode;
  asset: string;
  side: 'BUY' | 'SELL';
  amount_sol: number;
}

export const fetchExecutionStatus = () => get<ExecutionStatusDto>('/execution/status');

export const registerWalletSession = (publicKey: string) =>
  post<WalletSessionDto>('/wallet/session', { publicKey });

export const fetchWalletBalance = (publicKey: string) =>
  get<WalletBalanceDto>(`/wallet/balance/${encodeURIComponent(publicKey)}`);

export const requestDevnetAirdrop = (publicKey: string, amount_sol?: number) =>
  post<WalletAirdropDto>('/wallet/airdrop-devnet', { publicKey, amount_sol });
export const fetchGuardianPolicy = () => get<GuardianPolicyDto>('/guardian/policy');

export const validateExecution = (req: ExecuteRequest) =>
  post<{ guardian: GuardianResultDto }>('/execution/validate', req);

export const executeTrade = (req: ExecuteRequest) =>
  post<ExecutionWorkflowDto>('/execution/execute', req);

/** Guardian-validate + return unsigned tx for Phantom signing. */
export const prepareTrade = (req: ExecuteRequest) =>
  post<PreparedTxDto>('/execution/prepare-tx', req);

export const enableEmergencyStop = (reason?: string) =>
  post<{ ok: boolean }>('/execution/emergency-stop', { reason });

export const resumeFromEmergencyStop = () =>
  post<{ ok: boolean }>('/execution/emergency-resume', { confirm: true });

export const enableAutoExecute = () =>
  post<{ ok: boolean; auto_execute: boolean }>('/execution/auto-execute/enable');

export const disableAutoExecute = () =>
  post<{ ok: boolean; auto_execute: boolean }>('/execution/auto-execute/disable');

export interface AutoExecuteStatusDto {
  enabled: boolean;
  user_risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | null;
  max_trade_size_sol: number | null;
}

export const fetchAutoExecuteStatus = () =>
  get<AutoExecuteStatusDto>('/execution/auto-execute/status');

export const setBackendUserRiskLevel = (level: 'LOW' | 'MEDIUM' | 'HIGH') =>
  post<{ ok: boolean }>('/execution/user-risk-level', { level });

export const setBackendMaxTradeSize = (amount_sol: number) =>
  post<{ ok: boolean }>('/execution/max-trade-size', { amount_sol });
