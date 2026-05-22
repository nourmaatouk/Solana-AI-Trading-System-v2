/**
 * Public exports for the execution layer.
 *
 * The Node backend and any other consumer should import only from this file.
 * Internal modules may change shape; this barrel is the stable boundary.
 */
export { loadEnv, resetEnvCache, isOnChainMode, type ExecutionEnv } from './config/env';

export { GuardianValidator } from './guardian/validator';
export type {
  GuardianInput,
  GuardianResult,
  GuardianPolicy,
  ApprovedAction,
  ApprovedGuardianResult,
  BlockedGuardianResult,
  BlockReason,
  TradeAction,
  RiskLevel,
  ExecutionMode,
} from './guardian/types';

export { SignalClient } from './signalClient';
export type { NormalisedSignal } from './signalClient';

export { SolanaExecutionService } from './solana/executionService';
export { WalletService, defaultWalletService } from './solana/walletService';
export type {
  ExecutionContext,
  ExecutionResult,
  ExecutionStatus,
  ITransactionExecutor,
  PreparedTransaction,
  WalletInfo,
} from './solana/types';

export { ExecutionWorkflow } from './orchestrator/executionWorkflow';
export type {
  ExecutionRequest,
  ExecutionWorkflowResult,
  PrepareTransactionResult,
} from './orchestrator/types';
export { ElizaNarrator } from './orchestrator/elizaRuntime';
export { TRADING_AGENT_CHARACTER } from './orchestrator/character';

export { log } from './logger';
