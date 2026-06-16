/**
 * Guardian types — the deterministic safety gate sitting in front of every
 * Solana transaction. No LLM is allowed to influence these decisions.
 */

export type TradeAction = 'BUY' | 'SELL' | 'HOLD';
export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'UNKNOWN';
export type ExecutionMode = 'paper' | 'devnet';
export type WalletMode = 'paper' | 'system' | 'phantom';

/**
 * Input to the Guardian. Every field comes from a trusted upstream source
 * (the AI pipeline + the execution request). Free-text user input that could
 * carry prompt-injection material lives ONLY in `narration_request` and is
 * never read by Guardian logic.
 */
export interface GuardianInput {
  // Signal-level (from the AI pipeline)
  asset: string;                              // e.g. "SOL/USDT"
  action: TradeAction;
  confidence: number;                         // [0, 1]
  fused_score: number;                        // [-1, 1]
  risk_level: RiskLevel;
  spoofing_confidence: number;                // [0, 1]
  sentiment_relevance?: number;
  classification_confidence?: number;
  classification_probability_up?: number;
  signal_generated_at_utc?: string | null;    // ISO 8601

  // Execution-level (from the request)
  amount_sol: number;
  execution_mode: ExecutionMode;

  // Optional narration request (LLM input, NEVER read by Guardian)
  narration_request?: string;

  // Connected Phantom wallet public key (required when walletMode === 'phantom')
  phantom_public_key?: string;

  // Total SOL executed today across all confirmed trades. Passed by the
  // execution route so the Guardian can enforce MAX_DAILY_LOSS_SOL.
  daily_executed_sol?: number;

  // User's risk-tolerance preference, set in the frontend Settings page and
  // forwarded with every execution request. Falls back to policy.defaultUserRiskLevel.
  user_risk_level?: RiskLevel;
}

export type BlockReason =
  | 'EMERGENCY_STOP'
  | 'MISSING_FIELDS'
  | 'ACTION_HOLD'
  | 'ASSET_NOT_ALLOWED'
  | 'SIGNAL_STALE'
  | 'LOW_CONFIDENCE'
  | 'HIGH_SPOOFING'
  | 'TRADE_SIZE_EXCEEDED'
  | 'INVALID_AMOUNT'
  | 'PROMPT_OVERRIDE_ATTEMPT'
  | 'MISSING_PHANTOM_WALLET'
  | 'INVALID_PHANTOM_WALLET'
  | 'DAILY_LOSS_EXCEEDED'
  | 'USER_RISK_LEVEL_EXCEEDED';

export interface ApprovedAction {
  asset: string;          // base asset, e.g. "SOL"
  side: 'BUY' | 'SELL';
  amount_sol: number;
  execution_mode: ExecutionMode;
}

export interface ApprovedGuardianResult {
  approved: true;
  decision: 'APPROVE';
  reasons: string[];
  blocked_by: null;
  sanitized_action: ApprovedAction;
  evaluated_at_utc: string;
}

export interface BlockedGuardianResult {
  approved: false;
  decision: 'BLOCK';
  reasons: string[];
  blocked_by: BlockReason;
  sanitized_action: null;
  evaluated_at_utc: string;
}

export type GuardianResult = ApprovedGuardianResult | BlockedGuardianResult;

/**
 * Thresholds the Guardian reads at evaluation time. Pulled from env at
 * construction so test code can inject overrides without mutating process.env.
 */
export interface GuardianPolicy {
  confidenceThreshold: number;
  maxSpoofingConfidence: number;
  maxTradeSizeSol: number;
  maxDailyLossSol: number;
  signalMaxAgeSeconds: number;
  allowedAssets: string[];
  emergencyStop: boolean;
  walletMode: WalletMode;
  /** Server-side override: when false, HIGH-risk signals are always blocked
   *  even when the user has set their preference to HIGH. */
  allowHighRiskTrades: boolean;
  /** Fallback risk tolerance when the client omits user_risk_level. */
  defaultUserRiskLevel: RiskLevel;
}
