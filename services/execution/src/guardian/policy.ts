/**
 * Pure Guardian rule predicates. Each function returns null when the rule
 * passes, or a (BlockReason, human-readable string) pair when it fails.
 *
 * Importantly: every rule is a pure function of its arguments and policy
 * thresholds. There is no I/O, no clock read (timestamps are passed in), and
 * no LLM call. This is what makes the Guardian auditable and testable.
 */
import type {
  BlockReason,
  GuardianInput,
  GuardianPolicy,
  RiskLevel,
  TradeAction,
} from './types';

export type RuleFailure = { reason: BlockReason; message: string };

// ── Helpers ─────────────────────────────────────────────────────────────────

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function normaliseAsset(asset: string): string {
  return asset.trim().toUpperCase();
}

/**
 * Parses an ISO-8601 timestamp into ms-since-epoch.
 * Returns null on any parse failure so the caller treats it as "missing" rather
 * than crashing.
 */
export function parseUtc(raw: string | null | undefined): number | null {
  if (!raw) return null;
  const t = Date.parse(raw);
  return Number.isFinite(t) ? t : null;
}

// ── Individual rules (priority order matches validator.ts) ───────────────────

export function checkEmergencyStop(policy: GuardianPolicy): RuleFailure | null {
  if (policy.emergencyStop) {
    return {
      reason: 'EMERGENCY_STOP',
      message: 'Emergency stop is active — all execution is suspended.',
    };
  }
  return null;
}

export function checkRequiredFields(input: GuardianInput): RuleFailure | null {
  const missing: string[] = [];
  if (!input.asset || typeof input.asset !== 'string') missing.push('asset');
  if (!input.action) missing.push('action');
  if (!isFiniteNumber(input.confidence)) missing.push('confidence');
  if (!isFiniteNumber(input.spoofing_confidence)) missing.push('spoofing_confidence');
  if (!isFiniteNumber(input.amount_sol)) missing.push('amount_sol');
  if (!input.execution_mode) missing.push('execution_mode');

  if (missing.length > 0) {
    return {
      reason: 'MISSING_FIELDS',
      message: `Missing or invalid required fields: ${missing.join(', ')}`,
    };
  }
  return null;
}

export function checkActionHold(action: TradeAction): RuleFailure | null {
  if (action === 'HOLD') {
    return {
      reason: 'ACTION_HOLD',
      message: 'Action is HOLD — no execution needed.',
    };
  }
  return null;
}

export function checkAssetAllowed(
  asset: string,
  policy: GuardianPolicy,
): RuleFailure | null {
  const normalised = normaliseAsset(asset);
  const allowed = policy.allowedAssets.map(normaliseAsset);
  if (!allowed.includes(normalised)) {
    return {
      reason: 'ASSET_NOT_ALLOWED',
      message: `Asset ${normalised} is not in the allowed list (${allowed.join(', ')}).`,
    };
  }
  return null;
}

/**
 * Stale-signal check.
 *
 * @param signalGeneratedAtUtc ISO timestamp of when the signal was produced
 * @param nowMs current time in ms-since-epoch (passed in for determinism)
 * @param policy thresholds
 */
export function checkSignalFresh(
  signalGeneratedAtUtc: string | null | undefined,
  nowMs: number,
  policy: GuardianPolicy,
): RuleFailure | null {
  const ts = parseUtc(signalGeneratedAtUtc);
  if (ts === null) {
    // Missing or unparseable timestamp is treated as stale to fail safe.
    return {
      reason: 'SIGNAL_STALE',
      message: 'Signal timestamp is missing or unparseable.',
    };
  }
  const ageSec = (nowMs - ts) / 1000;
  if (ageSec > policy.signalMaxAgeSeconds) {
    return {
      reason: 'SIGNAL_STALE',
      message: `Signal is ${ageSec.toFixed(0)}s old (max ${policy.signalMaxAgeSeconds}s).`,
    };
  }
  return null;
}

export function checkConfidence(
  confidence: number,
  policy: GuardianPolicy,
): RuleFailure | null {
  if (confidence < policy.confidenceThreshold) {
    return {
      reason: 'LOW_CONFIDENCE',
      message: `Confidence ${confidence.toFixed(3)} below threshold ${policy.confidenceThreshold}.`,
    };
  }
  return null;
}

const RISK_ORDER: Record<string, number> = { LOW: 0, MEDIUM: 1, HIGH: 2, UNKNOWN: 99 };

/**
 * Compares the signal's risk_level against the user's preference.
 *
 * Rule cascade (first match wins):
 * 1. Missing / UNKNOWN signal risk → BLOCK (fail safe).
 * 2. HIGH signal + allowHighRiskTrades=false → BLOCK (server hard cap).
 * 3. Signal risk ordinal > user preference ordinal → BLOCK.
 * 4. Otherwise → pass.
 *
 * If the client omits user_risk_level, policy.defaultUserRiskLevel is used.
 */
export function checkUserRiskPreference(
  input: GuardianInput,
  policy: GuardianPolicy,
): RuleFailure | null {
  const signalRisk = String(input.risk_level ?? '').toUpperCase() as RiskLevel;

  if (!signalRisk || !(signalRisk in RISK_ORDER) || signalRisk === 'UNKNOWN') {
    return {
      reason: 'USER_RISK_LEVEL_EXCEEDED',
      message: `Signal risk level is "${input.risk_level ?? 'missing'}" — execution blocked to fail safe.`,
    };
  }

  if (signalRisk === 'HIGH' && !policy.allowHighRiskTrades) {
    return {
      reason: 'USER_RISK_LEVEL_EXCEEDED',
      message: 'Signal risk level HIGH is not permitted (ALLOW_HIGH_RISK_TRADES=false).',
    };
  }

  const userRisk = String(input.user_risk_level ?? policy.defaultUserRiskLevel).toUpperCase() as RiskLevel;
  if (!(userRisk in RISK_ORDER)) {
    return {
      reason: 'USER_RISK_LEVEL_EXCEEDED',
      message: `Invalid user risk level preference: "${userRisk}".`,
    };
  }

  if (RISK_ORDER[signalRisk] > RISK_ORDER[userRisk]) {
    return {
      reason: 'USER_RISK_LEVEL_EXCEEDED',
      message: `Signal risk level ${signalRisk} exceeds user preference ${userRisk}.`,
    };
  }

  return null;
}

export function checkSpoofing(
  spoofingConfidence: number,
  policy: GuardianPolicy,
): RuleFailure | null {
  if (spoofingConfidence > policy.maxSpoofingConfidence) {
    return {
      reason: 'HIGH_SPOOFING',
      message: `Spoofing confidence ${spoofingConfidence.toFixed(3)} exceeds ${policy.maxSpoofingConfidence}.`,
    };
  }
  return null;
}

export function checkTradeSize(
  amountSol: number,
  policy: GuardianPolicy,
): RuleFailure | null {
  if (!Number.isFinite(amountSol) || amountSol <= 0) {
    return {
      reason: 'INVALID_AMOUNT',
      message: `amount_sol must be a positive finite number (got ${amountSol}).`,
    };
  }
  if (amountSol > policy.maxTradeSizeSol) {
    return {
      reason: 'TRADE_SIZE_EXCEEDED',
      message: `Trade size ${amountSol} SOL exceeds max ${policy.maxTradeSizeSol} SOL.`,
    };
  }
  return null;
}

export function checkMainnetAllowed(
  executionMode: string,
  policy: GuardianPolicy,
): RuleFailure | null {
  if (executionMode === 'mainnet' && !policy.allowMainnetExecution) {
    return {
      reason: 'MAINNET_DISABLED',
      message: 'mainnet execution is disabled (ALLOW_MAINNET_EXECUTION=false).',
    };
  }
  return null;
}

/**
 * Defence against prompt-injection by way of the narration field.
 *
 * The Guardian doesn't *read* narration_request for decisions — but we still
 * scan it for known override sentinels and refuse the request entirely if any
 * are present, on the theory that a request containing such phrases is most
 * likely an attempt to bypass safety somewhere downstream.
 */
// Base58 Solana public keys are 32–44 characters (32 decoded bytes, base58-encoded).
const SOLANA_PUBKEY_RE = /^[1-9A-HJ-NP-Za-km-z]{32,44}$/;

/**
 * When WALLET_MODE=phantom the caller must supply a valid Phantom wallet public
 * key with every execution request.  The Guardian never signs with this key —
 * it only confirms that a real browser wallet was connected before the request
 * reached the execution layer.
 */
export function checkPhantomWallet(
  input: GuardianInput,
  policy: GuardianPolicy,
): RuleFailure | null {
  if (policy.walletMode !== 'phantom') return null;
  const pk = input.phantom_public_key?.trim() ?? '';
  if (!pk) {
    return {
      reason: 'MISSING_PHANTOM_WALLET',
      message: 'WALLET_MODE=phantom requires a connected Phantom wallet. No public key was provided.',
    };
  }
  if (!SOLANA_PUBKEY_RE.test(pk)) {
    return {
      reason: 'INVALID_PHANTOM_WALLET',
      message: `phantom_public_key has an invalid Solana base58 format ("${pk.slice(0, 10)}…").`,
    };
  }
  return null;
}

const PROMPT_OVERRIDE_PATTERNS: RegExp[] = [
  /ignore (the )?guardian/i,
  /bypass (the )?guardian/i,
  /skip (the )?safety/i,
  /override (the )?(safety|guardian|policy)/i,
  /disable (the )?safety/i,
  /you are now/i,
  /developer mode/i,
];

/**
 * Daily capital-at-risk ceiling. Blocks when the total SOL executed today
 * (across all confirmed trades) reaches or exceeds MAX_DAILY_LOSS_SOL.
 *
 * "Loss" here means capital deployed, not net P&L — we don't track prices.
 * This is a conservative interpretation: cap total SOL exposure per day.
 */
export function checkDailyLoss(
  input: GuardianInput,
  policy: GuardianPolicy,
): RuleFailure | null {
  const daily = input.daily_executed_sol ?? 0;
  if (daily >= policy.maxDailyLossSol) {
    return {
      reason: 'DAILY_LOSS_EXCEEDED',
      message: `Daily executed volume ${daily.toFixed(4)} SOL has reached the ${policy.maxDailyLossSol} SOL ceiling.`,
    };
  }
  return null;
}

export function checkPromptOverride(input: GuardianInput): RuleFailure | null {
  const haystack = input.narration_request ?? '';
  for (const pat of PROMPT_OVERRIDE_PATTERNS) {
    if (pat.test(haystack)) {
      return {
        reason: 'PROMPT_OVERRIDE_ATTEMPT',
        message: 'narration_request contains a safety-override pattern.',
      };
    }
  }
  return null;
}
