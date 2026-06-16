/**
 * GuardianValidator — composes the policy.ts rules in priority order and
 * returns a typed verdict.
 *
 * Design notes:
 * - First-failure-wins: rules are evaluated in a fixed order and the FIRST
 *   failing rule produces the BLOCK reason. This makes the verdict
 *   deterministic and easy to test ("why was it blocked? read the first
 *   rule that failed").
 * - The clock is injected so tests are reproducible.
 * - The policy is read from env once at construction (or passed in for tests).
 */
import { loadEnv, type ExecutionEnv } from '../config/env';
import {
  checkActionHold,
  checkAssetAllowed,
  checkConfidence,
  checkDailyLoss,
  checkEmergencyStop,
  checkPhantomWallet,
  checkPromptOverride,
  checkRequiredFields,
  checkSignalFresh,
  checkSpoofing,
  checkTradeSize,
  checkUserRiskPreference,
  type RuleFailure,
} from './policy';
import type {
  ApprovedAction,
  ExecutionMode,
  GuardianInput,
  GuardianPolicy,
  GuardianResult,
  WalletMode,
} from './types';

export interface ValidatorOptions {
  /** Inject a clock for deterministic tests. Defaults to Date.now. */
  now?: () => number;
  /** Optionally override policy values (test escape hatch). */
  policyOverrides?: Partial<GuardianPolicy>;
  /** Optionally inject env (test escape hatch). */
  env?: ExecutionEnv;
}

export class GuardianValidator {
  private readonly policy: GuardianPolicy;
  private readonly now: () => number;

  constructor(opts: ValidatorOptions = {}) {
    const env = opts.env ?? loadEnv();
    this.policy = {
      confidenceThreshold: env.CONFIDENCE_THRESHOLD,
      maxSpoofingConfidence: env.MAX_SPOOFING_CONFIDENCE,
      maxTradeSizeSol: env.MAX_TRADE_SIZE_SOL,
      maxDailyLossSol: env.MAX_DAILY_LOSS_SOL,
      signalMaxAgeSeconds: env.SIGNAL_MAX_AGE_SECONDS,
      allowedAssets: env.ALLOWED_ASSETS,
      emergencyStop: env.EMERGENCY_STOP,
      walletMode: (env.WALLET_MODE ?? 'phantom') as WalletMode,
      allowHighRiskTrades: env.ALLOW_HIGH_RISK_TRADES,
      defaultUserRiskLevel: env.DEFAULT_USER_RISK_LEVEL,
      ...opts.policyOverrides,
    };
    this.now = opts.now ?? (() => Date.now());
  }

  /** Read-only view of the active policy (safe to expose via the API). */
  describePolicy(): GuardianPolicy {
    return { ...this.policy };
  }

  validate(input: GuardianInput): GuardianResult {
    const checks: Array<() => RuleFailure | null> = [
      // Highest priority: kill-switch and prompt-injection. These run BEFORE
      // any input parsing so a malicious request cannot trick the validator
      // into reading any field at all.
      () => checkEmergencyStop(this.policy),
      () => checkPromptOverride(input),

      // Input shape — required so the remaining rules can trust numeric reads.
      () => checkRequiredFields(input),

      // Cheap semantic rules.
      () => checkActionHold(input.action),
      () => checkAssetAllowed(input.asset, this.policy),
      () => checkSignalFresh(input.signal_generated_at_utc, this.now(), this.policy),

      // Signal-quality rules (re-validation in case the upstream Python
      // Guardian was bypassed by a tampered HTTP payload).
      () => checkConfidence(input.confidence, this.policy),
      () => checkUserRiskPreference(input, this.policy),
      () => checkSpoofing(input.spoofing_confidence, this.policy),

      // Execution-level rules.
      () => checkTradeSize(input.amount_sol, this.policy),
      () => checkDailyLoss(input, this.policy),
      () => checkPhantomWallet(input, this.policy),
    ];

    for (const rule of checks) {
      const failure = rule();
      if (failure !== null) {
        return {
          approved: false,
          decision: 'BLOCK',
          reasons: [failure.message],
          blocked_by: failure.reason,
          sanitized_action: null,
          evaluated_at_utc: new Date(this.now()).toISOString(),
        };
      }
    }

    const action: ApprovedAction = {
      asset: input.asset.split('/')[0].toUpperCase(),
      side: input.action as 'BUY' | 'SELL',
      amount_sol: input.amount_sol,
      execution_mode: input.execution_mode as ExecutionMode,
    };

    return {
      approved: true,
      decision: 'APPROVE',
      reasons: [
        'All Guardian rules passed',
        `Confidence ${input.confidence.toFixed(3)} ≥ ${this.policy.confidenceThreshold}`,
        `Signal risk: ${input.risk_level} ≤ user preference ${input.user_risk_level ?? this.policy.defaultUserRiskLevel}`,
        `Spoofing: ${input.spoofing_confidence.toFixed(3)} ≤ ${this.policy.maxSpoofingConfidence}`,
        `Trade size: ${input.amount_sol} SOL ≤ ${this.policy.maxTradeSizeSol} SOL`,
        `Daily executed: ${(input.daily_executed_sol ?? 0).toFixed(4)} SOL / ${this.policy.maxDailyLossSol} SOL cap`,
        `Execution mode: ${input.execution_mode}`,
      ],
      blocked_by: null,
      sanitized_action: action,
      evaluated_at_utc: new Date(this.now()).toISOString(),
    };
  }

}
