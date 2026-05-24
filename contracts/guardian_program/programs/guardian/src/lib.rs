//! On-chain Guardian — minimal safety gate for the Solana AI Trading System.
//!
//! Mirrors the off-chain TypeScript Guardian (`services/execution/src/guardian/`)
//! on three rules that benefit from on-chain enforcement:
//!
//!   1. EMERGENCY_STOP — a kill-switch flipped by the admin, takes effect
//!      immediately at the validator level.
//!   2. MAX_TRADE_SIZE — per-trade cap in lamports.
//!   3. CONFIDENCE_THRESHOLD — minimum confidence (in basis points) required
//!      for the signal to authorize a trade.
//!
//! Replay protection is automatic: each `authorize_execution` call creates an
//! `ExecutionRecord` PDA seeded by the signal's SHA-256 hash. A second call
//! with the same signal_hash fails to `init` the account and the transaction
//! rolls back.
//!
//! This program is intended to be CPI-called by the off-chain backend in the
//! same transaction as the actual Jupiter swap, so the swap leg cannot be
//! reached unless `authorize_execution` succeeds.

use anchor_lang::prelude::*;

declare_id!("Gard1nGuard1anGuard1anGuard1anGuard1anGuard1an1");

/// Basis-point divisor. 10_000 bps = 100 %.
pub const BPS_DIVISOR: u16 = 10_000;

#[program]
pub mod guardian {
    use super::*;

    /// One-time setup. Creates the `GuardianConfig` PDA and locks in the
    /// initial policy. The signer becomes the `admin` and the only account
    /// allowed to call `set_emergency_stop` or re-initialise.
    ///
    /// * `max_trade_lamports`     - Hard ceiling on `amount_lamports` passed to
    ///                              `authorize_execution`. Mirror of
    ///                              `MAX_TRADE_SIZE_SOL * LAMPORTS_PER_SOL`.
    /// * `max_daily_loss_lamports`- Currently informational (tracking lives
    ///                              off-chain in `daily_volume.json`); stored
    ///                              for future enforcement.
    /// * `confidence_threshold_bps` - 0–10_000. 6_900 ≈ off-chain
    ///                              `CONFIDENCE_THRESHOLD=0.69`.
    /// * `allowed_executor`       - Pubkey of the backend wallet (or Phantom
    ///                              key) that is permitted to call
    ///                              `authorize_execution`. All other signers
    ///                              are rejected.
    pub fn initialize_config(
        ctx: Context<InitializeConfig>,
        max_trade_lamports: u64,
        max_daily_loss_lamports: u64,
        confidence_threshold_bps: u16,
        allowed_executor: Pubkey,
    ) -> Result<()> {
        require!(confidence_threshold_bps <= BPS_DIVISOR, GuardianError::InvalidBps);
        require!(max_trade_lamports > 0, GuardianError::InvalidAmount);

        let config = &mut ctx.accounts.config;
        config.admin = ctx.accounts.admin.key();
        config.allowed_executor = allowed_executor;
        config.max_trade_lamports = max_trade_lamports;
        config.max_daily_loss_lamports = max_daily_loss_lamports;
        config.confidence_threshold_bps = confidence_threshold_bps;
        config.emergency_stop = false;
        config.bump = ctx.bumps.config;

        emit!(ConfigInitialized {
            admin: config.admin,
            allowed_executor: config.allowed_executor,
            max_trade_lamports,
            confidence_threshold_bps,
        });
        Ok(())
    }

    /// Admin-only kill-switch. Flips `emergency_stop` to the supplied value.
    /// All subsequent `authorize_execution` calls fail with `EmergencyStop`
    /// until the admin flips it back.
    pub fn set_emergency_stop(ctx: Context<SetEmergencyStop>, enabled: bool) -> Result<()> {
        let config = &mut ctx.accounts.config;
        config.emergency_stop = enabled;

        emit!(EmergencyStopSet { enabled });
        Ok(())
    }

    /// The core authorisation check. Called by the backend (or any CPI caller
    /// holding the `allowed_executor` signer) immediately before invoking the
    /// Jupiter swap instruction in the same transaction. Creates a fresh
    /// `ExecutionRecord` PDA seeded by `signal_hash` — replays automatically
    /// fail because the account already exists.
    ///
    /// All four checks run before any state is written. If any one fails the
    /// transaction rolls back and the swap leg is never reached.
    pub fn authorize_execution(
        ctx: Context<AuthorizeExecution>,
        amount_lamports: u64,
        confidence_bps: u16,
        signal_hash: [u8; 32],
    ) -> Result<()> {
        let config = &ctx.accounts.config;

        // Rule 1 — emergency stop.
        require!(!config.emergency_stop, GuardianError::EmergencyStop);

        // Rule 2 — executor must match the configured key.
        require_keys_eq!(
            ctx.accounts.executor.key(),
            config.allowed_executor,
            GuardianError::NotExecutor
        );

        // Rule 3 — amount within cap.
        require!(amount_lamports > 0, GuardianError::InvalidAmount);
        require!(
            amount_lamports <= config.max_trade_lamports,
            GuardianError::TradeSizeExceeded
        );

        // Rule 4 — confidence above threshold.
        require!(confidence_bps <= BPS_DIVISOR, GuardianError::InvalidBps);
        require!(
            confidence_bps >= config.confidence_threshold_bps,
            GuardianError::LowConfidence
        );

        // All rules pass — record the execution. The PDA seed is the signal
        // hash, so a second call with the same hash will fail to `init` the
        // account (replay protection).
        let record = &mut ctx.accounts.execution_record;
        let clock = Clock::get()?;
        record.signal_hash = signal_hash;
        record.executor = ctx.accounts.executor.key();
        record.amount_lamports = amount_lamports;
        record.confidence_bps = confidence_bps;
        record.executed_at = clock.unix_timestamp;
        record.bump = ctx.bumps.execution_record;

        emit!(ExecutionAuthorised {
            executor: record.executor,
            amount_lamports,
            confidence_bps,
            signal_hash,
        });
        Ok(())
    }
}

// ── Accounts ────────────────────────────────────────────────────────────────

#[derive(Accounts)]
pub struct InitializeConfig<'info> {
    #[account(
        init,
        payer = admin,
        space = 8 + GuardianConfig::INIT_SPACE,
        seeds = [GuardianConfig::SEED, admin.key().as_ref()],
        bump
    )]
    pub config: Account<'info, GuardianConfig>,

    #[account(mut)]
    pub admin: Signer<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct SetEmergencyStop<'info> {
    #[account(
        mut,
        seeds = [GuardianConfig::SEED, admin.key().as_ref()],
        bump = config.bump,
        has_one = admin @ GuardianError::NotAdmin,
    )]
    pub config: Account<'info, GuardianConfig>,

    pub admin: Signer<'info>,
}

#[derive(Accounts)]
#[instruction(amount_lamports: u64, confidence_bps: u16, signal_hash: [u8; 32])]
pub struct AuthorizeExecution<'info> {
    #[account(
        seeds = [GuardianConfig::SEED, config.admin.as_ref()],
        bump = config.bump,
    )]
    pub config: Account<'info, GuardianConfig>,

    #[account(
        init,
        payer = executor,
        space = 8 + ExecutionRecord::INIT_SPACE,
        seeds = [ExecutionRecord::SEED, &signal_hash],
        bump
    )]
    pub execution_record: Account<'info, ExecutionRecord>,

    #[account(mut)]
    pub executor: Signer<'info>,

    pub system_program: Program<'info, System>,
}

// ── State ───────────────────────────────────────────────────────────────────

#[account]
#[derive(InitSpace)]
pub struct GuardianConfig {
    pub admin: Pubkey,                  // 32
    pub allowed_executor: Pubkey,       // 32
    pub max_trade_lamports: u64,        // 8
    pub max_daily_loss_lamports: u64,   // 8
    pub confidence_threshold_bps: u16,  // 2
    pub emergency_stop: bool,           // 1
    pub bump: u8,                       // 1
}

impl GuardianConfig {
    pub const SEED: &'static [u8] = b"guardian-config";
}

#[account]
#[derive(InitSpace)]
pub struct ExecutionRecord {
    pub signal_hash: [u8; 32],          // 32
    pub executor: Pubkey,               // 32
    pub amount_lamports: u64,           // 8
    pub confidence_bps: u16,            // 2
    pub executed_at: i64,               // 8
    pub bump: u8,                       // 1
}

impl ExecutionRecord {
    pub const SEED: &'static [u8] = b"execution-record";
}

// ── Errors ──────────────────────────────────────────────────────────────────

#[error_code]
pub enum GuardianError {
    #[msg("Emergency stop is active.")]
    EmergencyStop,
    #[msg("Trade amount exceeds the configured maximum.")]
    TradeSizeExceeded,
    #[msg("Confidence is below the configured threshold.")]
    LowConfidence,
    #[msg("Only the admin can call this instruction.")]
    NotAdmin,
    #[msg("Only the configured executor can authorize a trade.")]
    NotExecutor,
    #[msg("Amount must be a positive non-zero value.")]
    InvalidAmount,
    #[msg("Basis-point value must be in [0, 10_000].")]
    InvalidBps,
}

// ── Events ──────────────────────────────────────────────────────────────────

#[event]
pub struct ConfigInitialized {
    pub admin: Pubkey,
    pub allowed_executor: Pubkey,
    pub max_trade_lamports: u64,
    pub confidence_threshold_bps: u16,
}

#[event]
pub struct EmergencyStopSet {
    pub enabled: bool,
}

#[event]
pub struct ExecutionAuthorised {
    pub executor: Pubkey,
    pub amount_lamports: u64,
    pub confidence_bps: u16,
    pub signal_hash: [u8; 32],
}
