# Solana Agent Kit Plugin

**Status: Not yet implemented.** No code exists in this directory.

This directory is reserved for the Solana Agent Kit integration plugin
(Phase 6 of the roadmap).

## Planned capabilities

Once implemented, this plugin will provide the agent with:

- Keypair loading from environment variables (never hard-coded)
- Solana RPC connection pool
- Jupiter swap instruction builder (devnet first, mainnet after validation)
- Pre-trade simulation via `simulateTransaction` before signing
- Transaction broadcasting and confirmation monitoring

## Prerequisites before implementing

The following must be complete first:

1. Paper trading running stably for ≥30 days (done)
2. API layer exposing signals (Phase 2 — in progress)
3. Circuit breaker and risk checks in `services/execution/risk/` (Phase 6)
4. Devnet end-to-end test passing without real funds

## Do not implement yet

Real wallet key handling and mainnet transaction signing must not be built
until the devnet simulation path is stable and the risk checks are in place.
