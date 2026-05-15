# Solana Smart Contracts

**Status: Not yet implemented.** No Rust or Anchor code exists in this directory.

All subdirectories (`programs/trader/src/`, `migrations/`, `tests/`) contain
only README placeholder files.

## Planned (Phase 6)

Once the execution layer is validated on devnet, this directory will contain
Anchor programs for optional on-chain risk management:

- Position tracking and emergency stop (circuit breaker)
- Fee collection
- Access control

## When to implement

Smart contracts are **not required** for the first version of on-chain execution.
The initial execution path will use Jupiter/Raydium via the Solana Agent Kit
(off-chain instruction building + signing).  Anchor contracts are an optional
hardening step for production.

## Prerequisites

- Devnet execution stable for ≥2 weeks
- `services/execution/` risk checks implemented
- Solana Agent Kit plugin complete
