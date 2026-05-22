# On-chain Solana Audit — Solvanta AI Trading System

**Date:** 2026-05-21  
**Scope:** Read-only. No files were modified.  
**Branch:** cleanup/phase-1-deletions  

---

## 1. Rust / Anchor Program Presence

**Verdict: Does NOT exist.**

No Rust/Anchor program is present anywhere in this repository.

| Item | Present |
|---|---|
| `Anchor.toml` | ❌ |
| `Cargo.toml` | ❌ |
| `programs/` directory | ❌ |
| `contracts/guardian_program/` | ❌ |
| `src/lib.rs` | ❌ |
| `#[program]` macro | ❌ |
| `anchor_lang` crate | ❌ |
| `declare_id!` macro | ❌ |
| Program ID constant | ❌ |
| IDL files (`.json`) | ❌ |
| `target/idl/` directory | ❌ |
| `target/deploy/` directory | ❌ |

The `open_issues.md` memory file (last updated 2026-05-19) lists Phase 6 — Solana Execution as entirely **pending**:

```
[ ] Build apps/agent/plugins/solana_agent_kit/
[ ] Build services/execution/rpc/
[ ] Build services/execution/risk/
[ ] Build services/execution/standard/ — Jupiter/Raydium swap (devnet first)
```

All on-chain items are future roadmap — none are implemented.

---

## 2. On-chain Guardian Logic

**Verdict: None exists.**

No Anchor instructions are implemented. All items from the target architecture are missing:

| Feature | Status |
|---|---|
| `initialize_config` | ❌ Missing |
| `update_config` | ❌ Missing |
| `set_emergency_stop` | ❌ Missing |
| `authorize_execution` | ❌ Missing |
| `mark_executed` | ❌ Missing |
| `GuardianConfig` PDA | ❌ Missing |
| `ExecutionRecord` PDA | ❌ Missing |
| `emergency_stop` on-chain | ❌ Missing |
| `max_trade_lamports` | ❌ Missing |
| `max_daily_loss_lamports` | ❌ Missing |
| `allowed_executor` | ❌ Missing |
| Replay protection / nonce | ❌ Missing |
| `signal_hash` | ❌ Missing |
| Signer validation (Anchor) | ❌ Missing |

The **off-chain** TypeScript Guardian (`services/execution/src/guardian/`) does implement emergency stop, max trade size, confidence threshold, spoofing threshold, and `WALLET_MODE=phantom` key validation — but this runs entirely in Node.js memory, not on-chain.

---

## 3. Build and Test Status

**Anchor/Rust toolchain:** Not installed. No Rust program exists to build.

**What does build cleanly:**

| Target | Status |
|---|---|
| `services/execution` (`tsc`) | ✅ 0 errors |
| Backend (`tsc --noEmit`) | ✅ 0 errors |
| Frontend (`tsc --noEmit`) | ✅ 0 errors |
| Guardian tests (Vitest) | ✅ 35/35 pass |

There is nothing to `anchor build` or `anchor test` because no Rust program exists.

---

## 4. Devnet Deployment Status

**Verdict: No program deployed. No program ID exists.**

| Item | Present |
|---|---|
| `declare_id!` anywhere in codebase | ❌ |
| `GUARDIAN_PROGRAM_ID` in `.env` | ❌ |
| `GUARDIAN_PROGRAM_ID` in `.env.example` | ❌ |
| `GUARDIAN_PROGRAM_ID` in any backend config | ❌ |
| `target/deploy/*.json` keypair files | ❌ |
| `Anchor.toml` with cluster configuration | ❌ |

The only Solana network configuration present:

```env
SOLANA_CLUSTER=devnet
SOLANA_RPC_URL=https://api.devnet.solana.com
```

These are used by the backend to check wallet balances and perform SOL self-transfers only.

---

## 5. Backend Integration

**Verdict: No Anchor client exists. Execution is TypeScript-only via `@solana/web3.js` self-transfer.**

`apps/trading-dashboard/backend/package.json` dependencies:

| Package | Present |
|---|---|
| `@solana/web3.js` | ✅ (basic RPC only) |
| `@coral-xyz/anchor` | ❌ |

No `AnchorProvider`, no `Program`, no IDL import, no `authorizeExecution`, no `program.methods` exist anywhere.

From `docs/agentic_execution_architecture.md`, Section 10 (verbatim):
> "DEX integration: the current devnet/mainnet executors perform a SOL self-transfer as a path-verification step. A real swap requires the `@solana-agent-kit/plugin-defi` plugin or Jupiter integration."

The `DevnetExecutor` and `MainnetExecutor` in `services/execution/src/solana/modes/` use `@solana/web3.js` to send a SOL self-transfer (wallet sends to itself) as a proof-of-connectivity test. No Anchor program interaction occurs.

Expected files that do **not** exist:

- `services/execution/src/solana/anchorGuardianClient.ts` — **MISSING**
- `services/execution/src/solana/onchainGuardianService.ts` — **MISSING**
- `services/execution/src/solana/transactionBuilder.ts` — **MISSING**

No `ENABLE_ONCHAIN_GUARDIAN` or `ONCHAIN_GUARDIAN_REQUIRED` flag exists anywhere.

---

## 6. Frontend Integration

**Verdict: No on-chain Guardian status is shown. All displayed data is off-chain.**

**What the frontend does display:**
- Off-chain Guardian APPROVE/BLOCK verdict (from TypeScript Guardian in Node.js)
- Execution mode badge (paper/devnet/mainnet) from backend `.env`
- Emergency stop status (persisted to `data/realtime/emergency_stop.json`, not on-chain)
- SOL balance (fetched from RPC via backend `/api/wallet/balance/:publicKey`)
- Wallet connection status (Phantom browser wallet)

**What the frontend does NOT display:**
- Guardian Program ID — **missing**
- Config PDA address — **missing**
- On-chain Guardian enabled/disabled toggle — **missing**
- Last authorization transaction signature — **missing**
- Program deployment status — **missing**
- Solana Explorer link for on-chain Guardian — **missing**

---

## 7. Phantom + On-chain Flow

**Verdict: Only steps 1–3 exist. Steps 4–9 do not exist.**

| Step | Description | Status |
|---|---|---|
| 1 | User connects Phantom wallet | ✅ Implemented (`WalletConnectionButton`, `WalletProvider`) |
| 2 | Backend gets latest AI signal | ✅ Implemented (`SignalClient` fetches from Python API) |
| 3 | Off-chain Guardian validates signal | ✅ Implemented (TypeScript `GuardianValidator`, 12 rules) |
| 4 | Backend calls on-chain Guardian `authorize_execution` | ❌ Missing — no Anchor program |
| 5 | Anchor program creates `ExecutionRecord` PDA | ❌ Missing — no Anchor program |
| 6 | Backend prepares unsigned transaction for frontend signing | ❌ Missing |
| 7 | Frontend asks Phantom to sign | ❌ Missing — wallet connected but `signTransaction` never called |
| 8 | Signed transaction submitted to Solana | ❌ Missing — only self-transfer via backend keypair |
| 9 | Dashboard shows on-chain tx signature | ❌ Missing — signature shown is from backend self-transfer |

Phantom is connected in the UI and its `publicKey` is sent to the backend as an identity claim. Phantom **never signs any transaction** in the current implementation. The backend holds `SOLANA_PRIVATE_KEY` and signs the self-transfer itself.

---

## 8. Hard-coded Value Audit

| File | Value | Type | Verdict |
|---|---|---|---|
| `services/execution/src/config/env.ts` | `https://api.devnet.solana.com` | config default | ✅ Safe — backed by `SOLANA_RPC_URL` env var |
| `services/execution/src/guardian/policy.ts` | `/^[1-9A-HJ-NP-Za-km-z]{32,44}$/` | base58 alphabet regex | ✅ Safe — fixed specification |
| `apps/trading-dashboard/backend/src/services/walletRpc.service.ts` | `https://api.devnet.solana.com` | config default | ✅ Safe — backed by `SOLANA_RPC_URL` env var |
| `services/execution/src/solana/modes/devnetExecutor.ts` | SOL self-transfer | placeholder | ⚠️ Known limitation — documented as pending in arch doc |
| `.env.example` | `SOLANA_RPC_URL=https://api.devnet.solana.com` | template default | ✅ Safe — template only |
| `.env.example` | `MAX_TRADE_SIZE_SOL=0.1` | config value | ✅ Safe — backed by env var |

No `declare_id!`, no Program ID strings, no hardcoded PDA seeds, and no hardcoded IDL paths were found anywhere. Their absence is a consequence of no on-chain program existing at all.

---

## 9. Executive Summary

The Solvanta AI Trading System's on-chain Solana component **does not exist**.

The project implements a sophisticated off-chain safety stack (TypeScript Guardian with 12 validation rules, paper/devnet/mainnet execution modes, Phantom wallet connection) and a complete AI pipeline (XGBoost + FinBERT + TEN-GNN + PPO fusion), but the Rust/Anchor Guardian program, the on-chain authorization flow, and Phantom transaction signing are entirely absent. The devnet executor performs a SOL self-transfer as a connectivity test only.

### Summary Table

| Area | Status | Missing | Build Priority |
|---|---|---|---|
| Rust/Anchor program | ❌ Does not exist | Entire program | P0 |
| `GuardianConfig` PDA | ❌ Does not exist | Full implementation | P0 |
| `ExecutionRecord` PDA | ❌ Does not exist | Full implementation | P0 |
| Devnet deployment | ❌ Not deployed | Build + deploy pipeline | P0 |
| Backend Anchor client | ❌ Does not exist | `anchorGuardianClient.ts` | P1 |
| Frontend on-chain status | ❌ Does not exist | UI components | P2 |
| Phantom signing flow | ❌ Steps 4–9 missing | Full tx signing flow | P1 |
| Hard-coded values | ✅ Clean | None dangerous found | — |

### Missing Parts (in build order)

1. Rust/Anchor program with `initialize_config`, `authorize_execution`, `mark_executed`, `set_emergency_stop` instructions
2. `GuardianConfig` and `ExecutionRecord` PDA account definitions
3. Program deployment to devnet → record Program ID
4. `GUARDIAN_PROGRAM_ID` in `.env`, `.env.example`, and backend config
5. `anchorGuardianClient.ts` — TypeScript client wrapping the Anchor program
6. `onchainGuardianService.ts` — calls `authorize_execution` before any trade
7. `ENABLE_ONCHAIN_GUARDIAN` / `ONCHAIN_GUARDIAN_REQUIRED` env flags with Guardian integration
8. Transaction builder — creates unsigned tx for frontend signing
9. Frontend `signTransaction` call via `useWallet()`
10. Frontend on-chain Guardian status panel showing PDA state, last authorization, Explorer link

### Recommended Build Sequence

1. **Build the Anchor program** in `contracts/guardian_program/` — start with `initialize_config` and `authorize_execution` only
2. **Deploy to devnet** — record the Program ID in `.env` as `GUARDIAN_PROGRAM_ID`
3. **Create `anchorGuardianClient.ts`** in `services/execution/src/solana/` using `@coral-xyz/anchor`
4. **Gate execution** on `ONCHAIN_GUARDIAN_REQUIRED=true` — only call `SolanaExecutionService` after on-chain authorization succeeds
5. **Implement Phantom signing** — backend builds the transaction, sends unsigned tx to frontend, Phantom signs, frontend submits
6. **Add on-chain status panel** to frontend showing Config PDA, last `ExecutionRecord`, Solana Explorer link

The note in `docs/agentic_execution_architecture.md` Section 10 accurately identifies the DEX integration gap but understates the scope — the entire on-chain Guardian layer is missing, not just the swap logic.

---

*Audit performed read-only. No source files were modified.*
