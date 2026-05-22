# Agentic Execution Architecture

This document describes how the Solana AI Trading System integrates ElizaOS as
the orchestration layer and Solana Agent Kit v2 as the on-chain execution
toolkit, while keeping a deterministic Guardian agent as the sole authority
over trade approval.

## 1. Why this architecture

Trading systems that delegate decisions to an LLM have repeatedly demonstrated
two failure modes:

1. **Hallucinated trades** — the model is asked to "decide" and produces
   confident but unjustified outputs.
2. **Prompt-injection bypass** — user text reaches the model with the same
   trust level as system text, and the model can be convinced to circumvent
   safety logic.

Our design eliminates both by keeping the LLM **off the decision path**:

- The AI pipeline (XGBoost + FinBERT + TEN-GNN + PPO fusion) produces the
  trade decision deterministically from market data.
- The Guardian agent (rule-based, no LLM) is the only authority for
  approval. It re-validates every signal-level constraint AND adds
  execution-level constraints (size, mainnet flag, emergency stop, asset
  allowlist).
- The execution service (Solana Agent Kit v2) refuses to act without a
  positive Guardian verdict.
- ElizaOS is consulted ONLY after a decision is final. It receives the
  Guardian verdict + execution result and produces a human-readable
  narration. It has no tool access and cannot influence outcomes.

## 2. End-to-end flow

```
Market data + sentiment + order book
        │
        ▼
XGBoost  +  FinBERT  +  TEN-GNN  +  PPO  (services/models/*)
        │
        ▼
Fusion aggregator                        (services/models/fusion;
        │                                 PPO weight=0 in production)
        ▼
[ Python Guardian — paper-trader gate ]  (services/models/classification/guardian_agent.py)
        │
        ▼
last_fused_decision.json                 (read by FastAPI /signal)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ NODE BACKEND (apps/trading-dashboard/backend)                   │
│                                                                 │
│   SignalPoller polls FastAPI every SIGNAL_POLL_INTERVAL_MS      │
│   and emits 'signal_updated' on key change                      │
│                       │                                         │
│                       ▼                                         │
│   AutoExecutor (event-driven primary path)                      │
│     ├─ reads auto_execute.json, user_risk_level.json,           │
│     │  max_trade_size_sol.json, emergency_stop.json,            │
│     │  daily_volume.json                                        │
│     ├─ in-memory + persistent dedup (executed_signal_keys.json) │
│     ├─ WALLET_MODE=phantom → skip (frontend hook signs)         │
│     └─ WALLET_MODE=paper|system → workflow.execute()            │
│                       │                                         │
│                       ▼                                         │
│   ExecutionWorkflow (services/execution)                        │
│     │                                                           │
│     ├──► GuardianValidator (14 rules, first-failure-wins)       │
│     │           │                                               │
│     │           ├── BLOCK → log + return reason, no Solana call │
│     │           │                                               │
│     │           └── APPROVE                                     │
│     │                  │                                        │
│     ▼                  ▼                                        │
│  ElizaNarrator    SolanaExecutionService                        │
│  (optional,            │                                        │
│   post-decision)       ├── paper    → PaperExecutor (sim)       │
│                        ├── devnet   → DevnetExecutor (RPC)      │
│                        └── mainnet  → MainnetExecutor (gated)   │
│                                                                 │
│   ⇒ last_execution.json + execution_history.jsonl + WS broadcast│
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
React frontend (apps/trading-dashboard/frontend)
   ├─ ExecutionPanel: Validate / Execute buttons (manual)
   ├─ useAutoExecution hook (Phantom auto-signing, requires Arm toggle)
   └─ Settings → syncs user_risk_level + max_trade_size + auto_execute
                 to the backend on Save
```

### Two execution paths share the same Guardian

- **Manual** — the user types an amount on the Execution panel and clicks Validate or Execute. The frontend POSTs to `/api/execution/validate` or `/api/execution/execute` (paper/system) or `/api/execution/prepare-tx` followed by Phantom signing (phantom mode).
- **Auto** — the AutoExecutor fires when a new signal arrives. The user opts in via Settings (Trading Mode = `Auto`) which calls `POST /api/execution/auto-execute/enable`. In phantom mode the user must additionally click **Arm** in the Execution panel; auto-disarms after every successful trade.

In both paths the same `GuardianValidator` runs before any Solana call. The frontend cannot reach the executor without going through the Guardian.

## 3. Component responsibilities

### Python Guardian (`services/models/classification/guardian_agent.py`)

- Already exists and is wired into the paper trader.
- Validates signal-level constraints (confidence, risk_level, spoofing,
  PPO/Fusion conflict).
- Runs inside the AI pipeline.

### TypeScript Guardian (`services/execution/src/guardian/`)

- Defence in depth: re-validates signal-level rules so a tampered HTTP
  payload to `/api/execution/execute` cannot bypass them.
- Adds execution-level rules: trade size in SOL, mainnet flag, emergency
  stop, signal age, asset allowlist, prompt-override defence.
- Pure deterministic functions, no LLM.
- Returns a typed `GuardianResult` with first-failure-wins priority.

### Solana Execution Service (`services/execution/src/solana/`)

- Built on Solana Agent Kit v2 (lazy-loaded — paper-only deployments don't
  need the SDK installed).
- One executor per mode (`PaperExecutor`, `DevnetExecutor`, `MainnetExecutor`).
- Picks the executor based on the **Guardian-approved action's** mode, not
  the caller's request. A caller cannot trick the dispatcher by submitting
  `{mode: "mainnet"}` after the Guardian thought it validated a devnet
  request.
- All executors refuse to run when `EMERGENCY_STOP=true` even if reached by
  some other path.

### ElizaOS Orchestrator (`services/execution/src/orchestrator/`)

- Embedded as a library, not a separate process.
- Receives Guardian verdict + execution result; produces a 1–3 sentence
  narration.
- Falls back to a deterministic stub explainer when:
  - `ENABLE_ELIZA_NARRATION=false` (default), or
  - the `@elizaos/core` package isn't installed, or
  - the LLM call fails for any reason.
- The narration runtime never sees raw signals or the user's narration
  request string. Prompt-injection in user input cannot reach the LLM.

### Eliza Character (`apps/agent/character.trading-agent.json` and `services/execution/src/orchestrator/character.ts`)

- Named "Sentinel".
- System prompt explicitly states the agent has no trading authority.
- Empty `actions` array — Eliza has zero tools.
- Includes refusal examples for prompt-override requests.

### AutoExecutor (`apps/trading-dashboard/backend/src/services/autoExecutor.service.ts`)

The production entry point for backend-driven auto-execution. Sits between
SignalPoller and ExecutionWorkflow.

- **Two paths share the same safety chain:**
  - *Event-driven (primary):* subscribes to `SignalPoller.on('signal_updated', …)`.
    Fires on every fresh signal key. Latency ≈ one poll cycle (~10 s).
  - *Polling fallback:* every `AUTO_EXECUTOR_POLL_MS` (default 30 s), reads the
    poller's cached signal and runs the **same** dedup + Guardian chain. So
    even if an event is missed (race on startup, crashed listener), the
    fallback never re-executes a deduplicated signal.
- **Reads from disk on every signal** (no in-memory drift after Settings save):
  - `auto_execute.json` → enabled? (toggled by Settings)
  - `user_risk_level.json` → forwarded to Guardian as `user_risk_level`
  - `max_trade_size_sol.json` → forwarded as `amount_sol`
  - `emergency_stop.json` → instant abort
  - `daily_volume.json` → forwarded as `daily_executed_sol`
- **Persistent dedup** via `executed_signal_keys.json`:
  - Each successful trade appends `{signal_key, executed_at_utc, action, amount_sol}`.
  - On boot the in-memory cache is seeded from disk; entries outside
    `SIGNAL_DEDUP_WINDOW_SECONDS` are pruned on every write.
  - Survives crashes — a backend restart inside the dedup window cannot
    re-execute the same signal.
- **WALLET_MODE=phantom short-circuits the backend path entirely**: the
  AutoExecutor returns immediately without calling `workflow.execute()`. In
  that mode the frontend `useAutoExecution` hook owns auto-signing.

### Frontend auto-execution hook (`apps/trading-dashboard/frontend/src/hooks/useAutoExecution.ts`)

Owns the Phantom auto-signing flow.

- Watches the live signal from the Zustand store; same dedup-by-key logic as
  the backend.
- **Arm/disarm:** Phantom auto-signing requires an explicit `arm()` click in
  the Execution panel. Mounting the dashboard never triggers a wallet popup.
  Auto-disarms on (a) any successful trade, (b) wallet disconnect, (c) page
  reload.
- For Semi-Auto mode: auto-runs `/api/execution/validate` so the user sees the
  Guardian verdict before deciding to execute.
- For Auto + paper/system: skips with the reason "handled by backend
  AutoExecutor" (the backend path owns paper/system flows).

## 4. Safety rules (in priority order)

The Guardian evaluates rules in this order; the first failure produces the
BLOCK reason. Source of truth: `services/execution/src/guardian/validator.ts`.

| # | Rule | BlockReason | Source |
|---|------|-------------|--------|
| 1 | `EMERGENCY_STOP` active | `EMERGENCY_STOP` | env + persisted `emergency_stop.json` |
| 2 | Prompt-override sentinels in `narration_request` | `PROMPT_OVERRIDE_ATTEMPT` | input |
| 3 | Required fields present and finite | `MISSING_FIELDS` | input |
| 4 | `action == HOLD` | `ACTION_HOLD` | input |
| 5 | Asset in `ALLOWED_ASSETS` | `ASSET_NOT_ALLOWED` | env |
| 6 | Signal age ≤ `SIGNAL_MAX_AGE_SECONDS` | `SIGNAL_STALE` | env |
| 7 | `confidence ≥ CONFIDENCE_THRESHOLD` | `LOW_CONFIDENCE` | env |
| 8 | User-risk-preference enforcement (see below) | `USER_RISK_LEVEL_EXCEEDED` | env + input |
| 9 | `spoofing_confidence ≤ MAX_SPOOFING_CONFIDENCE` | `HIGH_SPOOFING` | env |
| 10 | `0 < amount_sol ≤ MAX_TRADE_SIZE_SOL` | `INVALID_AMOUNT` / `TRADE_SIZE_EXCEEDED` | env |
| 11 | mainnet requires `ALLOW_MAINNET_EXECUTION=true` | `MAINNET_DISABLED` | env |
| 12 | Daily executed SOL < `MAX_DAILY_LOSS_SOL` | `DAILY_LOSS_EXCEEDED` | env + persisted `daily_volume.json` |
| 13 | Phantom wallet present + valid base58 (when `WALLET_MODE=phantom`) | `MISSING_PHANTOM_WALLET` / `INVALID_PHANTOM_WALLET` | env + input |

### Rule 8 — user-risk-preference enforcement

Compares the signal's `risk_level` against the user's preference (`user_risk_level`
field, falling back to `DEFAULT_USER_RISK_LEVEL` from env when the client omits it):

- Missing or `UNKNOWN` signal risk → BLOCK (fail safe).
- `risk_level == HIGH` and `ALLOW_HIGH_RISK_TRADES=false` → BLOCK (server hard cap;
  the user cannot bypass).
- `risk_level` ordinal > user-preference ordinal → BLOCK (LOW=0, MEDIUM=1, HIGH=2).
- Otherwise pass.

## 5. HTTP API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/execution/validate` | Run Guardian only. No Solana calls. |
| POST | `/api/execution/execute` | Full workflow. Returns Guardian + execution result. |
| POST | `/api/execution/prepare-tx` | Guardian-validate + return base64 `VersionedTransaction` for Phantom signing. Null when blocked or in paper mode. |
| GET | `/api/execution/status` | Mode, wallet info, e-stop, last execution. |
| GET | `/api/execution/history` | Last 100 execution entries. |
| POST | `/api/execution/emergency-stop` | Enable persistent e-stop. Effective immediately. |
| POST | `/api/execution/emergency-resume` | Clear e-stop (requires `{confirm:true}`). |
| POST | `/api/execution/auto-execute/enable` | Set `auto_execute.json` to enabled. |
| POST | `/api/execution/auto-execute/disable` | Set `auto_execute.json` to disabled. |
| GET | `/api/execution/auto-execute/status` | Returns `{enabled, user_risk_level, max_trade_size_sol}`. |
| POST | `/api/execution/user-risk-level` | Body: `{level: 'LOW'\|'MEDIUM'\|'HIGH'}`. Persists to `user_risk_level.json`. |
| GET | `/api/execution/user-risk-level` | Returns the persisted level (or null). |
| POST | `/api/execution/max-trade-size` | Body: `{amount_sol: number}`. Persists to `max_trade_size_sol.json`. Capped at 10 SOL by zod schema. |
| GET | `/api/execution/max-trade-size` | Returns the persisted value (or null). |
| GET | `/api/guardian/policy` | Read-only view of active thresholds. |
| POST | `/api/wallet/session` | Register a Phantom public key for the current browser session. |
| GET | `/api/wallet/balance/:publicKey` | Read-only balance lookup. |
| POST | `/api/wallet/airdrop-devnet` | Devnet airdrop helper (capped by `MAX_DEVNET_AIRDROP_SOL`). |

## 6. Environment variables

All values are configurable. See `.env.example` for the canonical list. Key
ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `EXECUTION_MODE` | `paper` | `paper` / `devnet` / `mainnet` |
| `REQUIRE_GUARDIAN_APPROVAL` | `true` | Hard requirement |
| `ALLOW_MAINNET_EXECUTION` | `false` | Mainnet kill-switch |
| `EMERGENCY_STOP` | `false` | Persistent e-stop default |
| `ENABLE_ELIZA_NARRATION` | `false` | LLM narration opt-in |
| `SOLANA_CLUSTER` | `devnet` | Network |
| `SOLANA_RPC_URL` | `https://api.devnet.solana.com` | RPC endpoint |
| `SOLANA_PRIVATE_KEY` | (unset) | NEVER commit |
| `MAX_TRADE_SIZE_SOL` | `0.1` | Per-trade cap |
| `MAX_SPOOFING_CONFIDENCE` | `0.15` | Threshold |
| `SIGNAL_MAX_AGE_SECONDS` | `300` | Staleness limit |
| `CONFIDENCE_THRESHOLD` | `0.69` | Already used by AI pipeline |
| `ALLOWED_ASSETS` | `SOL/USDT` | Comma-separated |
| `MAX_DAILY_LOSS_SOL` | `0.5` | Daily exposure cap |
| `ALLOW_HIGH_RISK_TRADES` | `false` | Server hard cap on HIGH-risk signals |
| `DEFAULT_USER_RISK_LEVEL` | `LOW` | Fallback when client omits `user_risk_level` |
| `WALLET_MODE` | `phantom` | `paper` / `system` / `phantom` |
| `JUPITER_SWAP_ENABLED` | `false` | Real Jupiter v6 swap (mainnet program; see `.env.example`) |
| `AUTO_EXECUTOR_POLL_MS` | `30000` | AutoExecutor fallback poll interval |
| `AUTO_EXECUTE_AMOUNT_SOL` | `0.01` | Fallback per-trade size when `max_trade_size_sol.json` not set |
| `SIGNAL_DEDUP_WINDOW_SECONDS` | `300` | TTL for both in-memory and persistent dedup |

## 7. Running

### Paper mode (default — safe, no Solana network calls)

```bash
# Python API (existing)
uvicorn apps.api.main:app --reload --port 8000

# Node backend
cd apps/trading-dashboard/backend
npm install
npm run dev

# Frontend
cd apps/trading-dashboard/frontend
npm install
npm run dev
```

### Devnet mode

```bash
# 1. Install the Solana SDK in the execution package
cd services/execution
npm install solana-agent-kit @solana-agent-kit/plugin-token

# 2. Set up your devnet wallet
solana-keygen new --outfile ~/.config/solana/devnet-keypair.json
solana airdrop 1 -u devnet
# Convert the keypair to base58 and put it in .env as SOLANA_PRIVATE_KEY

# 3. Set in .env
# EXECUTION_MODE=devnet
# SOLANA_CLUSTER=devnet
# SOLANA_RPC_URL=https://api.devnet.solana.com

# 4. Restart the Node backend
```

### Eliza narration (optional)

```bash
cd services/execution
npm install @elizaos/core

# In .env
# ENABLE_ELIZA_NARRATION=true
# OPENAI_API_KEY=...
```

## 8. Tests

```bash
cd services/execution
npm install
npm test
```

The Guardian test suite covers every rule, every priority interaction, every
boundary value, and the prompt-injection defence. These tests are the
contract — if they fail, the rest of the execution layer must not be
deployed.

## 9. Security checklist (before any mainnet activity)

1. `SOLANA_PRIVATE_KEY` is in `.env` only. Confirm via `git check-ignore .env`.
2. `ALLOW_MAINNET_EXECUTION=true` is intentional, not accidental.
3. `MAX_TRADE_SIZE_SOL` is set to a value you can afford to lose to a bug.
4. Guardian tests pass: `cd services/execution && npm test`.
5. Frontend dashboard shows the correct mode badge before clicking Execute.
6. `EMERGENCY_STOP` toggle has been tested at least once.
7. Persistent emergency stop file lives at
   `services/models/classification/data/realtime/emergency_stop.json` and is
   tracked separately from `.env`.
8. The Eliza character file has been reviewed — its `actions` array must
   remain empty.
9. No log line ever contains the substring `SOLANA_PRIVATE_KEY` value
   (logger redacts by key name).

## 10. What is still pending

- DEX integration: the current devnet/mainnet executors perform a SOL
  self-transfer as a path-verification step. A real swap requires the
  `@solana-agent-kit/plugin-defi` plugin or Jupiter integration.
- Wallet exposure tracking: `MAX_DAILY_LOSS_SOL` is wired into the env but
  not yet enforced because we need cumulative state tracking. Add it before
  enabling devnet/mainnet for real.
- Frontend deep-link from a signal → "Execute this signal" with a
  pre-filled amount.
