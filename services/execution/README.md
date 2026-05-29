# @trading/execution

Agentic execution layer for the Solana AI Trading System.

```
AI pipeline (Python) → /signal
                          ↓
              ExecutionWorkflow.execute(req)
                          ↓
             ┌────────────────────────┐
             │ GuardianValidator      │  ← deterministic, no LLM
             │ (signal + execution    │
             │  rules)                │
             └────────────────────────┘
                  approve │  block
                          ▼   ▼
             SolanaExecutionService    (BLOCK → no Solana call)
                          ↓
             paper / devnet / mainnet
                          ↓
              ElizaNarrator (post-decision, optional)
```

## What's here

| File | Purpose |
|------|---------|
| `src/config/env.ts` | zod-validated environment loader. Every threshold and flag lives here. |
| `src/guardian/policy.ts` | Pure rule predicates. |
| `src/guardian/validator.ts` | Composes rules in priority order. First failure wins. |
| `src/signalClient.ts` | HTTP client for the FastAPI `/signal` endpoint. |
| `src/solana/client.ts` | Lazy wrapper around Solana Agent Kit v2. |
| `src/solana/walletService.ts` | Read-only wallet (public key + balance). |
| `src/solana/modes/*.ts` | One executor per mode (paper / devnet / mainnet). |
| `src/solana/executionService.ts` | Picks the executor based on the Guardian-approved action. |
| `src/orchestrator/elizaRuntime.ts` | ElizaOS AgentRuntime (Anthropic provider). Post-decision narration only. |
| `src/orchestrator/executionWorkflow.ts` | Top-level workflow. |

## Tests

```bash
cd services/execution
npm install
npm test
```

## Install (after reviewing this package)

```bash
cd services/execution
npm install
npm run typecheck
npm test
```

Optional Solana / Eliza runtime dependencies (only needed for devnet/mainnet
or LLM narration):

```bash
npm install solana-agent-kit @solana-agent-kit/plugin-token \
  @elizaos/core @elizaos/plugin-anthropic @opentelemetry/api
```

These are declared in `optionalDependencies` so a paper-only deployment
doesn't pull them in.

### Enabling LLM narration

The orchestrator produces a human-readable narration **after** the Guardian
verdict is final — the LLM is never on the decision path. To turn it on:

```bash
ENABLE_ELIZA_NARRATION=true
ANTHROPIC_API_KEY=sk-ant-...            # required; never logged
ANTHROPIC_SMALL_MODEL=claude-3-5-haiku-20241022   # optional override
```

When the key or the optional Eliza packages are absent, narration falls back
to a deterministic stub (or, if `ENABLE_LLM_NARRATION_FALLBACK_STUB=false`, an
empty string). The narrator never throws.

## Hard-coding policy

There is **zero** hard-coded business logic in this package. Every
threshold, URL, key, and mode comes from `.env`. If you find a magic number
in any file under `src/`, that's a bug.
