# Agent Module

The agent module contains the orchestration character file used by ElizaOS and
documents how it relates to the deterministic execution layer.

## Files

| File | Purpose |
|------|---------|
| `character.trading-agent.json` | Eliza character definition for the orchestration agent ("Sentinel"). Loaded by the Eliza CLI if you run a standalone agent process. Also mirrored in `services/execution/src/orchestrator/character.ts` for the embedded narration path. |

## What this agent CAN do

- Read a Guardian verdict and execution result, and produce a 1–3 sentence
  human-readable narration.
- Refuse any request that asks it to bypass the Guardian.

## What this agent CANNOT do

- Decide whether to BUY, SELL, or HOLD. That comes from the AI pipeline
  (XGBoost + FinBERT + TEN-GNN + PPO fusion).
- Approve a transaction. The Guardian is the only authority.
- Execute a transaction. Only `services/execution`'s `SolanaExecutionService`
  can do that, and only with a positive Guardian verdict.
- Access private keys. The signing keypair never crosses the LLM boundary.

## How it fits in

```
                          ┌──────────────────────────┐
   AI pipeline (Python) ─►│  ExecutionWorkflow       │
                          │                          │
                          │  1. GuardianValidator    │  ← deterministic
                          │  2. SolanaExecutionSvc   │  ← Solana Agent Kit v2
                          │  3. ElizaNarrator        │  ← this character
                          └──────────────────────────┘
```

ElizaOS is consulted ONLY at step 3, AFTER the decision is final. It receives
the verdict + execution result. It never receives raw signals, raw user input,
or signing material.

## Running

For most deployments you do NOT need to start a standalone Eliza process — the
narration runtime is embedded inside `services/execution` and runs in-process
inside the Node backend.

To run a standalone Eliza agent (optional, for chat-style introspection):

```bash
# Requires bun + @elizaos/cli (see https://docs.elizaos.ai/)
bun i -g @elizaos/cli
elizaos start --character apps/agent/character.trading-agent.json
```
