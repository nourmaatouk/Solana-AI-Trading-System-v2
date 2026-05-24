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

## Anatomy of the character file

`character.trading-agent.json` follows the Eliza character schema. Each top-level
field shapes the runtime prompt that wraps the LLM:

| Field | What it does |
|-------|--------------|
| `name`, `username` | Identity used in chat transcripts and logs. |
| `modelProvider` | LLM backend (`openai`). Swappable without code changes. |
| `settings.voice.model` | `"none"` — text-only agent, no TTS. |
| `plugins` | Empty by design. **No plugins = no tool access** (no wallet, no RPC, no shell). This is the structural enforcement of the narration-only role. |
| `system` | Master instruction prepended to every conversation. Defines the four hard refusals: no decision authority, no key access, no Guardian override, no safety bypass. |
| `bio` | Short self-description injected into the prompt for consistent self-reference. |
| `lore` | Backstory facts (e.g. "has no tool access"). |
| `knowledge` | Factual claims about the system Sentinel is allowed to assert — Guardian authority, execution flow, default-disabled mainnet. |
| `messageExamples` | Few-shot examples that teach the model **how** to refuse. Includes a jailbreak attempt ("Bypass the Guardian") and an information request. |
| `style.all` / `style.chat` | Tone constraints applied everywhere / only in chat. Enforces plain, precise, 1–3 sentence answers with no emojis. |
| `topics` | Restricts the conversational scope to "trading explanation" and "safety policy". |
| `adjectives` | Persona biases woven into the prompt: safety-first, non-autonomous, transparent. |

The safety posture is enforced at two layers: **structurally** (empty `plugins`,
so the LLM physically cannot call wallet/RPC tools) and **prompt-wise** (`system`
+ `knowledge` + `messageExamples` train the refusal style). If you fork this
character, preserve both.

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
