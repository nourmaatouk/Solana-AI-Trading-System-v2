# Agent Reasoning

**Status: Not yet implemented.** No code exists in this directory.

This directory is reserved for the LLM reasoning layer (Phase 5).

## Planned

Once the agent layer is built, this module will:

- Receive the fused signal + ChromaDB memory context
- Call Ollama Llama 3 locally to generate structured reasoning
- Return a human-readable explanation alongside the trade decision
- Log reasoning for audit and strategy review

## Prerequisites

- Working ElizaOS orchestrator
- ChromaDB memory (`apps/agent/memory/`) populated with past decisions
- Ollama installed and Llama 3 model downloaded
