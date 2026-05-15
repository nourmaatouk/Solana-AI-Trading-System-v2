# Agent Memory

**Status: Not yet implemented.** No code exists in this directory.

This directory is reserved for ChromaDB-based episodic memory (Phase 5).

## Planned

Once the agent layer is built, this module will:

- Store each fused trade decision as a vector embedding in ChromaDB
- Allow the LLM reasoning layer (Ollama Llama 3) to retrieve similar past
  decisions for contextual reasoning
- Persist trade rationale across sessions

## Prerequisites

- Working ElizaOS orchestrator integration
- ChromaDB client in `packages/db/chroma/`
- Defined embedding schema for trade decisions
