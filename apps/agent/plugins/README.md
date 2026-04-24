# Agent Plugins

Extensible plugin system for agent capabilities and integrations.

## Structure

- **solana_agent_kit/** - Solana blockchain interaction plugin for wallet management, transaction signing, and on-chain operations

## Purpose

Plugins provide:
- Modular capability extension
- Integration with external systems (Solana, DEX, etc.)
- Tool discovery and registration
- Plugin lifecycle management

## Capabilities

Each plugin can provide:
- Trading tools (swap, price check, balance query)
- Chain interaction (wallet management, transaction signing)
- Data retrieval (market data, position info)
- Liquidity management (pool interaction, LP operations)
