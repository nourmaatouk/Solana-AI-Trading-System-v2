# Standard Execution

Standard order execution routing and management.

## Purpose

Handles:
- Normal (non-HFT) order execution
- Multi-route liquidity aggregation
- Execution quality optimization
- Transaction construction
- Error handling and retries

## Routing

Routes trades through:
- Jupiter (aggregator)
- Raydium (AMM)
- Orca (AMM)
- OpenBook (DEX)
