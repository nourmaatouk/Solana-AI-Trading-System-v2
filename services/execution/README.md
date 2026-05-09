# Execution Services

**Status: Not yet implemented.**

No execution code exists. The full system currently stops at signal generation — it produces a directional prediction (UP / DOWN / NEUTRAL) but does not place any trades.

Before building this layer you need: paper trading validation, slippage and fee modelling, and risk sizing on top of the classifier output.

## Planned Sub-services

- `hft/` — Sub-millisecond Jito bundle execution (not yet built)
- `standard/` — Jupiter/Raydium/Orca order routing (not yet built)
- `risk/` — Pre-trade risk checks and circuit breakers (not yet built)
- `rpc/` — Solana RPC connection pool (not yet built)
