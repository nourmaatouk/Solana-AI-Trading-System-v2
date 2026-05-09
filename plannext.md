# Plan Next

## Goal
Move the architecture from isolated model layers to a stable, end-to-end signal pipeline.

## Current State
- NLP has a shared event contract and a passing regression test.
- Spoofing detection has a Kaggle T4 notebook for TEN smoke testing.
- The architecture still needs a consistent contract across all Layer 2 outputs and a formal Layer 3 fusion step.

## Next Steps

### 1. Standardize Layer 2 contracts
Define one shared schema for outputs from:
- NLP sentiment processing
- Spoofing detection
- Market feature processing

Required fields should include:
- `timestamp`
- `source`
- `signal_type`
- `signal_score`
- `confidence`
- `label`
- `event_type`
- `event_signal`
- `high_impact_event`

### 2. Add adapters for each module
Create normalization helpers so each module emits the same contract before fusion.
- Normalize timestamps to UTC.
- Fill missing optional fields with safe defaults.
- Reject malformed rows early.

### 3. Build Layer 3 fusion logic
Add a fusion component that combines:
- TEN spoofing signals
- GNN spatial signals
- XGBoost directional signals
- FinBERT/NLP signals

Fusion should produce:
- one final decision
- a confidence score
- a reason/explanation payload

### 4. Add one end-to-end integration test
Create a test that runs:
- raw article or market input
- Layer 2 processing
- Layer 3 fusion
- final signal output

The test should verify that contract changes do not break downstream consumers.

### 5. Wire the Kaggle T4 notebook to the contract
Update the TEN Kaggle notebook so it consumes the same shared schema and can serve as a smoke test for the spoofing branch.

## Recommended Order
1. Lock the shared schema.
2. Add normalization helpers.
3. Implement fusion.
4. Add the end-to-end test.
5. Connect the Kaggle T4 workflow.

## Success Criteria
- All layer boundaries use one schema.
- Tests pass across NLP, spoofing, and fusion paths.
- The architecture can produce one stable final signal from multiple model outputs.
