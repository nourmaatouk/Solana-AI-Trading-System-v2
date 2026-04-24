# Trader Program

Main Anchor smart contract for trade execution and position management on Solana.

## Structure

- **src/** - Rust source code
  - **instructions/** - Program instructions/handlers
  - **state/** - Account state definitions and data structures

## Purpose

Implements:
- Trade execution logic
- Position opening and closing
- Leverage and margin calculations
- Fee collection
- Risk constraints and limits
- Event emission for off-chain indexing

## Key Instructions

- Initialize trader accounts
- Open positions
- Close positions
- Collect fees
- Update configuration
