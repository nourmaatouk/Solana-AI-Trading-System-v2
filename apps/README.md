# Apps Layer

Contains the user-facing and orchestration applications for the Solana AI Trading System.

## Structure

- **agent/** - AI agent orchestration using ElizaOS for autonomous trading decisions
- **api/** - FastAPI backend server providing REST endpoints for trading operations, data retrieval, and system management
- **dashboard/** - React frontend for real-time monitoring, SHAP explainability visualization, and trading analytics

## Purpose

This layer provides the interface between the core trading system and end users/external systems. It handles:
- API request routing and validation
- Agent-based decision orchestration
- Real-time dashboard visualization
- Performance monitoring and logging
