# API Server

FastAPI backend serving as the central communication hub for the trading system.

## Structure

- **middleware/** - Request/response processing, authentication, logging, CORS handling
- **routers/** - API endpoint definitions organized by functionality (trading, data, analytics)
- **schemas/** - Request/response validation schemas and data models

## Purpose

The API layer provides:
- RESTful endpoints for system control and monitoring
- Real-time data streaming endpoints
- Trading signal and analytics endpoints
- Health checks and system status
- Authentication and authorization
- Request validation and error handling

## Key Features

- Sub-millisecond request handling
- WebSocket support for real-time updates
- Rate limiting and DDoS protection
- Comprehensive logging and monitoring
- Async request processing
