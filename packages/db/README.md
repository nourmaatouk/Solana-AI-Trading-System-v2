# Database Package

Database clients and connection management.

## Structure

- **chroma/** - ChromaDB vector database client
- **redis/** - Redis client and caching
- **timescale/** - TimescaleDB client and migrations

## Purpose

Provides:
- Database connection pooling
- Query builders and ORM
- Migration management
- Connection lifecycle management

## Databases

- **TimescaleDB** - Time-series data (market data, analytics)
- **Redis** - Caching and session storage
- **ChromaDB** - Vector database for embeddings and similarity search
