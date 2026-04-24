# TimescaleDB Migrations

Database schema migration scripts.

## Purpose

Manages:
- Schema versioning
- Table and hypertable creation
- Index management
- Continuous aggregates setup
- Data transformations

## Usage

Migrations are run automatically during deployment to:
- Initialize database schema
- Apply schema updates
- Maintain database consistency

## Migration Tools

Uses migration tools (Alembic, Flyway, etc.) for:
- Tracking applied migrations
- Reversing migrations if needed
- Maintaining migration history
