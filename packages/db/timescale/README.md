# TimescaleDB Client

**Status: Not yet implemented.** No Python client code exists in this directory.

The NLP module (`services/models/nlp/src/storage/timescale_writer.py`) contains
a standalone TimescaleDB writer used by the sentiment pipeline.  This package
directory is reserved for a shared client wrapper when multiple services need
database access.

## When to implement

When the project moves beyond CSV-based storage and multiple services need
a shared TimescaleDB instance (likely Phase 3+).
