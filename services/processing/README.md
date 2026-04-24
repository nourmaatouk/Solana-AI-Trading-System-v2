# Data Processing Service

Feature engineering and data transformation pipeline.

## Structure

- **features/** - Feature calculation and engineering
- **pipeline/** - Data processing pipeline orchestration

## Purpose

Transforms raw market data into ML-ready features:
- Technical indicator calculation
- Statistical features
- Microstructure features
- Sentiment features
- Network features

## Pipeline Stages

1. Raw data ingestion
2. Validation and cleaning
3. Feature engineering
4. Normalization and scaling
5. Feature aggregation
6. Storage for model inference

## Performance

Sub-millisecond feature calculation for real-time trading.
