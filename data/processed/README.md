# Processed Data

Cleaned, normalized, and feature-engineered datasets ready for model training and inference.

## Purpose

Contains:
- Feature-engineered datasets with technical indicators
- Normalized price and volume data
- Derived features (ratios, moving averages, momentum)
- Aligned temporal data for training
- Test/validation splits

## Format

Data is typically stored as:
- Parquet files (columnar storage)
- NumPy arrays
- CSV with metadata
- HDF5 for large datasets

## Usage

Used for:
- Model training and validation
- Backtesting
- Real-time inference feature preparation
