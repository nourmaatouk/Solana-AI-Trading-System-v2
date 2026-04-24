# Data Directory

Storage for training data, model artifacts, and processed datasets.

## Structure

- **model_artifacts/** - Trained model files (GNN, PPO, XGBoost, FinBERT)
- **processed/** - Cleaned and feature-engineered datasets ready for training
- **raw/** - Raw market data and source data

## Purpose

Centralized storage for:
- Historical market data
- Feature engineering outputs
- Trained model weights and configurations
- Data for backtesting
- Analysis and visualization data

## Data Management

- Raw data is ingested from market sources
- Processed data contains cleaned and normalized features
- Model artifacts are versioned for reproducibility
