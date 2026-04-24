# Model Artifacts

Trained model files and weights for production and evaluation.

## Structure

- **gnn/** - Graph Neural Network model files
- **ppo/** - Proximal Policy Optimization reinforcement learning models
- **xgboost/** - XGBoost classification and regression models

## Purpose

Stores:
- Model weights and parameters
- Training configurations
- Tokenizers and preprocessors
- Model metadata (accuracy, metrics)
- Versioned checkpoints for model management

## Usage

Models are loaded at inference time for:
- Real-time trading signal generation
- Backtesting and simulation
- Performance evaluation
- A/B testing different model versions
