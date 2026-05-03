# 🏗️ Solana AI Trading: Model Expansion & Roadmap

This document outlines the strategic roadmap to evolve the current **XGBoost + NLP** pipeline into the target architecture involving **GNN Spatial Encoding**, **Social Sentiment Fusion**, and **RL-PPO Execution**.

## 📍 Current Status
*   **Classification**: High-alpha XGBoost (H4) with Triple-Barrier Labeling (64% Accuracy).
*   **NLP**: FinBERT Sentiment scoring of news (GDELT/RSS).
*   **LOB**: 15m aggregated Orderbook imbalance from Binance.

---

## 🚀 Phase 1: Social Sentiment Expansion
**Goal**: Solve the "Signal Sparsity" problem in news-based NLP.
*   **Action**: Implement a **Reddit API Scraper** targeting `r/solana`, `r/cryptocurrency`, and `r/solanameme`.
*   **Logic**: Use the existing FinBERT pipeline to score social "hype" vs "fear" to catch retail-driven moves before they hit the news.
*   **Value**: Increases sentiment feature coverage from ~30% to >80% of candles.

## 🚀 Phase 2: Hawkes Process (Volatility Excitement)
**Goal**: Model the internal "tempo" of the market.
*   **Action**: Implement a **Hawkes Process Maximum Likelihood Estimation (MLE)** script.
*   **Logic**: Calculate a "Market Intensity" score. Unlike standard volatility, Hawkes models how one trade triggers a cascade of others (self-excitation).
*   **Value**: Provides the AI with a "Market Tempo" feature to predict upcoming volatility spikes.

## 🚀 Phase 3: TEN-GNN Spatial Fusion
**Goal**: Capture the *shape* of the Orderbook, not just the imbalance.
*   **Action**: Integrate the **Temporal Encoding Network (TEN)** and **Graph Neural Network (GNN)** prototypes into the classification pipeline.
*   **Logic**: 
    1.  TEN encodes sub-minute orderbook snapshots.
    2.  GNN encodes the spatial relationship between price levels (Spoofing detection).
    3.  The **GNN Output Tensor** is fed into the XGBoost as a "Deep Alpha" feature.
*   **Value**: Catches institutional "Layering" and "Spoofing" that technical indicators miss.

## 🚀 Phase 4: RL-Decision Aggregator (PPO) 
**Goal**: Shift from "Predictions" to "Profits."
*   **Action**: Deploy a **Proximal Policy Optimization (PPO)** Reinforcement Learning agent.
*   **Logic**: The RL agent takes the fused signals (XGBoost + GNN + NLP) and decides:
    *   **Position Sizing**: How much leverage to use?
    *   **Dynamic Exit**: When to move the Stop Loss to break-even?
*   **Value**: Minimizes drawdowns and maximizes the "Profit Factor" of the system.

---

## 📈 Integration Checklist
- [x] Correct Path Resolution (WSL/Linux fix)
- [x] Stationary Z-Score Implementation
- [x] Triple-Barrier Method (+2.0% / -1.5%)
- [ ] Reddit API Real-Time Streamer
- [ ] Hawkes Intensity MLE Implementation
- [ ] GNN Feature-Extraction Layer
- [ ] RL Execution Orchestrator
