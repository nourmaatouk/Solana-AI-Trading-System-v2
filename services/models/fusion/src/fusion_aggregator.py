"""
Fusion Aggregator - Layer 3
Merges signals from TEN (spoofing), XGBoost (classification), FinBERT (sentiment), 
and GNN (graph) into unified fused decisions.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from fusion_contracts import (
    SpoofingSignal,
    ClassificationSignal,
    SentimentSignal,
    GNNSignal,
    FusedDecision,
    normalize_fused_decision_frame,
)

# Load fusion/config.py by absolute path so it never collides with any other
# module named 'config' that may already be on sys.path (e.g. classification/config.py).
_FUSION_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.py"
_spec = importlib.util.spec_from_file_location("fusion.config", _FUSION_CONFIG_PATH)
_fusion_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fusion_cfg)

WEIGHT_CLASSIFICATION   = _fusion_cfg.WEIGHT_CLASSIFICATION
WEIGHT_SENTIMENT        = _fusion_cfg.WEIGHT_SENTIMENT
WEIGHT_SPOOFING         = _fusion_cfg.WEIGHT_SPOOFING
WEIGHT_GNN              = _fusion_cfg.WEIGHT_GNN
BUY_THRESHOLD           = _fusion_cfg.BUY_THRESHOLD
SELL_THRESHOLD          = _fusion_cfg.SELL_THRESHOLD
SPOOFING_HOLD_THRESHOLD = _fusion_cfg.SPOOFING_HOLD_THRESHOLD
RISK_HIGH_THRESHOLD     = _fusion_cfg.RISK_HIGH_THRESHOLD
RISK_MEDIUM_THRESHOLD   = _fusion_cfg.RISK_MEDIUM_THRESHOLD
ALIGNMENT_STRONG_AGREE     = _fusion_cfg.ALIGNMENT_STRONG_AGREE
ALIGNMENT_STRONG_DISAGREE  = _fusion_cfg.ALIGNMENT_STRONG_DISAGREE
ALIGNMENT_NEUTRAL          = _fusion_cfg.ALIGNMENT_NEUTRAL


class FusionAggregator:
    """
    Aggregates multi-model signals into unified trade decisions.
    
    Fusion Strategy:
    1. Spoofing Detection (TEN): High confidence → HOLD (reduce risk)
    2. Classification (XGBoost): Primary trend signal (BUY/SELL/HOLD)
    3. Sentiment (FinBERT): Amplifier / de-risker (positive = amplify, negative = reduce)
    4. GNN (if available): Network-wide stress indicator
    
    Weights:
    - classification: 0.50 (primary signal)
    - sentiment: 0.25 (amplifier)
    - spoofing: 0.20 (risk filter)
    - gnn: 0.05 (secondary)
    """
    
    def __init__(
        self,
        weight_classification: float = WEIGHT_CLASSIFICATION,
        weight_sentiment: float = WEIGHT_SENTIMENT,
        weight_spoofing: float = WEIGHT_SPOOFING,
        weight_gnn: float = WEIGHT_GNN,
    ):
        """Initialize fusion aggregator with signal weights."""
        total = weight_classification + weight_sentiment + weight_spoofing + weight_gnn
        assert total <= 1.01, f"Weights sum to {total}, should be ≤ 1.0"
        
        self.weight_classification = weight_classification
        self.weight_sentiment = weight_sentiment
        self.weight_spoofing = weight_spoofing
        self.weight_gnn = weight_gnn
    
    def fuse(
        self,
        spoofing_signal: Optional[SpoofingSignal] = None,
        classification_signal: Optional[ClassificationSignal] = None,
        sentiment_signal: Optional[SentimentSignal] = None,
        gnn_signal: Optional[GNNSignal] = None,
    ) -> FusedDecision:
        """
        Fuse individual signals into one unified decision.
        
        Args:
            spoofing_signal: TEN model output
            classification_signal: XGBoost model output
            sentiment_signal: FinBERT + NLP aggregator output
            gnn_signal: GNN model output (optional)
            
        Returns:
            FusedDecision with combined action and reasoning
        """
        # Validate we have at least one signal
        signals = [
            spoofing_signal,
            classification_signal,
            sentiment_signal,
            gnn_signal,
        ]
        if not any(signals):
            raise ValueError("At least one signal required for fusion")
        
        # Use classification timestamp as primary (most frequent update)
        timestamp = (
            classification_signal.timestamp
            if classification_signal
            else sentiment_signal.timestamp
            if sentiment_signal
            else datetime.now()
        )
        
        # Initialize scores
        classification_score = 0.0
        classification_conf = 0.0
        sentiment_score = 0.0
        sentiment_rel = 0.0
        spoofing_conf = 0.0
        gnn_conf = 0.0
        
        reasoning = {}
        
        # === CLASSIFICATION SIGNAL (primary trend) ===
        if classification_signal:
            # Convert to [-1, 1] scale: SELL = -1, HOLD = 0, BUY = +1
            # Higher prediction + BUY → higher positive score
            # Higher prediction + SELL → higher negative score (stronger sell)
            if classification_signal.signal == "BUY":
                classification_score = classification_signal.prediction  # [0, 1]
            elif classification_signal.signal == "SELL":
                classification_score = -classification_signal.prediction  # [-1, 0]
            else:  # HOLD
                classification_score = 0.0
            
            classification_conf = classification_signal.confidence
            reasoning["classification"] = f"XGBoost {classification_signal.signal} (p={classification_signal.prediction:.3f})"
        
        # === SENTIMENT SIGNAL (amplifier/de-risker) ===
        if sentiment_signal:
            # Sentiment [-1, 1] amplifies or dampens trend
            sentiment_score = sentiment_signal.sentiment_score
            sentiment_rel = sentiment_signal.relevance_score
            
            if sentiment_signal.high_impact_event:
                sentiment_rel = min(1.0, sentiment_rel + 0.2)  # boost relevance
            
            reasoning["sentiment"] = (
                f"FinBERT {sentiment_signal.sentiment_label} "
                f"({sentiment_signal.sentiment_score:.3f}), event={sentiment_signal.event_type}"
            )
        
        # === SPOOFING SIGNAL (risk filter) ===
        if spoofing_signal:
            spoofing_conf = spoofing_signal.confidence
            reasoning["spoofing"] = f"TEN confidence={spoofing_signal.confidence:.3f} ({spoofing_signal.severity})"
        
        # === GNN SIGNAL (network stress) ===
        if gnn_signal:
            gnn_conf = gnn_signal.anomaly_score
            reasoning["gnn"] = (
                f"Graph anomaly={gnn_signal.anomaly_score:.3f}, "
                f"liquidity_stress={gnn_signal.liquidity_stress:.3f}"
            )
        
        # === FUSE SIGNALS ===
        fused_score = self._compute_fused_score(
            classification_score=classification_score,
            classification_conf=classification_conf,
            sentiment_score=sentiment_score,
            sentiment_rel=sentiment_rel,
            spoofing_conf=spoofing_conf,
            gnn_conf=gnn_conf,
        )
        
        # Determine action from fused score
        action = self._score_to_action(fused_score, spoofing_conf)
        risk_level = self._compute_risk_level(spoofing_conf, gnn_conf, sentiment_rel)
        alignment_score = self._compute_alignment(
            classification_score, sentiment_score, spoofing_conf, gnn_conf
        )
        
        decision = FusedDecision(
            timestamp=timestamp,
            spoofing_confidence=spoofing_conf,
            classification_confidence=classification_conf,
            sentiment_relevance=sentiment_rel,
            gnn_confidence=gnn_conf,
            fused_score=fused_score,
            action=action,
            risk_level=risk_level,
            reasoning=reasoning,
            alignment_score=alignment_score,
            spoofing_signal=spoofing_signal,
            classification_signal=classification_signal,
            sentiment_signal=sentiment_signal,
            gnn_signal=gnn_signal,
        )
        
        return decision
    
    def fuse_batch(
        self,
        classification_df: Optional[pd.DataFrame] = None,
        sentiment_df: Optional[pd.DataFrame] = None,
        spoofing_df: Optional[pd.DataFrame] = None,
        gnn_df: Optional[pd.DataFrame] = None,
        join_key: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Fuse multiple batches of signals (join on timestamp, aggregate by time window).
        
        Args:
            classification_df: XGBoost predictions (columns: timestamp, prediction, signal, confidence)
            sentiment_df: Sentiment aggregations (columns: timestamp, sentiment_score, etc.)
            spoofing_df: Spoofing detections (columns: timestamp, confidence, severity, etc.)
            gnn_df: GNN outputs (columns: timestamp, anomaly_score, etc.)
            join_key: Temporal key for joining ("timestamp", "date", etc.)
            
        Returns:
            DataFrame of fused decisions with normalized schema
        """
        # Start with classification as base (most frequent)
        if classification_df is not None:
            result = classification_df.copy()
        elif sentiment_df is not None:
            result = sentiment_df.copy()
        else:
            raise ValueError("At least classification or sentiment DataFrame required")
        
        # Normalize timestamps to naive UTC for consistent joining
        if join_key in result.columns:
            result[join_key] = pd.to_datetime(result[join_key]).dt.tz_localize(None)
        
        # Left-join other signals on timestamp
        if sentiment_df is not None and classification_df is not None:
            sentiment_df = sentiment_df.copy()
            if join_key in sentiment_df.columns:
                sentiment_df[join_key] = pd.to_datetime(sentiment_df[join_key]).dt.tz_localize(None)
            
            result = result.merge(
                sentiment_df[[join_key, "sentiment_score", "sentiment_label", "relevance_score"]],
                on=join_key,
                how="left",
            )
        
        if spoofing_df is not None and len(result) > 0:
            spoofing_df = spoofing_df.copy()
            if join_key in spoofing_df.columns:
                spoofing_df[join_key] = pd.to_datetime(spoofing_df[join_key]).dt.tz_localize(None)
            
            result = result.merge(
                spoofing_df[[join_key, "confidence", "severity"]].rename(
                    columns={"confidence": "spoofing_confidence", "severity": "spoofing_severity"}
                ),
                on=join_key,
                how="left",
            )
        
        if gnn_df is not None and len(result) > 0:
            gnn_df = gnn_df.copy()
            if join_key in gnn_df.columns:
                gnn_df[join_key] = pd.to_datetime(gnn_df[join_key]).dt.tz_localize(None)
            
            result = result.merge(
                gnn_df[[join_key, "anomaly_score", "liquidity_stress"]],
                on=join_key,
                how="left",
            )
        
        # Compute fused scores row-by-row
        fused_rows = []
        for _, row in result.iterrows():
            try:
                classification_score = 0.0
                if pd.notna(row.get("prediction")):
                    if row.get("signal") == "BUY":
                        classification_score = float(row["prediction"])
                    elif row.get("signal") == "SELL":
                        classification_score = -float(row["prediction"])
                
                sentiment_score = float(row.get("sentiment_score", 0.0)) if pd.notna(row.get("sentiment_score")) else 0.0
                sentiment_rel = float(row.get("relevance_score", 0.0)) if pd.notna(row.get("relevance_score")) else 0.0
                spoofing_conf = float(row.get("spoofing_confidence", 0.0)) if pd.notna(row.get("spoofing_confidence")) else 0.0
                gnn_conf = float(row.get("anomaly_score", 0.0)) if pd.notna(row.get("anomaly_score")) else 0.0
                
                fused_score = self._compute_fused_score(
                    classification_score=classification_score,
                    classification_conf=float(row.get("confidence", 0.5)) if pd.notna(row.get("confidence")) else 0.5,
                    sentiment_score=sentiment_score,
                    sentiment_rel=sentiment_rel,
                    spoofing_conf=spoofing_conf,
                    gnn_conf=gnn_conf,
                )
                
                action = self._score_to_action(fused_score, spoofing_conf)
                risk_level = self._compute_risk_level(spoofing_conf, gnn_conf, sentiment_rel)
                alignment = self._compute_alignment(classification_score, sentiment_score, spoofing_conf, gnn_conf)
                
                fused_rows.append({
                    join_key: row[join_key],
                    "spoofing_confidence": spoofing_conf,
                    "classification_confidence": float(row.get("confidence", 0.5)) if pd.notna(row.get("confidence")) else 0.5,
                    "sentiment_relevance": sentiment_rel,
                    "fused_score": fused_score,
                    "action": action,
                    "risk_level": risk_level,
                    "alignment_score": alignment,
                })
            except Exception as e:
                # Log but continue
                print(f"Warning: error fusing row {row[join_key]}: {e}")
                continue
        
        if not fused_rows:
            return pd.DataFrame()
        
        fused_df = pd.DataFrame(fused_rows)
        return normalize_fused_decision_frame(fused_df)
    
    def _compute_fused_score(
        self,
        classification_score: float,
        classification_conf: float,
        sentiment_score: float,
        sentiment_rel: float,
        spoofing_conf: float,
        gnn_conf: float,
    ) -> float:
        """Compute [-1, 1] fused score from individual signals."""
        # Weighted average
        fused = (
            self.weight_classification * classification_score
            + self.weight_sentiment * (sentiment_score * sentiment_rel)
            + self.weight_spoofing * (-(spoofing_conf ** 2))  # spoofing reduces score
            + self.weight_gnn * (-(gnn_conf * 0.5))  # mild network stress penalty
        )
        
        return np.clip(fused, -1.0, 1.0)
    
    def _score_to_action(self, fused_score: float, spoofing_conf: float) -> str:
        """Convert fused score to BUY/SELL/HOLD action."""
        if spoofing_conf > SPOOFING_HOLD_THRESHOLD:
            return "HOLD"
        if fused_score > BUY_THRESHOLD:
            return "BUY"
        elif fused_score < -SELL_THRESHOLD:
            return "SELL"
        else:
            return "HOLD"

    def _compute_risk_level(
        self, spoofing_conf: float, gnn_conf: float, sentiment_rel: float
    ) -> str:
        """Compute risk level from combined risk factors."""
        risk_score = (spoofing_conf * 0.4) + (gnn_conf * 0.3) + ((1 - sentiment_rel) * 0.3)
        if risk_score > RISK_HIGH_THRESHOLD:
            return "HIGH"
        elif risk_score > RISK_MEDIUM_THRESHOLD:
            return "MEDIUM"
        else:
            return "LOW"

    def _compute_alignment(
        self, classification: float, sentiment: float, spoofing: float, gnn: float
    ) -> float:
        """Compute signal alignment score [0, 1]."""
        classification_sign = 1 if classification > 0.1 else (-1 if classification < -0.1 else 0)
        sentiment_sign = 1 if sentiment > 0.1 else (-1 if sentiment < -0.1 else 0)
        if classification_sign == 0 or sentiment_sign == 0:
            return ALIGNMENT_NEUTRAL
        if classification_sign == sentiment_sign:
            return ALIGNMENT_STRONG_AGREE
        else:
            return ALIGNMENT_STRONG_DISAGREE
