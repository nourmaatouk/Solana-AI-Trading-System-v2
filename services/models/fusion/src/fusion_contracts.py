"""
Fusion Layer 3 Contracts
Defines shared schemas for model outputs and fused decision signals.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd


@dataclass
class SpoofingSignal:
    """Signal from TEN spoofing detection model."""
    timestamp: datetime
    confidence: float  # [0, 1] probability of spoofing detected
    event_window_summary: str = ""  # e.g., "high_volume_cancel_bid"
    severity: str = "low"  # "low", "medium", "high"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ClassificationSignal:
    """Signal from XGBoost classification model (4-hour trend)."""
    timestamp: datetime
    prediction: float  # probability of price increase in next 4h
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # model confidence [0, 1]
    technical_features: Dict = field(default_factory=dict)  # OHLCV indicators
    
    def to_dict(self):
        return asdict(self)


@dataclass
class SentimentSignal:
    """Signal from FinBERT NLP + aggregation."""
    timestamp: datetime
    sentiment_score: float  # [-1, 1] negative to positive
    sentiment_label: str  # "NEGATIVE", "NEUTRAL", "POSITIVE"
    event_type: str  # "hack", "outage", "partnership", etc.
    high_impact_event: bool
    relevance_score: float  # [0, 1] relevance to trading decision
    n_articles: int = 0  # number of articles in 15m window
    
    def to_dict(self):
        return asdict(self)


@dataclass
class GNNSignal:
    """Signal from Graph Neural Network (if available)."""
    timestamp: datetime
    anomaly_score: float  # [0, 1] graph anomaly detection
    community_sentiment: float  # [-1, 1] sentiment from related tokens
    liquidity_stress: float  # [0, 1] network-wide stress indicator
    
    def to_dict(self):
        return asdict(self)


@dataclass
class FusedDecision:
    """Unified fused decision from all four signal branches."""
    timestamp: datetime
    
    # Individual signal confidences
    spoofing_confidence: float
    classification_confidence: float
    sentiment_relevance: float
    
    # Derived decision
    fused_score: float  # [-1, 1] overall directional score (negative=SELL, positive=BUY)
    action: str  # "BUY", "SELL", "HOLD"
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    
    # Explainability
    reasoning: Dict = field(default_factory=dict)  # {spoofing: ..., classification: ..., sentiment: ..., gnn: ...}
    alignment_score: float = 0.0  # [0, 1] how well signals agree
    
    # Source tracking
    spoofing_signal: Optional[SpoofingSignal] = None
    classification_signal: Optional[ClassificationSignal] = None
    sentiment_signal: Optional[SentimentSignal] = None
    gnn_signal: Optional[GNNSignal] = None
    gnn_confidence: Optional[float] = None
    
    def to_dict(self):
        d = asdict(self)
        # Convert nested dataclass dicts
        if self.spoofing_signal:
            d["spoofing_signal"] = self.spoofing_signal.to_dict()
        if self.classification_signal:
            d["classification_signal"] = self.classification_signal.to_dict()
        if self.sentiment_signal:
            d["sentiment_signal"] = self.sentiment_signal.to_dict()
        if self.gnn_signal:
            d["gnn_signal"] = self.gnn_signal.to_dict()
        return d
    
    def to_frame(self) -> pd.DataFrame:
        """Convert to single-row DataFrame."""
        d = self.to_dict()
        # Flatten nested dicts for DataFrame
        flattened = {k: v for k, v in d.items() if not isinstance(v, dict)}
        return pd.DataFrame([flattened])


def normalize_fused_decision_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure fused decision DataFrame has all required columns with defaults.
    
    Args:
        df: DataFrame with fused decision signals
        
    Returns:
        DataFrame with normalized schema
    """
    required_cols = [
        "timestamp", "spoofing_confidence", "classification_confidence",
        "sentiment_relevance", "fused_score", "action", "risk_level", "alignment_score"
    ]
    
    for col in required_cols:
        if col not in df.columns:
            if col == "alignment_score":
                df[col] = 0.5
            elif col == "fused_score":
                df[col] = 0.0
            elif col == "action":
                df[col] = "HOLD"
            elif col == "risk_level":
                df[col] = "MEDIUM"
            else:
                df[col] = 0.0
    
    return df[required_cols]
