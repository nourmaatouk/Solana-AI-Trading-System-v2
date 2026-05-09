"""Fusion Layer 3 - Multi-model signal aggregation."""

from fusion_contracts import (
    SpoofingSignal,
    ClassificationSignal,
    SentimentSignal,
    GNNSignal,
    FusedDecision,
    normalize_fused_decision_frame,
)
from fusion_aggregator import FusionAggregator

__all__ = [
    "SpoofingSignal",
    "ClassificationSignal",
    "SentimentSignal",
    "GNNSignal",
    "FusedDecision",
    "normalize_fused_decision_frame",
    "FusionAggregator",
]
