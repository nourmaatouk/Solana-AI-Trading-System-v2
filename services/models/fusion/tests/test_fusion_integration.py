"""
Test Suite for Fusion Layer
Validates signal aggregation, fused decision generation, and integration with Layer 2 contracts.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fusion_contracts import (
    SpoofingSignal,
    ClassificationSignal,
    SentimentSignal,
    GNNSignal,
    FusedDecision,
    normalize_fused_decision_frame,
)
from fusion_aggregator import FusionAggregator


class TestFusionContracts:
    """Test fusion contract dataclasses."""
    
    def test_spoofing_signal_creation(self):
        sig = SpoofingSignal(
            timestamp=datetime.now(),
            confidence=0.85,
            event_window_summary="high_volume_cancel_bid",
            severity="high",
        )
        assert sig.confidence == 0.85
        assert sig.severity == "high"
        
        d = sig.to_dict()
        assert "timestamp" in d
        assert d["confidence"] == 0.85
    
    def test_classification_signal_creation(self):
        sig = ClassificationSignal(
            timestamp=datetime.now(),
            prediction=0.72,
            signal="BUY",
            confidence=0.81,
        )
        assert sig.signal == "BUY"
        assert sig.prediction == 0.72
        
        d = sig.to_dict()
        assert d["signal"] == "BUY"
    
    def test_sentiment_signal_creation(self):
        sig = SentimentSignal(
            timestamp=datetime.now(),
            sentiment_score=0.65,
            sentiment_label="POSITIVE",
            event_type="partnership",
            high_impact_event=True,
            relevance_score=0.88,
            n_articles=5,
        )
        assert sig.sentiment_label == "POSITIVE"
        assert sig.high_impact_event is True
    
    def test_gnn_signal_creation(self):
        sig = GNNSignal(
            timestamp=datetime.now(),
            anomaly_score=0.45,
            community_sentiment=0.22,
            liquidity_stress=0.15,
        )
        assert sig.anomaly_score == 0.45
        
    def test_fused_decision_creation(self):
        ts = datetime.now()
        decision = FusedDecision(
            timestamp=ts,
            spoofing_confidence=0.2,
            classification_confidence=0.81,
            sentiment_relevance=0.88,
            fused_score=0.45,
            action="BUY",
            risk_level="MEDIUM",
        )
        assert decision.action == "BUY"
        assert decision.risk_level == "MEDIUM"
        
        d = decision.to_dict()
        assert d["action"] == "BUY"
        assert d["timestamp"] == ts
    
    def test_fused_decision_with_source_signals(self):
        ts = datetime.now()
        class_sig = ClassificationSignal(
            timestamp=ts,
            prediction=0.7,
            signal="BUY",
            confidence=0.8,
        )
        
        decision = FusedDecision(
            timestamp=ts,
            spoofing_confidence=0.1,
            classification_confidence=0.8,
            sentiment_relevance=0.9,
            fused_score=0.5,
            action="BUY",
            risk_level="LOW",
            classification_signal=class_sig,
        )
        
        d = decision.to_dict()
        assert d["classification_signal"]["signal"] == "BUY"
    
    def test_fused_decision_to_frame(self):
        ts = datetime.now()
        decision = FusedDecision(
            timestamp=ts,
            spoofing_confidence=0.2,
            classification_confidence=0.81,
            sentiment_relevance=0.88,
            fused_score=0.45,
            action="BUY",
            risk_level="MEDIUM",
        )
        
        df = decision.to_frame()
        assert len(df) == 1
        assert df.iloc[0]["action"] == "BUY"
    
    def test_normalize_fused_decision_frame(self):
        df = pd.DataFrame([{
            "timestamp": datetime.now(),
            "spoofing_confidence": 0.2,
            "classification_confidence": 0.81,
            "sentiment_relevance": 0.88,
            "fused_score": 0.45,
            "action": "BUY",
            "risk_level": "MEDIUM",
            "alignment_score": 0.75,
        }])
        
        normalized = normalize_fused_decision_frame(df)
        assert len(normalized.columns) == 8
        assert all(col in normalized.columns for col in [
            "timestamp", "spoofing_confidence", "fused_score", "action"
        ])
    
    def test_normalize_fused_decision_frame_with_missing_columns(self):
        df = pd.DataFrame([{
            "timestamp": datetime.now(),
            "spoofing_confidence": 0.2,
            "action": "BUY",
        }])
        
        normalized = normalize_fused_decision_frame(df)
        assert "classification_confidence" in normalized.columns
        assert normalized.iloc[0]["classification_confidence"] == 0.0
        assert normalized.iloc[0]["alignment_score"] == 0.5


class TestFusionAggregator:
    """Test fusion aggregation logic."""
    
    def test_aggregator_initialization(self):
        agg = FusionAggregator(
            weight_classification=0.50,
            weight_sentiment=0.25,
            weight_spoofing=0.20,
            weight_gnn=0.05,
        )
        assert agg.weight_classification == 0.50
    
    def test_fuse_with_all_signals(self):
        ts = datetime.now()
        agg = FusionAggregator()
        
        spoofing = SpoofingSignal(timestamp=ts, confidence=0.15, severity="low")
        classification = ClassificationSignal(
            timestamp=ts, prediction=0.75, signal="BUY", confidence=0.85
        )
        sentiment = SentimentSignal(
            timestamp=ts,
            sentiment_score=0.60,
            sentiment_label="POSITIVE",
            event_type="partnership",
            high_impact_event=False,
            relevance_score=0.75,
        )
        gnn = GNNSignal(
            timestamp=ts,
            anomaly_score=0.25,
            community_sentiment=0.40,
            liquidity_stress=0.10,
        )
        
        decision = agg.fuse(
            spoofing_signal=spoofing,
            classification_signal=classification,
            sentiment_signal=sentiment,
            gnn_signal=gnn,
        )
        
        assert decision.action == "BUY"
        assert decision.risk_level == "LOW"
        assert decision.fused_score > 0.3  # positive score → BUY
        assert "classification" in decision.reasoning
        assert "sentiment" in decision.reasoning
    
    def test_fuse_only_classification(self):
        ts = datetime.now()
        agg = FusionAggregator()
        
        classification = ClassificationSignal(
            timestamp=ts, prediction=0.70, signal="SELL", confidence=0.80
        )
        
        decision = agg.fuse(classification_signal=classification)
        
        assert decision.action == "SELL"
        assert decision.fused_score < 0.0  # negative score
    
    def test_fuse_with_spoofing_override(self):
        """High spoofing confidence should force HOLD."""
        ts = datetime.now()
        agg = FusionAggregator()
        
        spoofing = SpoofingSignal(timestamp=ts, confidence=0.85, severity="high")
        classification = ClassificationSignal(
            timestamp=ts, prediction=0.90, signal="BUY", confidence=0.95
        )
        
        decision = agg.fuse(
            spoofing_signal=spoofing,
            classification_signal=classification,
        )
        
        # Spoofing confidence > 0.7 should force HOLD
        assert decision.action == "HOLD"
    
    def test_fuse_sentiment_amplification(self):
        """Positive sentiment should amplify BUY signal."""
        ts = datetime.now()
        agg = FusionAggregator()
        
        classification = ClassificationSignal(
            timestamp=ts, prediction=0.65, signal="BUY", confidence=0.75
        )
        sentiment_pos = SentimentSignal(
            timestamp=ts,
            sentiment_score=0.80,  # strong positive
            sentiment_label="POSITIVE",
            event_type="partnership",
            high_impact_event=True,
            relevance_score=0.90,
        )
        
        decision = agg.fuse(
            classification_signal=classification,
            sentiment_signal=sentiment_pos,
        )
        
        # Positive sentiment + BUY should have high fused score
        assert decision.fused_score > 0.5
        assert decision.action == "BUY"
    
    def test_fuse_sentiment_dampening(self):
        """Negative sentiment should dampen BUY signal."""
        ts = datetime.now()
        agg = FusionAggregator()
        
        classification = ClassificationSignal(
            timestamp=ts, prediction=0.65, signal="BUY", confidence=0.75
        )
        sentiment_neg = SentimentSignal(
            timestamp=ts,
            sentiment_score=-0.70,  # strong negative
            sentiment_label="NEGATIVE",
            event_type="hack",
            high_impact_event=True,
            relevance_score=0.95,
        )
        
        decision = agg.fuse(
            classification_signal=classification,
            sentiment_signal=sentiment_neg,
        )
        
        # Negative sentiment should reduce or flip BUY signal
        assert decision.fused_score < 0.3
    
    def test_error_without_signals(self):
        agg = FusionAggregator()
        with pytest.raises(ValueError, match="At least one signal required"):
            agg.fuse()
    
    def test_alignment_score_agreement(self):
        """Alignment should be high when signals agree."""
        ts = datetime.now()
        agg = FusionAggregator()
        
        # Both BUY signals
        classification = ClassificationSignal(
            timestamp=ts, prediction=0.75, signal="BUY", confidence=0.85
        )
        sentiment = SentimentSignal(
            timestamp=ts,
            sentiment_score=0.65,
            sentiment_label="POSITIVE",
            event_type="partnership",
            high_impact_event=False,
            relevance_score=0.80,
        )
        
        decision = agg.fuse(
            classification_signal=classification,
            sentiment_signal=sentiment,
        )
        
        # Should have high alignment
        assert decision.alignment_score > 0.5
    
    def test_alignment_score_disagreement(self):
        """Alignment should be low when signals disagree."""
        ts = datetime.now()
        agg = FusionAggregator()
        
        # BUY vs NEGATIVE
        classification = ClassificationSignal(
            timestamp=ts, prediction=0.75, signal="BUY", confidence=0.85
        )
        sentiment = SentimentSignal(
            timestamp=ts,
            sentiment_score=-0.75,
            sentiment_label="NEGATIVE",
            event_type="hack",
            high_impact_event=True,
            relevance_score=0.85,
        )
        
        decision = agg.fuse(
            classification_signal=classification,
            sentiment_signal=sentiment,
        )
        
        # Should have low alignment
        assert decision.alignment_score < 0.5


class TestFusionBatch:
    """Test batch fusion of DataFrames."""
    
    def test_fuse_batch_with_classification_and_sentiment(self):
        agg = FusionAggregator()
        
        ts_base = datetime.now()
        classification_df = pd.DataFrame([
            {
                "timestamp": ts_base,
                "prediction": 0.75,
                "signal": "BUY",
                "confidence": 0.85,
            },
            {
                "timestamp": ts_base + timedelta(hours=4),
                "prediction": 0.22,   # prob_up=0.22 → prob_down=0.78, genuine SELL
                "signal": "SELL",
                "confidence": 0.78,
            },
        ])
        
        sentiment_df = pd.DataFrame([
            {
                "timestamp": ts_base,
                "sentiment_score": 0.65,
                "sentiment_label": "POSITIVE",
                "relevance_score": 0.80,
            },
            {
                "timestamp": ts_base + timedelta(hours=4),
                "sentiment_score": -0.50,
                "sentiment_label": "NEGATIVE",
                "relevance_score": 0.75,
            },
        ])
        
        fused_df = agg.fuse_batch(
            classification_df=classification_df,
            sentiment_df=sentiment_df,
        )
        
        assert len(fused_df) == 2
        assert all(col in fused_df.columns for col in [
            "spoofing_confidence", "classification_confidence", "sentiment_relevance",
            "fused_score", "action", "risk_level", "alignment_score"
        ])
        
        # First row: BUY + POSITIVE → BUY
        assert fused_df.iloc[0]["action"] == "BUY"
        # Second row: SELL + NEGATIVE → SELL
        assert fused_df.iloc[1]["action"] == "SELL"
    
    def test_fuse_batch_with_spoofing(self):
        agg = FusionAggregator()
        
        ts_base = datetime.now()
        classification_df = pd.DataFrame([
            {
                "timestamp": ts_base,
                "prediction": 0.85,
                "signal": "BUY",
                "confidence": 0.90,
            },
        ])
        
        spoofing_df = pd.DataFrame([
            {
                "timestamp": ts_base,
                "confidence": 0.80,  # high spoofing
                "severity": "high",
            },
        ])
        
        fused_df = agg.fuse_batch(
            classification_df=classification_df,
            spoofing_df=spoofing_df,
        )
        
        assert len(fused_df) == 1
        # High spoofing should force HOLD
        assert fused_df.iloc[0]["action"] == "HOLD"
    
    def test_fuse_batch_partial_signals(self):
        """Batch with missing signals for some rows."""
        agg = FusionAggregator()
        
        ts_base = datetime.now()
        classification_df = pd.DataFrame([
            {"timestamp": ts_base, "prediction": 0.70, "signal": "BUY", "confidence": 0.80},
            {"timestamp": ts_base + timedelta(hours=4), "prediction": 0.55, "signal": "HOLD", "confidence": 0.65},
        ])
        
        sentiment_df = pd.DataFrame([
            {"timestamp": ts_base, "sentiment_score": 0.60, "sentiment_label": "POSITIVE", "relevance_score": 0.85},
            # Second row missing sentiment
        ])
        
        fused_df = agg.fuse_batch(
            classification_df=classification_df,
            sentiment_df=sentiment_df,
        )
        
        assert len(fused_df) >= 1
        # Second row should still have defaults filled
        if len(fused_df) > 1:
            assert fused_df.iloc[1]["sentiment_relevance"] == 0.0


class TestFusionIntegration:
    """Integration tests across multiple layers."""
    
    def test_fusion_with_layer2_contracts(self):
        """Verify fusion works with Layer 2 contract schemas."""
        # Import Layer 2 contracts
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "nlp" / "src" / "contracts"))
            from layer2_contract import BUCKET_CONTRACT, normalize_layer2_bucket_frame
        except ImportError:
            pytest.skip("Layer 2 contracts not available for integration test")
        
        # Simulate NLP bucket output
        nlp_bucket = pd.DataFrame([{
            "timestamp": datetime.now(),
            "sentiment_score": 0.65,
            "sentiment_label": "POSITIVE",
            "event_type": "partnership",
            "high_impact_event": True,
            "relevance_score": 0.88,
        }])
        
        normalized_nlp = normalize_layer2_bucket_frame(nlp_bucket)
        assert "sentiment_score" in normalized_nlp.columns
        
        # Can create sentiment signal from normalized data
        row = normalized_nlp.iloc[0]
        sentiment_sig = SentimentSignal(
            timestamp=row["timestamp"],
            sentiment_score=row["sentiment_score"],
            sentiment_label=row["sentiment_label"],
            event_type=row["event_type"],
            high_impact_event=row["high_impact_event"],
            relevance_score=row["relevance_score"],
        )
        
        assert sentiment_sig.sentiment_score == 0.65
    
    def test_end_to_end_minimal(self):
        """Minimal end-to-end: raw signals → fused decision → output frame."""
        ts = datetime.now()
        
        # Create signals
        classification = ClassificationSignal(
            timestamp=ts,
            prediction=0.72,
            signal="BUY",
            confidence=0.82,
        )
        
        sentiment = SentimentSignal(
            timestamp=ts,
            sentiment_score=0.55,
            sentiment_label="POSITIVE",
            event_type="partnership",
            high_impact_event=False,
            relevance_score=0.78,
        )
        
        # Fuse
        agg = FusionAggregator()
        decision = agg.fuse(
            classification_signal=classification,
            sentiment_signal=sentiment,
        )
        
        # Verify decision properties
        assert decision.timestamp == ts
        assert decision.action in ["BUY", "SELL", "HOLD"]
        assert 0 <= decision.fused_score <= 1 or -1 <= decision.fused_score <= 1
        assert 0 <= decision.alignment_score <= 1
        
        # Convert to frame
        df = decision.to_frame()
        assert len(df) == 1
        
        # Normalize
        normalized = normalize_fused_decision_frame(df)
        assert len(normalized) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
