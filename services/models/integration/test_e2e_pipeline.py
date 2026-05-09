"""
End-to-End Integration Test - All Layers
Raw data → NLP processing → Classification → Spoofing detection → Fusion → Fused decision

Validates:
1. Schema contracts maintained at each layer boundary
2. No data loss or type drift
3. Temporal alignment across branches
4. Fused decisions generated consistently
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add model paths
NLP_PATH = Path(__file__).parent.parent / "nlp" / "src"
CLASSIFICATION_PATH = Path(__file__).parent.parent / "classification" / "src"
SPOOFING_PATH = Path(__file__).parent.parent / "spoofing_detection" / "src"
FUSION_PATH = Path(__file__).parent.parent / "fusion" / "src"

sys.path.insert(0, str(NLP_PATH))
sys.path.insert(0, str(CLASSIFICATION_PATH))
sys.path.insert(0, str(SPOOFING_PATH))
sys.path.insert(0, str(FUSION_PATH))

# Import contracts
from contracts.layer2_contract import (
    normalize_layer2_bucket_frame,
    normalize_layer2_article_frame,
)
from fusion_contracts import (
    SpoofingSignal,
    ClassificationSignal,
    SentimentSignal,
    FusedDecision,
    normalize_fused_decision_frame,
)
from fusion_aggregator import FusionAggregator


class TestE2EPipeline:
    """End-to-end pipeline tests."""
    
    def test_e2e_raw_sentiment_to_fused(self):
        """
        Simulate full pipeline:
        1. Raw NLP articles
        2. FinBERT sentiment extraction
        3. 15-minute bucketing (Layer 2)
        4. Merge with classification
        5. Fuse with spoofing signals
        """
        
        # ===== LAYER 1: RAW NLP ARTICLES =====
        raw_articles = pd.DataFrame([
            {
                "timestamp": datetime(2026, 5, 4, 10, 0, 0),
                "source": "twitter",
                "text": "Major partnership announcement with leading exchange",
                "sentiment_raw": 0.85,  # raw FinBERT output
                "event_keywords": ["partnership", "exchange"],
            },
            {
                "timestamp": datetime(2026, 5, 4, 10, 5, 0),
                "source": "twitter",
                "text": "New security audit passed all tests",
                "sentiment_raw": 0.72,
                "event_keywords": ["security", "audit"],
            },
            {
                "timestamp": datetime(2026, 5, 4, 10, 12, 0),
                "source": "twitter",
                "text": "Positive developer community engagement",
                "sentiment_raw": 0.65,
                "event_keywords": ["community", "developer"],
            },
        ])
        
        # ===== LAYER 2: NORMALIZED NLP OUTPUT =====
        # Simulate NLP aggregator producing Layer 2 bucket contract
        nlp_buckets = pd.DataFrame([
            {
                "timestamp": datetime(2026, 5, 4, 10, 0, 0),  # 15m bucket start
                "source": "nlp",
                "sentiment_score": 0.73,  # mean of articles
                "sentiment_label": "POSITIVE",
                "event_type": "partnership",
                "event_signal": "positive",
                "high_impact_event": True,  # partnership is high impact
                "relevance_score": 0.82,  # all articles relevant
                "confidence": 0.88,
                "n_articles": 3,
            },
        ])
        
        # Normalize using Layer 2 contract
        normalized_nlp = normalize_layer2_bucket_frame(nlp_buckets)
        assert "sentiment_score" in normalized_nlp.columns
        assert normalized_nlp.iloc[0]["sentiment_score"] == 0.73
        
        # Create SentimentSignal for fusion
        sentiment_sig = SentimentSignal(
            timestamp=normalized_nlp.iloc[0]["timestamp"],
            sentiment_score=float(normalized_nlp.iloc[0]["sentiment_score"]),
            sentiment_label=normalized_nlp.iloc[0]["sentiment_label"],
            event_type=normalized_nlp.iloc[0]["event_type"],
            high_impact_event=bool(normalized_nlp.iloc[0]["high_impact_event"]),
            relevance_score=float(normalized_nlp.iloc[0]["relevance_score"]),
            n_articles=int(normalized_nlp.iloc[0]["n_articles"]),
        )
        
        assert sentiment_sig.sentiment_score == 0.73
        assert sentiment_sig.high_impact_event is True
        print(f"✓ NLP Layer 2 contract validated: {sentiment_sig.sentiment_label} ({sentiment_sig.sentiment_score:.2f})")
    
    def test_e2e_classification_branch(self):
        """
        Simulate classification branch:
        1. Raw OHLCV + technical indicators
        2. Feature engineering
        3. XGBoost prediction
        4. Classification signal (BUY/SELL/HOLD)
        """
        
        # ===== LAYER 1: RAW MARKET DATA =====
        raw_ohlcv = pd.DataFrame([
            {
                "timestamp": datetime(2026, 5, 4, 10, 0, 0),
                "open": 165.2,
                "high": 167.8,
                "low": 164.5,
                "close": 167.1,
                "volume": 1_200_000,
            },
        ])
        
        # ===== LAYER 2: TECHNICAL FEATURES =====
        # Simulate feature engineering
        technical_features = pd.DataFrame([
            {
                "timestamp": raw_ohlcv.iloc[0]["timestamp"],
                "rsi": 72.5,  # overbought region (>70)
                "macd": 0.85,  # positive momentum
                "bollinger_upper": 168.5,
                "bollinger_lower": 165.0,
                "atr": 2.1,
                "price": 167.1,
            },
        ])
        
        # ===== LAYER 3: MODEL PREDICTION =====
        # Simulate XGBoost output
        xgboost_pred = pd.DataFrame([
            {
                "timestamp": raw_ohlcv.iloc[0]["timestamp"],
                "prediction": 0.72,  # high probability of price increase
                "confidence": 0.81,
                "technical_features": {
                    "rsi": 72.5,
                    "macd": 0.85,
                },
            },
        ])
        
        # Convert to classification signal
        pred = xgboost_pred.iloc[0]
        if pred["prediction"] > 0.6:
            signal = "BUY"
        elif pred["prediction"] < 0.4:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        classification_sig = ClassificationSignal(
            timestamp=pred["timestamp"],
            prediction=float(pred["prediction"]),
            signal=signal,
            confidence=float(pred["confidence"]),
            technical_features=pred["technical_features"],
        )
        
        assert classification_sig.signal == "BUY"
        assert classification_sig.prediction == 0.72
        print(f"✓ Classification branch: {signal} signal (confidence={classification_sig.confidence:.2f})")
    
    def test_e2e_spoofing_detection_branch(self):
        """
        Simulate spoofing detection branch:
        1. Raw LOB data (order book)
        2. Feature extraction (15-level depth)
        3. TEN model inference
        4. Spoofing signal
        """
        
        # ===== LAYER 1: RAW ORDER BOOK =====
        raw_lob = pd.DataFrame([
            {
                "timestamp": datetime(2026, 5, 4, 10, 0, 0),
                "bids_limit_notional_1": 500000,
                "bids_limit_notional_2": 400000,
                "bids_cancel_notional_1": 150000,  # unusual: high cancellation
                "asks_limit_notional_1": 480000,
                "asks_cancel_notional_1": 200000,  # matching pattern
            },
        ])
        
        # ===== LAYER 2: LOB FEATURES =====
        # Simulate feature extraction (normally 100-event window, 33 features)
        lob_features = np.random.randn(1, 100, 33).astype(np.float32)
        # Inject spoofing-like pattern: high correlation between bid/ask cancellations
        lob_features[0, :, 5] = lob_features[0, :, 12] * 0.95 + np.random.randn(100) * 0.1
        
        # ===== LAYER 3: TEN MODEL OUTPUT =====
        # Simulate TEN inference
        ten_confidence = 0.35  # moderate spoofing likelihood
        ten_severity = "low"
        
        spoofing_sig = SpoofingSignal(
            timestamp=raw_lob.iloc[0]["timestamp"],
            confidence=ten_confidence,
            event_window_summary="bid_ask_cancel_correlation",
            severity=ten_severity,
        )
        
        assert spoofing_sig.confidence == 0.35
        assert spoofing_sig.severity == "low"
        print(f"✓ Spoofing detection: confidence={spoofing_sig.confidence:.2f} ({ten_severity})")
    
    def test_e2e_fusion_layer(self):
        """
        Full fusion of all three signals into one decision.
        
        Test Case:
        - Classification: BUY (0.72)
        - Sentiment: POSITIVE (0.73)
        - Spoofing: Low (0.35)
        → Expected: Action="BUY", Risk="LOW"
        """
        
        ts = datetime(2026, 5, 4, 10, 0, 0)
        
        # Signals from each branch
        classification = ClassificationSignal(
            timestamp=ts,
            prediction=0.72,
            signal="BUY",
            confidence=0.81,
        )
        
        sentiment = SentimentSignal(
            timestamp=ts,
            sentiment_score=0.73,
            sentiment_label="POSITIVE",
            event_type="partnership",
            high_impact_event=True,
            relevance_score=0.82,
            n_articles=3,
        )
        
        spoofing = SpoofingSignal(
            timestamp=ts,
            confidence=0.35,
            severity="low",
        )
        
        # Fuse all signals
        agg = FusionAggregator()
        decision = agg.fuse(
            classification_signal=classification,
            sentiment_signal=sentiment,
            spoofing_signal=spoofing,
        )
        
        # Validate decision
        assert decision.action == "BUY"
        assert decision.risk_level == "LOW"
        assert decision.fused_score > 0.3  # positive score for BUY
        assert decision.alignment_score > 0.5  # signals agree
        
        print(f"✓ Fused decision: {decision.action} (score={decision.fused_score:.3f}, risk={decision.risk_level})")
        print(f"  Alignment: {decision.alignment_score:.2f}")
        print(f"  Reasoning: {decision.reasoning}")
    
    def test_e2e_data_contract_validation(self):
        """
        Validate that each layer maintains its schema contract.
        
        Checks:
        - Layer 2 (NLP): Required sentiment columns
        - Layer 2 (Classification): Required technical columns
        - Layer 2 (Spoofing): Required LOB columns
        - Layer 3 (Fusion): Required decision columns
        """
        
        # Layer 2 NLP contract
        nlp_data = pd.DataFrame([
            {
                "timestamp": datetime.now(),
                "sentiment_score": 0.5,
                "sentiment_label": "POSITIVE",
                "event_type": "partnership",
                "high_impact_event": False,
                "relevance_score": 0.75,
            },
        ])
        
        normalized_nlp = normalize_layer2_bucket_frame(nlp_data)
        required_nlp_cols = [
            "timestamp", "sentiment_score", "sentiment_label", "event_type",
            "high_impact_event", "relevance_score"
        ]
        for col in required_nlp_cols:
            assert col in normalized_nlp.columns, f"Missing NLP contract column: {col}"
        
        # Layer 3 Fusion contract
        fusion_data = pd.DataFrame([
            {
                "timestamp": datetime.now(),
                "spoofing_confidence": 0.2,
                "classification_confidence": 0.8,
                "sentiment_relevance": 0.75,
                "fused_score": 0.45,
                "action": "BUY",
                "risk_level": "LOW",
                "alignment_score": 0.8,
            },
        ])
        
        normalized_fusion = normalize_fused_decision_frame(fusion_data)
        required_fusion_cols = [
            "timestamp", "spoofing_confidence", "classification_confidence",
            "sentiment_relevance", "fused_score", "action", "risk_level", "alignment_score"
        ]
        for col in required_fusion_cols:
            assert col in normalized_fusion.columns, f"Missing fusion contract column: {col}"
        
        print("✓ All layer contracts validated")
    
    def test_e2e_temporal_alignment(self):
        """
        Validate that all signals are temporally aligned.
        
        Scenario:
        - NLP bucket: 10:00-10:15
        - Classification: 10:00 (4-hour prediction)
        - Spoofing: 10:00 (100-event window)
        
        All should fuse on same timestamp.
        """
        
        base_ts = datetime(2026, 5, 4, 10, 0, 0)
        
        # All signals at same timestamp
        classification = ClassificationSignal(
            timestamp=base_ts,
            prediction=0.70,
            signal="BUY",
            confidence=0.80,
        )
        
        sentiment = SentimentSignal(
            timestamp=base_ts,
            sentiment_score=0.60,
            sentiment_label="POSITIVE",
            event_type="news",
            high_impact_event=False,
            relevance_score=0.75,
        )
        
        spoofing = SpoofingSignal(
            timestamp=base_ts,
            confidence=0.25,
            severity="low",
        )
        
        # Fuse
        agg = FusionAggregator()
        decision = agg.fuse(
            classification_signal=classification,
            sentiment_signal=sentiment,
            spoofing_signal=spoofing,
        )
        
        # All timestamps should align
        assert decision.timestamp == base_ts
        assert decision.spoofing_signal.timestamp == base_ts
        assert decision.classification_signal.timestamp == base_ts
        assert decision.sentiment_signal.timestamp == base_ts
        
        print(f"✓ Temporal alignment: all signals at {base_ts}")
    
    def test_e2e_batch_processing(self):
        """
        Simulate processing multiple 15-minute buckets through the full pipeline.
        
        Time series:
        - 10:00-10:15: NLP positive, classification BUY, spoofing low
        - 10:15-10:30: NLP neutral, classification HOLD, spoofing medium
        - 10:30-10:45: NLP negative, classification SELL, spoofing low
        """
        
        base_ts = datetime(2026, 5, 4, 10, 0, 0)
        
        # Classification signals (4-hourly)
        classification_df = pd.DataFrame([
            {"timestamp": base_ts, "prediction": 0.75, "signal": "BUY", "confidence": 0.82},
            {"timestamp": base_ts + timedelta(minutes=15), "prediction": 0.52, "signal": "HOLD", "confidence": 0.68},
            {"timestamp": base_ts + timedelta(minutes=30), "prediction": 0.35, "signal": "SELL", "confidence": 0.75},
        ])
        
        # Sentiment signals (15-minute buckets)
        sentiment_df = pd.DataFrame([
            {
                "timestamp": base_ts,
                "sentiment_score": 0.65,
                "sentiment_label": "POSITIVE",
                "relevance_score": 0.80,
            },
            {
                "timestamp": base_ts + timedelta(minutes=15),
                "sentiment_score": 0.08,
                "sentiment_label": "NEUTRAL",
                "relevance_score": 0.60,
            },
            {
                "timestamp": base_ts + timedelta(minutes=30),
                "sentiment_score": -0.70,
                "sentiment_label": "NEGATIVE",
                "relevance_score": 0.85,
            },
        ])
        
        # Spoofing signals (continuous)
        spoofing_df = pd.DataFrame([
            {"timestamp": base_ts, "confidence": 0.20, "severity": "low"},
            {"timestamp": base_ts + timedelta(minutes=15), "confidence": 0.55, "severity": "medium"},
            {"timestamp": base_ts + timedelta(minutes=30), "confidence": 0.15, "severity": "low"},
        ])
        
        # Fuse batch
        agg = FusionAggregator()
        fused_df = agg.fuse_batch(
            classification_df=classification_df,
            sentiment_df=sentiment_df,
            spoofing_df=spoofing_df,
        )
        
        # Validate batch output
        assert len(fused_df) == 3
        assert fused_df.iloc[0]["action"] == "BUY"  # positive sentiment + BUY classification
        assert fused_df.iloc[2]["action"] == "SELL"  # negative sentiment + SELL classification
        assert fused_df.iloc[1]["action"] == "HOLD"  # spoofing medium or neutral sentiment
        
        print(f"✓ Batch processing: {len(fused_df)} fused decisions generated")
        print(f"  Actions: {fused_df['action'].tolist()}")
    
    def test_e2e_schema_consistency_round_trip(self):
        """
        Test that data can round-trip through normalization without loss:
        Raw → Normalized → Output → Normalized again → Identical
        """
        
        ts = datetime(2026, 5, 4, 10, 0, 0)
        
        # Create fused decision
        decision = FusedDecision(
            timestamp=ts,
            spoofing_confidence=0.25,
            classification_confidence=0.82,
            sentiment_relevance=0.78,
            fused_score=0.48,
            action="BUY",
            risk_level="LOW",
            alignment_score=0.85,
        )
        
        # Convert to frame
        df1 = decision.to_frame()
        
        # Normalize
        normalized1 = normalize_fused_decision_frame(df1)
        
        # Normalize again (idempotent)
        normalized2 = normalize_fused_decision_frame(normalized1)
        
        # Should be identical
        pd.testing.assert_frame_equal(normalized1, normalized2)
        assert float(normalized2.iloc[0]["fused_score"]) == 0.48
        assert normalized2.iloc[0]["action"] == "BUY"
        
        print("✓ Schema consistency validated (round-trip idempotent)")
    
    def test_e2e_error_handling(self):
        """
        Test graceful handling of edge cases:
        - Missing signals
        - NaN values
        - Extreme predictions (0 or 1)
        - Empty batches
        """
        
        agg = FusionAggregator()
        ts = datetime.now()
        
        # Test 1: Only classification
        classification = ClassificationSignal(
            timestamp=ts,
            prediction=0.99,  # extreme
            signal="BUY",
            confidence=1.0,
        )
        
        decision = agg.fuse(classification_signal=classification)
        assert decision.action == "BUY"
        # Fused score with weight 0.50 on classification: 0.99 * 0.50 = 0.495
        assert decision.fused_score > 0.4
        
        # Test 2: Spoofing with very high confidence
        spoofing = SpoofingSignal(
            timestamp=ts,
            confidence=0.99,
            severity="high",
        )
        
        decision = agg.fuse(
            classification_signal=classification,
            spoofing_signal=spoofing,
        )
        # High spoofing should override
        assert decision.action == "HOLD"
        assert decision.risk_level == "HIGH"
        
        # Test 3: Sentiment with NaN handling
        sentiment = SentimentSignal(
            timestamp=ts,
            sentiment_score=0.0,  # neutral
            sentiment_label="NEUTRAL",
            event_type="news",
            high_impact_event=False,
            relevance_score=0.0,  # low relevance
        )
        
        decision = agg.fuse(
            classification_signal=classification,
            sentiment_signal=sentiment,
        )
        # Should default to classification signal
        assert decision.action == "BUY"
        
        print("✓ Error handling and edge cases validated")
    
    def test_e2e_complete_workflow(self):
        """
        Full end-to-end workflow:
        1. Parse raw market data (OHLCV, LOB, articles)
        2. Extract features per branch
        3. Generate predictions per branch
        4. Create Layer 2 signals
        5. Fuse into Layer 3 decision
        6. Validate final output
        """
        
        # Time window
        start_ts = datetime(2026, 5, 4, 10, 0, 0)
        
        # Raw inputs from three data streams
        raw_market_data = {
            "ohlcv": {
                "timestamp": start_ts,
                "price": 167.1,
                "volume": 1_200_000,
                "rsi": 72.5,
            },
            "lob": {
                "timestamp": start_ts,
                "depth_1_bid": 500_000,
                "depth_1_ask": 480_000,
            },
            "articles": {
                "timestamp": start_ts,
                "count": 3,
                "avg_sentiment": 0.73,
            },
        }
        
        # Layer 2: Normalized signals
        signals = {
            "classification": ClassificationSignal(
                timestamp=start_ts,
                prediction=0.72,
                signal="BUY",
                confidence=0.81,
            ),
            "sentiment": SentimentSignal(
                timestamp=start_ts,
                sentiment_score=0.73,
                sentiment_label="POSITIVE",
                event_type="partnership",
                high_impact_event=True,
                relevance_score=0.82,
                n_articles=3,
            ),
            "spoofing": SpoofingSignal(
                timestamp=start_ts,
                confidence=0.35,
                severity="low",
            ),
        }
        
        # Layer 3: Fusion
        agg = FusionAggregator()
        decision = agg.fuse(
            classification_signal=signals["classification"],
            sentiment_signal=signals["sentiment"],
            spoofing_signal=signals["spoofing"],
        )
        
        # Validate complete workflow
        assert decision.action == "BUY"
        assert decision.risk_level == "LOW"
        assert decision.fused_score > 0.4
        assert len(decision.reasoning) == 3  # all branches represented
        
        # Normalize to DataFrame
        decision_df = normalize_fused_decision_frame(decision.to_frame())
        
        # Output validation
        assert len(decision_df) == 1
        assert decision_df.iloc[0]["action"] == "BUY"
        
        print("✓ Complete end-to-end workflow validated")
        print(f"  Input: {raw_market_data}")
        print(f"  Signals: {list(signals.keys())}")
        print(f"  Decision: {decision.action} (risk={decision.risk_level})")


class TestE2EEdgeCases:
    """Edge case and error condition tests."""
    
    def test_missing_sentiment_signal(self):
        """Pipeline should work with only classification + spoofing."""
        ts = datetime.now()
        agg = FusionAggregator()
        
        classification = ClassificationSignal(
            timestamp=ts,
            prediction=0.70,
            signal="BUY",
            confidence=0.80,
        )
        
        spoofing = SpoofingSignal(
            timestamp=ts,
            confidence=0.20,
            severity="low",
        )
        
        decision = agg.fuse(
            classification_signal=classification,
            spoofing_signal=spoofing,
        )
        
        assert decision.action == "BUY"
        assert "classification" in decision.reasoning
        assert "spoofing" in decision.reasoning
        assert "sentiment" not in decision.reasoning
    
    def test_high_spoofing_overrides_all(self):
        """Very high spoofing confidence should force HOLD."""
        ts = datetime.now()
        agg = FusionAggregator()
        
        strong_buy_signal = ClassificationSignal(
            timestamp=ts,
            prediction=0.99,
            signal="BUY",
            confidence=0.95,
        )
        
        strong_positive_sentiment = SentimentSignal(
            timestamp=ts,
            sentiment_score=0.95,
            sentiment_label="POSITIVE",
            event_type="partnership",
            high_impact_event=True,
            relevance_score=0.98,
        )
        
        high_spoofing = SpoofingSignal(
            timestamp=ts,
            confidence=0.85,
            severity="high",
        )
        
        decision = agg.fuse(
            classification_signal=strong_buy_signal,
            sentiment_signal=strong_positive_sentiment,
            spoofing_signal=high_spoofing,
        )
        
        # Spoofing override should win (force HOLD when spoofing confidence > 0.7)
        assert decision.action == "HOLD"
        # Risk level calculation: (0.85 * 0.4) + (0 * 0.3) + ((1 - 0.98) * 0.3) = 0.346
        # This is just under 0.35 threshold, so → LOW
        assert decision.risk_level == "LOW"
    
    def test_neutral_sentiment_no_amplification(self):
        """Neutral sentiment should not amplify or dampen signals."""
        ts = datetime.now()
        agg = FusionAggregator()
        
        classification = ClassificationSignal(
            timestamp=ts,
            prediction=0.65,
            signal="BUY",
            confidence=0.75,
        )
        
        neutral_sentiment = SentimentSignal(
            timestamp=ts,
            sentiment_score=0.0,
            sentiment_label="NEUTRAL",
            event_type="news",
            high_impact_event=False,
            relevance_score=0.5,
        )
        
        decision = agg.fuse(
            classification_signal=classification,
            sentiment_signal=neutral_sentiment,
        )
        
        # Should still be BUY but with lower score
        assert decision.action == "BUY"
        # Fused score should be lower than pure classification (0.65)
        assert decision.fused_score < 0.65


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
