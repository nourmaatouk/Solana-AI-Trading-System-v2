"""
config.py — Single source of truth for fusion layer constants.

All weights, thresholds, and risk boundaries that control how individual
model signals are combined live here.  Import from this module; do not
hardcode these values in fusion_aggregator.py.
"""

# ── Signal weights (must sum to ≤ 1.0) ───────────────────────────────────────
# Primary trend signal from XGBoost classifier.
WEIGHT_CLASSIFICATION: float = 0.50
# Amplifier / de-risker from FinBERT sentiment.
WEIGHT_SENTIMENT: float = 0.25
# Risk filter from TEN spoofing detection.
WEIGHT_SPOOFING: float = 0.20
# Secondary network-stress indicator from GNN spatial encoder.
WEIGHT_GNN: float = 0.05

# ── Fused-score action thresholds ────────────────────────────────────────────
# fused_score > BUY_THRESHOLD  → BUY
# fused_score < -SELL_THRESHOLD → SELL
# otherwise                    → HOLD
BUY_THRESHOLD: float = 0.30
SELL_THRESHOLD: float = 0.30

# ── Spoofing hard-block threshold ─────────────────────────────────────────────
# When spoofing confidence exceeds this value the action is forced to HOLD
# regardless of the fused score.
SPOOFING_HOLD_THRESHOLD: float = 0.70

# ── Risk level boundaries ─────────────────────────────────────────────────────
# risk_score = spoofing*0.4 + gnn*0.3 + (1-sentiment_rel)*0.3
RISK_HIGH_THRESHOLD: float = 0.65
RISK_MEDIUM_THRESHOLD: float = 0.35

# ── Alignment score constants ─────────────────────────────────────────────────
ALIGNMENT_STRONG_AGREE: float = 0.90
ALIGNMENT_STRONG_DISAGREE: float = 0.20
ALIGNMENT_NEUTRAL: float = 0.50
