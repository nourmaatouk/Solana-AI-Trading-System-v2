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
# PPO reinforcement learning agent (parallel signal).
# When this signal is available, the base fused score is blended:
#   final = (1 - WEIGHT_RL) * base_score + WEIGHT_RL * rl_score
# When the RL model has not been trained yet, the base score is used unchanged.
WEIGHT_RL: float = 0.15  # enabled: val Sharpe=1.926, return=+10%, max_dd=3.39%

# ── Fused-score action thresholds ────────────────────────────────────────────
# fused_score > BUY_THRESHOLD  → BUY
# fused_score < -SELL_THRESHOLD → SELL
# otherwise                    → HOLD
#
# Calibration: when sentiment and spoofing are absent (the typical live case),
# only XGBoost contributes (weight=0.50).  The intended gate is prob ≥ 0.69, so
# the effective fused-score threshold must be 0.50 × 0.69 = 0.345.  Using 0.30
# admitted signals at prob ≥ 0.60, 9 pp below the optimised gate.
BUY_THRESHOLD: float = 0.345
SELL_THRESHOLD: float = 0.345

# ── Spoofing hard-block threshold ─────────────────────────────────────────────
# When spoofing confidence exceeds this value the action is forced to HOLD
# regardless of the fused score.
SPOOFING_HOLD_THRESHOLD: float = 0.70

# ── Risk level boundaries ─────────────────────────────────────────────────────
# risk_score = spoofing*RISK_WEIGHT_SPOOFING + gnn*RISK_WEIGHT_GNN + (1-sentiment_rel)*RISK_WEIGHT_SENTIMENT_PENALTY
RISK_HIGH_THRESHOLD: float = 0.65
RISK_MEDIUM_THRESHOLD: float = 0.35

# ── Alignment score constants ─────────────────────────────────────────────────
ALIGNMENT_STRONG_AGREE: float = 0.90
ALIGNMENT_STRONG_DISAGREE: float = 0.20
ALIGNMENT_NEUTRAL: float = 0.50
# Score magnitude below this is treated as "flat" (neither bullish nor bearish).
ALIGNMENT_SIGNAL_THRESHOLD: float = 0.1

# ── Spoofing severity thresholds ──────────────────────────────────────────────
# confidence > HIGH  → severity="high"
# confidence > MEDIUM → severity="medium"
# otherwise          → severity="low"
SPOOFING_SEVERITY_HIGH: float = 0.6
SPOOFING_SEVERITY_MEDIUM: float = 0.3

# ── Sentiment label thresholds ────────────────────────────────────────────────
# relevance_weighted_sentiment > POSITIVE_THRESHOLD → label="POSITIVE"
# relevance_weighted_sentiment < NEGATIVE_THRESHOLD → label="NEGATIVE"
# otherwise                                          → label="NEUTRAL"
SENTIMENT_POSITIVE_THRESHOLD: float = 0.1
SENTIMENT_NEGATIVE_THRESHOLD: float = -0.1

# ── GNN stress penalty ────────────────────────────────────────────────────────
# Mild multiplier applied to GNN anomaly score in the fused-score formula.
GNN_STRESS_PENALTY: float = 0.5

# ── High-impact event relevance boost ────────────────────────────────────────
# Amount added to sentiment_rel when a high-impact news event is flagged.
HIGH_IMPACT_RELEVANCE_BOOST: float = 0.2

# ── Risk score component weights (must sum to 1.0) ────────────────────────────
# risk_score = spoofing * RISK_WEIGHT_SPOOFING
#            + gnn      * RISK_WEIGHT_GNN
#            + (1 - sentiment_rel) * RISK_WEIGHT_SENTIMENT_PENALTY
RISK_WEIGHT_SPOOFING: float = 0.4
RISK_WEIGHT_GNN: float = 0.3
RISK_WEIGHT_SENTIMENT_PENALTY: float = 0.3
