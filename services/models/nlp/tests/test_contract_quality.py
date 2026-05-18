"""
Contract quality tests for the NLP pipeline.

These guard against silent regressions when the feature contract, relevance
scorer, or aggregation logic changes.  Run with:
    cd services/models/nlp && python -m pytest tests/test_contract_quality.py -v
"""
import pandas as pd
import pytest

from src.contracts.layer2_contract import BUCKET_DEFAULTS, normalize_layer2_bucket_frame
from src.aggregation.feature_aggregator import aggregate_sentiment_15m
from src.processing.text_cleaner import calculate_relevance_score


# ── Relevance scorer ──────────────────────────────────────────────────────────

class TestRelevanceScorerContract:
    """Ecosystem keywords must score 1.5 (same weight as 'solana'/'sol')."""

    @pytest.mark.parametrize("keyword", ["jupiter", "raydium", "jito", "marinade", "phantom"])
    def test_ecosystem_keyword_scores_1_5(self, keyword):
        score = calculate_relevance_score(f"{keyword} protocol announcement")
        assert score == 1.5, f"'{keyword}' should score 1.5, got {score}"

    @pytest.mark.parametrize("keyword", ["solana", "sol"])
    def test_core_keyword_scores_1_5(self, keyword):
        score = calculate_relevance_score(f"the {keyword} blockchain")
        assert score == 1.5, f"'{keyword}' should score 1.5, got {score}"

    @pytest.mark.parametrize("keyword", ["etf", "hack", "exploit", "whale", "defi", "dex"])
    def test_tier2_keyword_scores_1_0(self, keyword):
        score = calculate_relevance_score(f"crypto {keyword} news today")
        assert score == 1.0, f"'{keyword}' should score 1.0, got {score}"

    def test_off_topic_scores_zero(self):
        assert calculate_relevance_score("stock market rally fed rate cut") == 0.0
        assert calculate_relevance_score("nba finals game seven overtime") == 0.0

    def test_combined_ecosystem_and_event_scores_additive(self):
        # solana(1.5) + hack(1.0) + exploit(1.0) = 3.5
        score = calculate_relevance_score("solana hack exploit reported")
        assert score == 3.5

    def test_threshold_1_5_boundary(self):
        """Any single core/ecosystem keyword meets the production threshold of 1.5."""
        assert calculate_relevance_score("solana staking update") >= 1.5
        assert calculate_relevance_score("raydium pool launched") >= 1.5
        assert calculate_relevance_score("bitcoin nft art drop") < 1.5   # no Solana keyword


# ── BUCKET_DEFAULTS contract ──────────────────────────────────────────────────

class TestBucketDefaultsContract:
    """BUCKET_DEFAULTS must always include gdelt_tone_mean."""

    REQUIRED_BUCKET_FIELDS = [
        "timestamp", "sentiment_mean", "sentiment_std", "news_volume",
        "positive_ratio", "negative_ratio", "neutral_ratio",
        "strong_positive_ratio", "strong_negative_ratio",
        "max_confidence", "event_count", "negative_event_count",
        "positive_event_count", "high_impact_event_count", "high_impact_event",
        "relevance_weighted_sentiment", "event_signal_mean", "event_signal_sum",
        "gdelt_volume", "rss_volume", "gdelt_sentiment", "rss_sentiment",
        "sentiment_momentum", "news_volume_change", "sentiment_spike",
        "gdelt_tone_mean",
    ]

    def test_all_required_fields_in_bucket_defaults(self):
        missing = [f for f in self.REQUIRED_BUCKET_FIELDS if f not in BUCKET_DEFAULTS]
        assert not missing, f"Missing from BUCKET_DEFAULTS: {missing}"

    def test_normalize_bucket_frame_adds_gdelt_tone_mean(self):
        df = pd.DataFrame([{"timestamp": "2026-05-15T10:00:00Z", "sentiment_mean": 0.3}])
        out = normalize_layer2_bucket_frame(df)
        assert "gdelt_tone_mean" in out.columns
        assert out.iloc[0]["gdelt_tone_mean"] == 0.0

    def test_normalize_bucket_frame_output_has_all_required_fields(self):
        df = pd.DataFrame([{"timestamp": "2026-05-15T10:00:00Z"}])
        out = normalize_layer2_bucket_frame(df)
        missing = [f for f in self.REQUIRED_BUCKET_FIELDS if f not in out.columns]
        assert not missing, f"normalize_layer2_bucket_frame missing columns: {missing}"


# ── gdelt_tone_mean aggregation ───────────────────────────────────────────────

def _make_article(ts: str, source: str, score: float, tone: float | None = None) -> dict:
    return {
        "timestamp": ts,
        "source": source,
        "sentiment_score": score,
        "confidence": 0.8,
        "sentiment_label": "positive" if score > 0 else "negative",
        "relevance_score": 2.0,
        "event_type": "general",
        "event_weight": 0.5,
        "event_signal": score * 0.5,
        "high_impact_event": 0,
        "gdelt_tone_norm": tone,
    }


class TestGdeltToneMeanAggregation:

    def test_gdelt_tone_mean_computed_from_gdelt_articles(self):
        articles = [
            _make_article("2026-05-15T10:00:00Z", "gdelt", 0.5, tone=0.12),
            _make_article("2026-05-15T10:05:00Z", "gdelt", 0.3, tone=0.08),
            _make_article("2026-05-15T10:08:00Z", "rss",   0.4, tone=None),
        ]
        df = pd.DataFrame(articles)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        out = aggregate_sentiment_15m(df)
        assert "gdelt_tone_mean" in out.columns
        # RSS row has no tone; only two GDELT rows → mean = (0.12 + 0.08) / 2 = 0.10
        assert abs(out.iloc[0]["gdelt_tone_mean"] - 0.10) < 1e-6

    def test_gdelt_tone_mean_zero_when_no_gdelt_articles(self):
        articles = [
            _make_article("2026-05-15T10:00:00Z", "rss", 0.4, tone=None),
            _make_article("2026-05-15T10:05:00Z", "rss", -0.2, tone=None),
        ]
        df = pd.DataFrame(articles)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        out = aggregate_sentiment_15m(df)
        assert "gdelt_tone_mean" in out.columns
        assert out.iloc[0]["gdelt_tone_mean"] == 0.0

    def test_gdelt_tone_mean_present_even_without_gdelt_tone_norm_column(self):
        """Aggregator must add gdelt_tone_mean=0.0 when column is absent."""
        articles = [_make_article("2026-05-15T10:00:00Z", "rss", 0.4)]
        df = pd.DataFrame(articles)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # Drop the column entirely to simulate a pipeline without gdelt_tone_norm
        df = df.drop(columns=["gdelt_tone_norm"])

        out = aggregate_sentiment_15m(df)
        assert "gdelt_tone_mean" in out.columns
        assert out.iloc[0]["gdelt_tone_mean"] == 0.0
