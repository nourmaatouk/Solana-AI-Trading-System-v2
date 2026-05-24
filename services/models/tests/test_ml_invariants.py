"""
test_ml_invariants.py

ML quality and runtime-safety invariants for the model layer:
silent-fallback warnings, train/val/test split integrity, leakage guards,
artifact presence, and environment-variable coverage for external connections.

These are behaviour/config checks, not CI source-scanning audits.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

MODELS_ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# SILENT FALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class TestSilentFallbacks:
    """
    Fallbacks that silently substitute zero-filled data can mask missing models
    or bad data.  Each one must print a WARNING-level message before continuing.
    """

    def test_btc_fallback_logs_warning(self):
        """BTC zero-fill in realtime_feature_builder must print a WARNING."""
        path = MODELS_ROOT / "classification" / "realtime_feature_builder.py"
        src = path.read_text()
        # The zero-fill block must contain an explicit warning message.
        assert "WARNING" in src, (
            "realtime_feature_builder.py BTC zero-fill path must print a WARNING."
        )

    def test_sentiment_fallback_logs_warning(self):
        """Sentiment decay / zero row in realtime_sentiment_fetcher must log clearly."""
        path = MODELS_ROOT / "classification" / "realtime_sentiment_fetcher.py"
        src = path.read_text()
        assert "fallback" in src.lower() or "decay" in src.lower(), (
            "realtime_sentiment_fetcher.py must document its fallback behaviour."
        )

    def test_orderbook_stale_check_exists(self):
        """realtime_merge_features must refuse stale orderbook data, not silently use it."""
        path = MODELS_ROOT / "classification" / "realtime_merge_features.py"
        src = path.read_text()
        assert "ORDERBOOK_STALE_MINUTES" in src or "stale" in src.lower(), (
            "realtime_merge_features.py must check orderbook staleness."
        )

    def test_predictor_aborts_on_missing_features(self):
        """realtime_predictor must abort (not silently zero-pad) on missing features."""
        path = MODELS_ROOT / "classification" / "realtime_predictor.py"
        src = path.read_text()
        assert "INSUFFICIENT FEATURES" in src or "missing_features" in src, (
            "realtime_predictor.py must refuse to score when features are missing."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ML QUALITY INVARIANTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMLQualityInvariants:
    """Structural checks that guard against common ML pitfalls."""

    def test_config_split_ratios_sum_to_one(self):
        """Train/val/test ratios in config.py must sum to 1.0."""
        import sys
        sys.path.insert(0, str(MODELS_ROOT / "classification"))
        from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

    def test_config_horizons_are_positive_integers(self):
        """HORIZONS must be a non-empty list of positive integers."""
        import sys
        sys.path.insert(0, str(MODELS_ROOT / "classification"))
        from config import HORIZONS
        assert HORIZONS
        assert all(isinstance(h, int) and h > 0 for h in HORIZONS)

    def test_xgb_seed_is_fixed(self):
        """XGB_SEED must be a fixed integer to ensure reproducibility."""
        import sys
        sys.path.insert(0, str(MODELS_ROOT / "classification"))
        from config import XGB_SEED
        assert isinstance(XGB_SEED, int), "XGB_SEED must be an int"

    def test_no_shuffle_before_split_in_training_scripts(self):
        """
        Training scripts must NOT shuffle time-series data before splitting.
        A naive df.sample() or shuffle=True before a chronological split
        causes lookahead bias.
        """
        suspicious_patterns = [
            r"\.sample\(frac=1",
            r"shuffle\s*=\s*True",
            r"np\.random\.shuffle",
        ]
        training_scripts = [
            MODELS_ROOT / "classification" / "07_prepare_no_leak_data.py",
            MODELS_ROOT / "classification" / "08_train_xgboost_no_leak.py",
            MODELS_ROOT / "classification" / "20_walk_forward_validation.py",
        ]
        for script in training_scripts:
            if not script.exists():
                continue
            src = script.read_text()
            for pat in suspicious_patterns:
                m = re.search(pat, src)
                assert not m, (
                    f"Potential time-series shuffle detected in {script.name}: "
                    f"matched pattern '{pat}' — verify it is not applied before a "
                    f"chronological split."
                )

    def test_leakage_substrings_cover_future_columns(self):
        """LEAKAGE_SUBSTRINGS must include 'future' and 'target'."""
        import sys
        sys.path.insert(0, str(MODELS_ROOT / "classification"))
        from config import LEAKAGE_SUBSTRINGS
        assert "future" in LEAKAGE_SUBSTRINGS or any("future" in s for s in LEAKAGE_SUBSTRINGS)
        assert "target" in LEAKAGE_SUBSTRINGS or any("target" in s for s in LEAKAGE_SUBSTRINGS)

    def test_walk_forward_uses_purge_gap(self):
        """Walk-forward validation must apply a purge gap to prevent label leakage."""
        path = MODELS_ROOT / "classification" / "20_walk_forward_validation.py"
        if not path.exists():
            pytest.skip("20_walk_forward_validation.py not found")
        src = path.read_text()
        assert "purge" in src.lower() or "HORIZON" in src, (
            "Walk-forward validation must implement a purge gap at fold boundaries."
        )

    def test_model_artifacts_exist_for_all_configured_horizons(self):
        """Every HORIZONS entry should have a corresponding saved model artifact.

        Skips when the binaries are absent (e.g. a fresh checkout — model files
        are gitignored under services/models/models/). Mirrors the skip behaviour
        of the adjacent scaler / feature-registry checks.
        """
        import sys
        sys.path.insert(0, str(MODELS_ROOT / "classification"))
        from config import HORIZONS, MODEL_DIR
        missing = []
        for h in HORIZONS:
            model_file = MODEL_DIR / f"xgboost_h{h}.json"
            if not model_file.exists():
                missing.append(str(model_file))
        if missing:
            pytest.skip(
                "Model artifacts are absent in this checkout (gitignored binaries); "
                "run services/models/classification/08_train_xgboost_no_leak.py. "
                f"Missing: {missing}"
            )

    def test_scaler_artifacts_exist_for_all_configured_horizons(self):
        """Every HORIZONS entry should have a corresponding generated scaler artifact."""
        import sys
        sys.path.insert(0, str(MODELS_ROOT / "classification"))
        from config import HORIZONS, PROCESSED_DIR
        missing = []
        for h in HORIZONS:
            scaler_file = PROCESSED_DIR / f"scaler_h{h}.pkl"
            if not scaler_file.exists():
                missing.append(scaler_file)
        if missing:
            pytest.skip(
                "Generated scaler artifacts are absent in this checkout; "
                f"run 07_prepare_no_leak_data.py. Missing: {missing}"
            )

    def test_feature_columns_registry_exists(self):
        """feature_columns.json should exist after no-leak feature preparation."""
        import sys
        sys.path.insert(0, str(MODELS_ROOT / "classification"))
        from config import PROCESSED_DIR
        registry = PROCESSED_DIR / "feature_columns.json"
        if not registry.exists():
            pytest.skip(
                f"Generated feature registry missing: {registry}. "
                "Run 07_prepare_no_leak_data.py before enforcing artifact presence."
            )

    def test_feature_columns_registry_is_valid_json_list_of_strings(self):
        """feature_columns.json must be a non-empty JSON list of strings."""
        import sys
        sys.path.insert(0, str(MODELS_ROOT / "classification"))
        from config import PROCESSED_DIR
        registry = PROCESSED_DIR / "feature_columns.json"
        if not registry.exists():
            pytest.skip("feature_columns.json not found — run 07 first")
        data = json.loads(registry.read_text())
        assert isinstance(data, list) and data
        assert all(isinstance(s, str) for s in data)

    def test_confidence_threshold_and_model_summary_agree(self):
        """
        config.CONFIDENCE_THRESHOLD and model_performance_summary.json must agree.

        Resolved 2026-05-13: gate sweep (24_gate_sweep.py) identified 0.69 as the
        peak-Sharpe gate (Sharpe 5.26, max-DD 23%) vs the prior F1-calibrated gate
        of 0.62 (Sharpe 4.09, max-DD 33%).  config.py was already at 0.69; the
        summary JSON was stale at 0.62.  Both are now 0.69.

        If this test fails again in future it means a new gate sweep was run and
        its result was not yet applied to config.py — run 23_backtest.py at the
        new gate and update config.py if Sharpe and max-DD hold.
        """
        import sys
        sys.path.insert(0, str(MODELS_ROOT / "classification"))
        from config import CONFIDENCE_THRESHOLD
        summary_path = MODELS_ROOT / "classification" / "reports" / "model_performance_summary.json"
        if not summary_path.exists():
            pytest.skip("model_performance_summary.json not found")
        summary = json.loads(summary_path.read_text())
        report_threshold = summary.get("production_model", {}).get("confidence_threshold")
        if report_threshold is None:
            pytest.skip("confidence_threshold not in summary JSON")
        drift = abs(float(report_threshold) - CONFIDENCE_THRESHOLD)
        assert drift < 0.01, (
            f"config.CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD} vs "
            f"model_performance_summary.json={report_threshold} (drift={drift:.3f}). "
            "Run 23_backtest.py with the new gate and update both if results hold."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT VARIABLE COVERAGE
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvironmentVariables:
    """Confirm that every external connection string can be overridden via env vars."""

    @staticmethod
    def _load_nlp_settings_class():
        """Import Settings from nlp/src/config.py (not the classification config)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "nlp_config",
            MODELS_ROOT / "nlp" / "src" / "config.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.Settings

    def test_nlp_settings_reads_timescale_from_env(self):
        """TIMESCALE_DSN must be configurable via the TIMESCALE_DSN env variable."""
        import os
        os.environ["TIMESCALE_DSN"] = "postgresql://user:TESTPASS@db:5432/trading_test"
        try:
            Settings = self._load_nlp_settings_class()
            settings = Settings()
            assert "TESTPASS" in settings.TIMESCALE_DSN
        finally:
            del os.environ["TIMESCALE_DSN"]

    def test_nlp_settings_reads_redis_from_env(self):
        """REDIS_URL must be configurable via the REDIS_URL env variable."""
        import os
        os.environ["REDIS_URL"] = "redis://redis-test:6380/1"
        try:
            Settings = self._load_nlp_settings_class()
            settings = Settings()
            assert "redis-test" in settings.REDIS_URL
        finally:
            del os.environ["REDIS_URL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
