"""
test_validate_integration_data.py

Unit tests for the integration data validation script.
Uses temporary CSV fixtures to test each validation check in isolation.

Usage:
    pytest test_validate_integration_data.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

# We import directly from the validation module
# Adjust sys.path so the import works from any CWD
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from validate_integration_data import (  # noqa: E402
    LEAKAGE_COLUMNS,
    ValidationResult,
    find_timestamp_column,
    validate_file,
)


# =========================================================
# FIXTURES
# =========================================================
@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False)
    return path


# =========================================================
# TEST: find_timestamp_column
# =========================================================
class TestFindTimestampColumn:
    def test_finds_timestamp(self):
        df = pd.DataFrame({"timestamp": [1], "close": [100]})
        assert find_timestamp_column(df) == "timestamp"

    def test_finds_datetime(self):
        df = pd.DataFrame({"datetime": [1], "close": [100]})
        assert find_timestamp_column(df) == "datetime"

    def test_finds_suffix(self):
        df = pd.DataFrame({"event_time": [1], "close": [100]})
        assert find_timestamp_column(df) == "event_time"

    def test_returns_none_for_no_match(self):
        df = pd.DataFrame({"close": [100], "volume": [200]})
        assert find_timestamp_column(df) is None


# =========================================================
# TEST: File existence
# =========================================================
class TestFileExistence:
    def test_missing_file_fails(self, tmp_dir):
        result = validate_file(tmp_dir / "nonexistent.csv", "test")
        assert not result.passed
        assert not result.exists
        assert any("NOT FOUND" in e for e in result.errors)

    def test_existing_file_passes_existence(self, tmp_dir):
        csv = _write_csv(
            tmp_dir / "ok.csv",
            pd.DataFrame({"timestamp": ["2026-01-01 00:00:00"], "value": [1]}),
        )
        result = validate_file(csv, "test")
        assert result.exists


# =========================================================
# TEST: Timestamp validation
# =========================================================
class TestTimestampValidation:
    def test_no_timestamp_column_fails(self, tmp_dir):
        csv = _write_csv(
            tmp_dir / "no_ts.csv",
            pd.DataFrame({"close": [1, 2], "volume": [100, 200]}),
        )
        result = validate_file(csv, "test")
        assert not result.passed
        assert not result.has_timestamp

    def test_valid_timestamps_pass(self, tmp_dir):
        csv = _write_csv(
            tmp_dir / "good_ts.csv",
            pd.DataFrame({
                "timestamp": [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:15:00",
                    "2026-01-01 00:30:00",
                ],
                "value": [1, 2, 3],
            }),
        )
        result = validate_file(csv, "test")
        assert result.has_timestamp
        assert result.timestamp_parseable
        assert result.duplicate_timestamps == 0

    def test_unparseable_timestamps_fail(self, tmp_dir):
        csv = _write_csv(
            tmp_dir / "bad_ts.csv",
            pd.DataFrame({
                "timestamp": ["not-a-date", "also-bad", "nope"],
                "value": [1, 2, 3],
            }),
        )
        result = validate_file(csv, "test")
        assert not result.passed

    def test_duplicate_timestamps_fail(self, tmp_dir):
        csv = _write_csv(
            tmp_dir / "dup_ts.csv",
            pd.DataFrame({
                "timestamp": [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",  # duplicate
                    "2026-01-01 00:15:00",
                ],
                "value": [1, 2, 3],
            }),
        )
        result = validate_file(csv, "test")
        assert not result.passed
        assert result.duplicate_timestamps == 1
        assert any("DUPLICATE" in e for e in result.errors)


# =========================================================
# TEST: Leakage detection
# =========================================================
class TestLeakageDetection:
    @pytest.mark.parametrize(
        "leak_col",
        ["future_return", "target", "target_up", "future_return_h", "target_direction", "lead", "future", "t+1"],
    )
    def test_leakage_warning_pre_merge(self, tmp_dir, leak_col):
        csv = _write_csv(
            tmp_dir / "leak_pre.csv",
            pd.DataFrame({
                "timestamp": ["2026-01-01 00:00:00"],
                "close": [100],
                leak_col: [1],
            }),
        )
        result = validate_file(csv, "test", mode="pre_merge")
        assert result.passed  # It is a warning, so it should pass
        assert leak_col in result.leakage_columns_found
        assert any("LEAKAGE" in w for w in result.warnings)
        assert not any("LEAKAGE" in e for e in result.errors)

    @pytest.mark.parametrize(
        "leak_col",
        ["future_return", "target", "target_up", "future_return_h", "target_direction", "lead", "future", "t+1"],
    )
    def test_leakage_error_post_merge(self, tmp_dir, leak_col):
        csv = _write_csv(
            tmp_dir / "leak_post.csv",
            pd.DataFrame({
                "timestamp": ["2026-01-01 00:00:00"],
                "close": [100],
                leak_col: [1],
            }),
        )
        result = validate_file(csv, "test", mode="post_merge")
        assert not result.passed  # It is an error in post-merge
        assert leak_col in result.leakage_columns_found
        assert any("LEAKAGE" in e for e in result.errors)
        assert not any("LEAKAGE" in w for w in result.warnings)

    def test_no_leakage_passes(self, tmp_dir):
        csv = _write_csv(
            tmp_dir / "clean.csv",
            pd.DataFrame({
                "timestamp": ["2026-01-01 00:00:00"],
                "close": [100],
                "rsi_14": [55.0],
            }),
        )
        result = validate_file(csv, "test")
        assert result.leakage_columns_found == []
        assert result.passed

    def test_case_insensitive_leakage(self, tmp_dir):
        csv = _write_csv(
            tmp_dir / "case_leak.csv",
            pd.DataFrame({
                "timestamp": ["2026-01-01 00:00:00"],
                "close": [100],
                "Future_Return": [0.01],  # mixed case
            }),
        )
        # Testing in post_merge to ensure it fails
        result = validate_file(csv, "test", mode="post_merge")
        assert not result.passed
        assert len(result.leakage_columns_found) > 0


# =========================================================
# TEST: Empty file
# =========================================================
class TestEmptyFile:
    def test_empty_csv_fails(self, tmp_dir):
        csv = _write_csv(
            tmp_dir / "empty.csv",
            pd.DataFrame({"timestamp": pd.Series(dtype="str"), "v": pd.Series(dtype="float")}),
        )
        result = validate_file(csv, "test")
        assert not result.passed
        assert any("EMPTY" in e for e in result.errors)


# =========================================================
# TEST: Missing Values
# =========================================================
class TestMissingValues:
    def test_missing_values_fails(self, tmp_dir):
        csv = _write_csv(
            tmp_dir / "nan.csv",
            pd.DataFrame({
                "timestamp": ["2026-01-01 00:00:00"],
                "close": [100.0],
                "volume": [None]
            })
        )
        result = validate_file(csv, "test")
        assert not result.passed
        assert result.missing_values == 1
        assert "volume" in result.missing_value_cols
        assert any("MISSING VALUES" in e for e in result.errors)



# =========================================================
# TEST: Full passing file (integration sanity)
# =========================================================
class TestFullPass:
    def test_clean_file_passes_all_checks(self, tmp_dir):
        ts = pd.date_range("2026-01-01", periods=10, freq="15min")
        csv = _write_csv(
            tmp_dir / "perfect.csv",
            pd.DataFrame({
                "timestamp": ts.astype(str),
                "close": range(10),
                "rsi_14": [50.0] * 10,
                "volume": [1000] * 10,
            }),
        )
        result = validate_file(csv, "test")
        assert result.passed
        assert result.exists
        assert result.has_timestamp
        assert result.timestamp_parseable
        assert result.duplicate_timestamps == 0
        assert result.missing_values == 0
        assert result.leakage_columns_found == []
        assert result.row_count == 10
        assert result.col_count == 4


# =========================================================
# TEST: ValidationResult.passed property
# =========================================================
class TestValidationResult:
    def test_no_errors_is_pass(self):
        r = ValidationResult(file_path="x.csv", exists=True)
        assert r.passed

    def test_with_errors_is_fail(self):
        r = ValidationResult(file_path="x.csv", exists=True, errors=["boom"])
        assert not r.passed

    def test_warnings_dont_fail(self):
        r = ValidationResult(file_path="x.csv", exists=True, warnings=["note"])
        assert r.passed
