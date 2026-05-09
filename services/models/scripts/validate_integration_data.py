"""
validate_integration_data.py

Pre-integration validation for the classification + NLP sentiment pipeline.
Checks both data files for: existence, timestamp integrity, duplicates,
and leakage columns BEFORE any training code is modified.

Usage:
    python validate_integration_data.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


# =========================================================
# PATHS - resolved relative to this script
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent          # services/models/

CLASSIFICATION_DATA_DIR = MODELS_DIR / "classification" / "data" / "processed"
NLP_DATA_DIR = MODELS_DIR / "nlp" / "data" / "processed"

CLASSIFICATION_CSV = CLASSIFICATION_DATA_DIR / "sol_usdt_15m_features_h4.csv"
SENTIMENT_CSV = NLP_DATA_DIR / "sentiment_features_15m.csv"

# Columns that indicate data leakage (target or future-derived)
LEAKAGE_COLUMNS = {
    "future_return",
    "future_return_h",
    "target",
    "target_up",
    "target_direction",
    "lead",
    "future",
    "t+1",
}


# =========================================================
# RESULT DATACLASS
# =========================================================
@dataclass
class ValidationResult:
    file_path: str
    exists: bool = False
    row_count: int = 0
    col_count: int = 0
    has_timestamp: bool = False
    timestamp_col: Optional[str] = None
    timestamp_parseable: bool = False
    duplicate_timestamps: int = 0
    missing_values: int = 0
    missing_value_cols: list[str] = field(default_factory=list)
    leakage_columns_found: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


# =========================================================
# VALIDATION LOGIC
# =========================================================
def find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Find the most likely timestamp column."""
    candidates = ["timestamp", "datetime", "date", "time", "ts"]
    for col in candidates:
        if col in df.columns:
            return col
    # Fallback: first column ending in common suffixes
    for col in df.columns:
        if col.lower().endswith(("_time", "_timestamp", "_date", "_ts")):
            return col
    return None


def validate_file(file_path: Path, label: str, mode: str = "pre_merge") -> ValidationResult:
    """Run all validation checks on a single CSV file."""
    result = ValidationResult(file_path=str(file_path))

    # --- CHECK 1: File exists ---
    if not file_path.exists():
        result.errors.append(f"FILE NOT FOUND: {file_path}")
        return result
    result.exists = True

    # --- CHECK 2: Load CSV ---
    try:
        df = pd.read_csv(file_path, nrows=0)  # header only first
        result.col_count = len(df.columns)
    except Exception as e:
        result.errors.append(f"CANNOT READ CSV: {e}")
        return result

    # Load full file (ok for these sizes)
    df = pd.read_csv(file_path)
    result.row_count = len(df)

    if result.row_count == 0:
        result.errors.append("FILE IS EMPTY (0 rows)")
        return result

    # --- CHECK 3: Timestamp column present ---
    ts_col = find_timestamp_column(df)
    if ts_col is None:
        result.errors.append(
            f"NO TIMESTAMP COLUMN FOUND. Columns: {list(df.columns)}"
        )
        return result
    result.has_timestamp = True
    result.timestamp_col = ts_col

    # --- CHECK 4: Timestamp parseable and UTC conversion ---
    try:
        parsed = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        null_count = parsed.isna().sum()
        if null_count == len(df):
            result.errors.append(
                f"ALL timestamps unparseable in column '{ts_col}'"
            )
        elif null_count > 0:
            result.warnings.append(
                f"{null_count}/{len(df)} timestamps could not be parsed"
            )
            result.timestamp_parseable = True
        else:
            result.timestamp_parseable = True
    except Exception as e:
        result.errors.append(f"TIMESTAMP PARSE FAILED: {e}")

    # --- CHECK 5: Duplicate timestamps ---
    if result.timestamp_parseable:
        dup_count = int(df[ts_col].duplicated().sum())
        result.duplicate_timestamps = dup_count
        if dup_count > 0:
            result.errors.append(
                f"DUPLICATE TIMESTAMPS: {dup_count} duplicates in '{ts_col}'"
            )

    # --- CHECK 6: Missing values ---
    missing_sum = df.isna().sum()
    if missing_sum.sum() > 0:
        result.missing_values = int(missing_sum.sum())
        result.missing_value_cols = missing_sum[missing_sum > 0].index.tolist()
        result.errors.append(
            f"MISSING VALUES: {result.missing_values} missing values found in columns: {result.missing_value_cols}"
        )

    # --- CHECK 7: Leakage columns ---
    cols_lower = {str(c).lower(): c for c in df.columns}
    for leak in LEAKAGE_COLUMNS:
        if leak.lower() in cols_lower:
            actual_name = cols_lower[leak.lower()]
            result.leakage_columns_found.append(actual_name)

    if result.leakage_columns_found:
        msg = f"LEAKAGE COLUMNS DETECTED: {result.leakage_columns_found}"
        if mode == "pre_merge":
            result.warnings.append(msg + " (Warning in pre_merge mode)")
        else:
            result.errors.append(msg)

    return result


def print_result(label: str, result: ValidationResult) -> None:
    """Pretty-print a single validation result."""
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"\n{'─' * 70}")
    print(f"  {label}: {status}")
    print(f"{'─' * 70}")
    print(f"  Path:            {result.file_path}")
    print(f"  Exists:          {result.exists}")

    if result.exists:
        print(f"  Shape:           {result.row_count} rows × {result.col_count} cols")
        print(f"  Timestamp col:   {result.timestamp_col}")
        print(f"  Parseable (UTC): {result.timestamp_parseable}")
        print(f"  Dup timestamps:  {result.duplicate_timestamps}")
        print(f"  Missing values:  {result.missing_values}")
        print(f"  Leakage cols:    {result.leakage_columns_found or 'None'}")

    if result.warnings:
        for w in result.warnings:
            print(f"  ⚠️  {w}")

    if result.errors:
        for e in result.errors:
            print(f"  ❌ {e}")


# =========================================================
# MAIN
# =========================================================
def main() -> int:
    print("=" * 70)
    print("  DATA INTEGRATION VALIDATION")
    print("  Validating files before merging NLP sentiment → Classification")
    print("=" * 70)

    # Pre-merge validation
    cls_result = validate_file(CLASSIFICATION_CSV, "Classification features", mode="pre_merge")
    sent_result = validate_file(SENTIMENT_CSV, "Sentiment features", mode="pre_merge")

    print_result("Classification features (pre-merge)", cls_result)
    print_result("Sentiment features (pre-merge)", sent_result)

    # Post-merge validation (if file exists)
    MERGED_CSV = CLASSIFICATION_DATA_DIR / "sol_usdt_15m_features_h4_with_sentiment.csv"
    if MERGED_CSV.exists():
        merged_result = validate_file(MERGED_CSV, "Merged features", mode="post_merge")
        print_result("Merged features (post-merge)", merged_result)
        all_passed = cls_result.passed and sent_result.passed and merged_result.passed
    else:
        merged_result = None
        all_passed = cls_result.passed and sent_result.passed

    # --- SUMMARY ---
    print(f"\n{'=' * 70}")
    if all_passed:
        print("  ✅ ALL CHECKS PASSED — safe to proceed with integration")
    else:
        print("  ❌ VALIDATION FAILED — fix issues before modifying training code")
        if not cls_result.passed:
            print(f"\n  Classification file issues:")
            for e in cls_result.errors:
                print(f"    • {e}")
        if not sent_result.passed:
            print(f"\n  Sentiment file issues:")
            for e in sent_result.errors:
                print(f"    • {e}")
        if merged_result is not None and not merged_result.passed:
            print(f"\n  Merged file issues:")
            for e in merged_result.errors:
                print(f"    • {e}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
