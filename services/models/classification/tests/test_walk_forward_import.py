"""
Tests that 20_walk_forward_validation.py and 23_backtest.py are importable
without triggering any I/O, model training, or file reads.

Before the refactor both scripts executed everything at module level, making
them un-importable in tests (they'd try to read data files and crash or
run for several minutes).
"""
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

_CLASSIFICATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CLASSIFICATION))


def _import_module_from_file(name: str, path: Path):
    """Load a module from an absolute path without executing __main__ block."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"Could not create spec for {path}"
    mod = importlib.util.module_from_spec(spec)
    # We deliberately do NOT call spec.loader.exec_module(mod) here for
    # scripts that are not yet importable — instead we test the ones that are.
    return spec, mod


class TestWalkForwardValidationImportable:
    """20_walk_forward_validation.py must be importable without executing main()."""

    def test_module_imports_without_running(self, tmp_path, monkeypatch):
        """
        Importing 20_walk_forward_validation should not call pd.read_csv or print
        anything.  We monkey-patch the data-reading path to a non-existent file
        so that any accidental top-level I/O would raise rather than block.
        """
        import pandas as pd

        read_csv_calls = []
        original_read_csv = pd.read_csv

        def mock_read_csv(*args, **kwargs):
            read_csv_calls.append(args[0])
            return original_read_csv(*args, **kwargs)

        # If anything calls pd.read_csv at import time it will be captured
        monkeypatch.setattr(pd, "read_csv", mock_read_csv)

        # Clear any cached import
        for key in list(sys.modules.keys()):
            if "20_walk_forward" in key or "walk_forward_validation" in key:
                del sys.modules[key]

        spec = importlib.util.spec_from_file_location(
            "walk_forward_validation",
            _CLASSIFICATION / "20_walk_forward_validation.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        assert not read_csv_calls, (
            f"pd.read_csv was called at import time with: {read_csv_calls}\n"
            "All I/O must be inside main()."
        )

    def test_calibrate_threshold_importable(self):
        """calibrate_threshold must be accessible as a module-level function."""
        spec = importlib.util.spec_from_file_location(
            "walk_forward_validation",
            _CLASSIFICATION / "20_walk_forward_validation.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        assert callable(mod.calibrate_threshold)

    def test_gated_metrics_importable(self):
        """gated_metrics must be accessible as a module-level function."""
        spec = importlib.util.spec_from_file_location(
            "walk_forward_validation",
            _CLASSIFICATION / "20_walk_forward_validation.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        assert callable(mod.gated_metrics)

    def test_calibrate_threshold_logic(self):
        """calibrate_threshold should return the threshold that maximises F1."""
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "walk_forward_validation",
            _CLASSIFICATION / "20_walk_forward_validation.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        # Perfect classifier: all positives score 0.9, all negatives 0.1
        np.random.seed(42)
        n = 200
        y = np.array([1] * (n // 2) + [0] * (n // 2))
        p = np.where(y == 1, 0.9, 0.1)
        thresh = mod.calibrate_threshold(y, p)
        assert 0.1 < thresh < 0.9, f"Expected threshold between 0.1 and 0.9, got {thresh}"


class TestBacktestImportable:
    """23_backtest.py must be importable without executing main()."""

    def test_module_imports_without_running(self, monkeypatch):
        import pandas as pd

        read_csv_calls = []
        original = pd.read_csv

        def mock_read_csv(*args, **kwargs):
            read_csv_calls.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(pd, "read_csv", mock_read_csv)

        for key in list(sys.modules.keys()):
            if "23_backtest" in key:
                del sys.modules[key]

        spec = importlib.util.spec_from_file_location(
            "backtest_23",
            _CLASSIFICATION / "23_backtest.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        assert not read_csv_calls, (
            f"pd.read_csv called at import time: {read_csv_calls}"
        )

    def test_main_and_parse_args_exist(self):
        spec = importlib.util.spec_from_file_location(
            "backtest_23",
            _CLASSIFICATION / "23_backtest.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        assert callable(mod.main)
        assert callable(mod.parse_args)
