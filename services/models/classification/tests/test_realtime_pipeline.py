import sys
from pathlib import Path
from types import SimpleNamespace


sys.path.append(str(Path(__file__).resolve().parents[1]))
from run_realtime_pipeline import build_steps, run_step  # noqa: E402


def test_build_steps_allows_skipping_expensive_fetches():
    args = SimpleNamespace(
        skip_market=True,
        skip_sentiment=True,
        skip_dashboard=False,
    )

    step_names = [step.name for step in build_steps(args)]

    assert step_names == [
        "feature_builder",
        "feature_merge",
        "predictor",
        "dashboard",
    ]


def test_run_step_dry_run_writes_log_without_executing(tmp_path):
    script = tmp_path / "would_fail.py"
    script.write_text("raise SystemExit(99)\n")
    log_path = tmp_path / "pipeline.log"

    step = SimpleNamespace(name="dry", script=script, critical=True)
    result = run_step(step, timeout_seconds=1, dry_run=True, log_path=log_path)

    assert result["status"] == "dry_run"
    assert result["returncode"] == 0
    assert log_path.exists()
    assert "dry-run" in log_path.read_text()
