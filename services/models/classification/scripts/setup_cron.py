"""
setup_cron.py — Install or remove the 15-minute pipeline cron job.

Usage:
    python scripts/setup_cron.py install   # add cron entry
    python scripts/setup_cron.py remove    # remove cron entry
    python scripts/setup_cron.py status    # show current crontab

The cron entry runs run_pipeline.sh at :00, :15, :30, :45 of every hour.
Logs per-run output to data/realtime/logs/cron_<timestamp>.log (written by
the shell script itself) and appends a summary line to cron_master.log.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR        = Path(__file__).resolve().parent
CLASSIFICATION_DIR = SCRIPT_DIR.parent
LOG_DIR           = CLASSIFICATION_DIR / "data" / "realtime" / "logs"
PIPELINE_SCRIPT   = SCRIPT_DIR / "run_pipeline.sh"
MASTER_LOG        = LOG_DIR / "cron_master.log"

# Marker so we can find and remove our entry without touching other cron jobs
CRON_MARKER = "# sol-trading-pipeline"

CRON_LINE = (
    f"*/15 * * * * {PIPELINE_SCRIPT} >> {MASTER_LOG} 2>&1  {CRON_MARKER}"
)


def _get_crontab() -> str:
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout


def _set_crontab(content: str) -> None:
    proc = subprocess.run(["crontab", "-"], input=content, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"ERROR: crontab -l failed: {proc.stderr}")
        sys.exit(1)


def install() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_SCRIPT.chmod(0o755)

    existing = _get_crontab()
    if CRON_MARKER in existing:
        print("Cron job already installed.")
        print(f"  Entry: {[l for l in existing.splitlines() if CRON_MARKER in l][0]}")
        return

    new_crontab = existing.rstrip("\n") + f"\n{CRON_LINE}\n"
    _set_crontab(new_crontab)
    print("Cron job installed.")
    print(f"  Schedule : every 15 minutes")
    print(f"  Script   : {PIPELINE_SCRIPT}")
    print(f"  Log      : {MASTER_LOG}")
    print()
    print("To start the orderbook daemon now:")
    print(f"  bash {SCRIPT_DIR}/start_orderbook.sh")


def remove() -> None:
    existing = _get_crontab()
    if CRON_MARKER not in existing:
        print("No cron job found to remove.")
        return
    lines = [l for l in existing.splitlines() if CRON_MARKER not in l]
    _set_crontab("\n".join(lines) + "\n")
    print("Cron job removed.")


def status() -> None:
    existing = _get_crontab()
    entries  = [l for l in existing.splitlines() if CRON_MARKER in l]
    if entries:
        print(f"Cron job ACTIVE:")
        for e in entries:
            print(f"  {e}")
    else:
        print("Cron job NOT installed.")
    print()
    print("Full crontab:")
    print(existing or "  (empty)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "install":
        install()
    elif cmd == "remove":
        remove()
    elif cmd == "status":
        status()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python setup_cron.py [install|remove|status]")
        sys.exit(1)
