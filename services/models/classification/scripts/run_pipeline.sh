#!/usr/bin/env bash
# run_pipeline.sh — Run one cycle of the 15-minute realtime pipeline.
#
# Designed to be invoked by cron every 15 minutes:
#   */15 * * * * /path/to/scripts/run_pipeline.sh >> /path/to/logs/cron.log 2>&1
#
# Flags forwarded to run_realtime_pipeline.py:
#   --fusion           include DNN + TEN fusion signal
#   --fusion-aggregate run multi-signal aggregator → last_fused_decision.json
#   --paper-trade      simulate trades from the fused decision
#
# The orderbook collector (realtime_orderbook_collector.py) must be running
# as a separate persistent process — start it with start_orderbook.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFICATION_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$CLASSIFICATION_DIR")")")"

PYTHON="${REPO_ROOT}/.venv/bin/python"
PIPELINE="${CLASSIFICATION_DIR}/run_realtime_pipeline.py"
LOG_DIR="${CLASSIFICATION_DIR}/data/realtime/logs"

mkdir -p "$LOG_DIR"

# Timestamp for this run
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOGFILE="${LOG_DIR}/cron_${RUN_TS}.log"

echo "======================================" | tee -a "$LOGFILE"
echo "Pipeline run: ${RUN_TS}" | tee -a "$LOGFILE"
echo "======================================" | tee -a "$LOGFILE"

# Guard: don't run two instances at once
LOCKFILE="/tmp/sol_pipeline.lock"
if [ -e "$LOCKFILE" ]; then
    LOCK_AGE=$(( $(date +%s) - $(stat -c %Y "$LOCKFILE" 2>/dev/null || echo 0) ))
    if [ "$LOCK_AGE" -lt 900 ]; then
        echo "SKIP: pipeline already running (lock age ${LOCK_AGE}s)" | tee -a "$LOGFILE"
        exit 0
    fi
    echo "WARNING: stale lock (${LOCK_AGE}s), removing" | tee -a "$LOGFILE"
    rm -f "$LOCKFILE"
fi

touch "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

cd "$CLASSIFICATION_DIR"

"$PYTHON" "$PIPELINE" \
    --fusion \
    --fusion-aggregate \
    --paper-trade \
    2>&1 | tee -a "$LOGFILE"

echo "Done: ${RUN_TS}" | tee -a "$LOGFILE"
