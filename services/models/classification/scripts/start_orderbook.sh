#!/usr/bin/env bash
# start_orderbook.sh — Start the orderbook WebSocket collector as a background daemon.
#
# The collector runs continuously and must be started once before the pipeline.
# It writes to data/realtime/orderbook_features.csv (rolling 3-hour window).
#
# Usage:
#   ./scripts/start_orderbook.sh          # start in background
#   ./scripts/start_orderbook.sh --fg     # run in foreground (debug)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFICATION_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$CLASSIFICATION_DIR")")")"

PYTHON="${REPO_ROOT}/.venv/bin/python"
COLLECTOR="${CLASSIFICATION_DIR}/realtime_orderbook_collector.py"
LOG_DIR="${CLASSIFICATION_DIR}/data/realtime/logs"
PIDFILE="/tmp/sol_orderbook.pid"

mkdir -p "$LOG_DIR"

# Check if already running
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Orderbook collector already running (PID $OLD_PID)."
        exit 0
    fi
    rm -f "$PIDFILE"
fi

if [ "${1:-}" = "--fg" ]; then
    echo "Starting orderbook collector in foreground..."
    cd "$CLASSIFICATION_DIR"
    exec "$PYTHON" "$COLLECTOR"
fi

LOGFILE="${LOG_DIR}/orderbook_collector.log"
echo "Starting orderbook collector in background..."
echo "  Log : $LOGFILE"
echo "  PID : $PIDFILE"

cd "$CLASSIFICATION_DIR"
nohup "$PYTHON" "$COLLECTOR" >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

echo "Started (PID $(cat "$PIDFILE")). Stop with: kill $(cat "$PIDFILE")"
