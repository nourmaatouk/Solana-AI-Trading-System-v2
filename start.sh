#!/usr/bin/env bash
# Start all services. Restarts cleanly if already running.
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

if pm2 list 2>/dev/null | grep -q "online"; then
  echo "[start] Restarting existing PM2 services..."
  pm2 restart ecosystem.config.cjs
else
  echo "[start] Starting all services..."
  pm2 start ecosystem.config.cjs
fi

echo ""
pm2 status
echo ""
echo "[start] Dashboard → http://localhost:3001"
echo "[start] Python API → http://localhost:8000"
echo ""
echo "Useful commands:"
echo "  pm2 logs                   — live logs from all services"
echo "  pm2 logs pipeline-scheduler — pipeline run logs"
echo "  pm2 status                 — service health"
echo "  pm2 stop all               — stop everything"
