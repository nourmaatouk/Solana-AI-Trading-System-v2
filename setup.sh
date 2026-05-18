#!/usr/bin/env bash
# First-time setup: copy env, build backend + frontend
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"

echo "[setup] Copying NLP .env..."
if [ ! -f "$REPO/services/models/nlp/.env" ]; then
  cp "$REPO/services/models/nlp/.env.example" "$REPO/services/models/nlp/.env"
  echo "[setup] Created services/models/nlp/.env — edit it to add API keys if needed."
else
  echo "[setup] services/models/nlp/.env already exists, skipping."
fi

echo "[setup] Building backend..."
cd "$REPO/apps/trading-dashboard/backend"
npm install --silent
npm run build

echo "[setup] Building frontend..."
cd "$REPO/apps/trading-dashboard/frontend"
npm install --silent
npm run build

echo ""
echo "[setup] Done. Run './start.sh' to launch the system."
