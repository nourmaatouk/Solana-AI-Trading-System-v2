const fs     = require('fs');
const path   = require('path');

const REPO   = __dirname;  // resolves to the repo root on any machine
const PYTHON  = `${REPO}/.venv/bin/python`;
const UVICORN = `${REPO}/.venv/bin/uvicorn`;

// Load .env file into an object (ignored if file doesn't exist)
function loadDotenv(envPath) {
  try {
    return Object.fromEntries(
      fs.readFileSync(envPath, 'utf8')
        .split('\n')
        .filter(l => l.trim() && !l.trim().startsWith('#') && l.includes('='))
        .map(l => { const i = l.indexOf('='); return [l.slice(0, i).trim(), l.slice(i + 1).trim()]; })
    );
  } catch { return {}; }
}
const DOT_ENV = loadDotenv(path.join(REPO, '.env'));

// ── Shared constants (single place to change) ────────────────────────────────
const SYMBOL            = (DOT_ENV.ACTIVE_SYMBOL ?? 'SOLUSDT').replace('/', '');
const SCHEDULER_INTERVAL = 900;   // seconds — must match config.SCHEDULER_INTERVAL_SECONDS
const ORDERBOOK_TOP_N    = 20;
const ORDERBOOK_FLUSH    = 60;    // seconds
const ORDERBOOK_MAX_ROWS = 10000;

module.exports = {
  apps: [
    {
      name: 'fastapi',
      script: UVICORN,
      args: 'apps.api.main:app --host 0.0.0.0 --port 8000',
      cwd: REPO,
      interpreter: 'none',
      watch: false,
      restart_delay: 5000,
      max_restarts: 20,
      env: { ...DOT_ENV, NODE_ENV: 'production' },
    },
    {
      name: 'orderbook-collector',
      script: PYTHON,
      args: `services/models/classification/realtime_orderbook_collector.py --symbol ${SYMBOL} --top-n ${ORDERBOOK_TOP_N} --flush-interval ${ORDERBOOK_FLUSH} --max-rows ${ORDERBOOK_MAX_ROWS}`,
      cwd: REPO,
      interpreter: 'none',
      watch: false,
      restart_delay: 5000,
      max_restarts: 20,
    },
    {
      name: 'sentiment-daemon',
      script: PYTHON,
      args: `-u services/models/classification/realtime_sentiment_daemon.py --interval ${SCHEDULER_INTERVAL}`,
      cwd: REPO,
      interpreter: 'none',
      watch: false,
      restart_delay: 10000,
      max_restarts: 20,
      env: { PYTHONUNBUFFERED: '1' },
    },
    {
      name: 'pipeline-scheduler',
      script: PYTHON,
      args: `services/models/classification/realtime_scheduler.py --interval ${SCHEDULER_INTERVAL}`,
      cwd: REPO,
      interpreter: 'none',
      watch: false,
      restart_delay: 5000,
      max_restarts: 20,
    },
    {
      // Weekly auto-retrain: Sundays at 02:00 UTC
      // Runs the full training pipeline and swaps the model if AUC improves.
      name: 'auto-retrain',
      script: PYTHON,
      args: 'services/models/classification/auto_retrain.py',
      cwd: REPO,
      interpreter: 'none',
      watch: false,
      autorestart: false,
      cron_restart: '0 2 * * 0',
      env: { PYTHONUNBUFFERED: '1' },
    },
    {
      name: 'node-backend',
      script: 'dist/server.js',
      cwd: `${REPO}/apps/trading-dashboard/backend`,
      interpreter: 'node',
      watch: false,
      restart_delay: 5000,
      max_restarts: 20,
      env: {
        ...DOT_ENV,
        NODE_ENV: 'production',
        PORT: '3001',
        PYTHON_API: 'http://localhost:8000',
        REALTIME_DIR: `${REPO}/services/models/classification/data/realtime`,
      },
    },
  ],
};
