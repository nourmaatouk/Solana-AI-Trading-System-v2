const REPO = '/home/nourmaatouk/Solana-AI-Trading-System';
const PYTHON = `${REPO}/.venv/bin/python`;
const UVICORN = `${REPO}/.venv/bin/uvicorn`;

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
      env: { NODE_ENV: 'production' },
    },
    {
      name: 'orderbook-collector',
      script: PYTHON,
      args: 'services/models/classification/realtime_orderbook_collector.py --symbol SOLUSDT --top-n 20 --flush-interval 60 --max-rows 10000',
      cwd: REPO,
      interpreter: 'none',
      watch: false,
      restart_delay: 5000,
      max_restarts: 20,
    },
    {
      name: 'pipeline-scheduler',
      script: PYTHON,
      args: 'services/models/classification/realtime_scheduler.py --interval 900',
      cwd: REPO,
      interpreter: 'none',
      watch: false,
      restart_delay: 5000,
      max_restarts: 20,
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
        NODE_ENV: 'production',
        PORT: '3001',
        PYTHON_API: 'http://localhost:8000',
        REALTIME_DIR: `${REPO}/services/models/classification/data/realtime`,
      },
    },
  ],
};
