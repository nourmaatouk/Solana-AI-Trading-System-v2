module.exports = {
  apps: [
    {
      name: "api",
      script: ".venv/bin/uvicorn",
      args: "apps.api.main:app --host 127.0.0.1 --port 8000",
      cwd: "/home/nourmaatouk/Solana-AI-Trading-System",
      interpreter: "none",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    },
    {
      name: "node-backend",
      script: "npm",
      args: "start",
      cwd: "/home/nourmaatouk/Solana-AI-Trading-System/apps/trading-dashboard/backend"
    },
    {
      name: "frontend",
      script: "npm",
      args: "run dev",
      cwd: "/home/nourmaatouk/Solana-AI-Trading-System/apps/trading-dashboard/frontend"
    },
    {
      name: "orderbook-collector",
      script: "services/models/classification/realtime_orderbook_collector.py",
      interpreter: ".venv/bin/python",
      cwd: "/home/nourmaatouk/Solana-AI-Trading-System",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    },
    {
      name: "sentiment-daemon",
      script: "services/models/classification/realtime_sentiment_daemon.py",
      interpreter: ".venv/bin/python",
      cwd: "/home/nourmaatouk/Solana-AI-Trading-System",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    },
    {
      name: "scheduler",
      script: "services/models/classification/realtime_scheduler.py",
      interpreter: ".venv/bin/python",
      cwd: "/home/nourmaatouk/Solana-AI-Trading-System",
      env: {
        PYTHONUNBUFFERED: "1"
      }
    }
  ]
};
