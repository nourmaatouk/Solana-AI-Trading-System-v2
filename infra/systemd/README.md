# realtime-orderbook-collector systemd setup

This service keeps `services/models/classification/realtime_orderbook_collector.py`
running in the background so the realtime pipeline can pass `orderbook_preflight`.

The service writes logs to journald and writes health/output files to:

- `services/models/classification/data/realtime/orderbook_status.json`
- `services/models/classification/data/realtime/orderbook_features.csv`

## 1. Prepare Python

On the deployment host, install the project and dependencies. The service file
assumes the app is deployed at `/opt/solana-ai-trading-system`; change the env
file if your path is different.

```bash
cd /opt/solana-ai-trading-system
python3 -m venv .venv
.venv/bin/pip install -r services/models/classification/requirements.txt
```

## 2. Create service user

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin solana-ai
sudo chown -R solana-ai:solana-ai /opt/solana-ai-trading-system
```

If you are running from a development checkout in your home directory, either
change `User=` and `Group=` in the service file to your username or deploy the
repo under `/opt/solana-ai-trading-system`.

## 3. Install service and env file

From the repository root:

```bash
sudo install -o root -g root -m 0644 infra/systemd/realtime-orderbook-collector.service /etc/systemd/system/realtime-orderbook-collector.service
sudo install -o root -g root -m 0644 infra/systemd/realtime-orderbook-collector.env.example /etc/default/realtime-orderbook-collector
sudo nano /etc/default/realtime-orderbook-collector
```

Edit these values if needed:

```text
APP_DIR=/opt/solana-ai-trading-system
PYTHON_BIN=/opt/solana-ai-trading-system/.venv/bin/python
ORDERBOOK_SYMBOL=solusdt
ORDERBOOK_TOP_N=10
ORDERBOOK_FLUSH_INTERVAL=60
ORDERBOOK_MAX_ROWS=180
```

## 4. Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable realtime-orderbook-collector
sudo systemctl start realtime-orderbook-collector
```

## 5. Operate the service

```bash
sudo systemctl status realtime-orderbook-collector
sudo systemctl restart realtime-orderbook-collector
sudo systemctl stop realtime-orderbook-collector
journalctl -u realtime-orderbook-collector -f
```

## 6. Validate health

```bash
cat services/models/classification/data/realtime/orderbook_status.json
tail -n 5 services/models/classification/data/realtime/orderbook_features.csv
```

Expected status fields:

```json
{
  "status": "running",
  "running": true,
  "connected": true,
  "source": "binance_partial_depth",
  "latest_orderbook_timestamp": "...",
  "last_error": null
}
```

Then run:

```bash
python services/models/classification/run_realtime_pipeline.py --skip-sentiment --fusion --fusion-aggregate --paper-trade --timeout 60
```

The `orderbook_preflight` step should pass before model scoring.

## Troubleshooting

| Problem | Check | Fix |
|---|---|---|
| Python path is wrong | `journalctl -u realtime-orderbook-collector -n 50` shows missing executable | Edit `PYTHON_BIN` in `/etc/default/realtime-orderbook-collector`; install dependencies |
| Working directory is wrong | Logs show `cd: ... No such file or directory` | Edit `APP_DIR` in `/etc/default/realtime-orderbook-collector` |
| `orderbook_status.json` is missing | `systemctl status realtime-orderbook-collector` | Start service; check file permissions on `services/models/classification/data/realtime` |
| Status says `stopped` | `sudo systemctl status realtime-orderbook-collector` | Start or restart the service |
| Status says `error` | `journalctl -u realtime-orderbook-collector -n 100` | Inspect `last_error`; usually network/WebSocket or dependency issue |
| Binance WebSocket fails | journal shows connection errors | Confirm network/DNS/firewall access to `wss://stream.binance.com:9443` |
| Pipeline says orderbook is stale | compare `latest_orderbook_timestamp` with current UTC time | Restart service and wait for the next flush |
| Permission denied writing CSV/status | journal shows `PermissionError` | `sudo chown -R solana-ai:solana-ai /opt/solana-ai-trading-system/services/models/classification/data/realtime` |
