---
name: run-app
description: Launch and drive the Solvanta AI trading dashboard end-to-end — Node backend (Express + WebSocket on :3001) plus the Vite frontend (:5173). Use this when asked to run, start, launch, or screenshot the app, or to verify a change works in the real running stack rather than just tests. Also covers the rebuild ordering between services/execution and the backend.
---

# Running the Solvanta AI Trading Dashboard

The project is a three-piece stack:

- `services/execution` — TypeScript library, compiles to `dist/`. The Node backend imports it via the path alias `@trading/execution` → `services/execution/dist/index`. **Stale dist is a real footgun:** the backend will silently run old Guardian / workflow code if you forget to rebuild here first.
- `apps/trading-dashboard/backend` — Express + WebSocket server on port `3001`. Serves `/api/*` routes and (in production) the built frontend at `/`.
- `apps/trading-dashboard/frontend` — Vite + React. Dev server on `:5173` proxies `/api/*` and `/ws` to the backend.

A Python FastAPI service at `:8000` is referenced by `PYTHON_API` for live signal data. The dashboard runs without it — pages render, state files work, Guardian-policy reads work — but `POST /api/execution/validate` and `/execute` return `{error: "validation_failed", detail: "fetch failed"}` until it's up. Launch it with:

```bash
.venv/bin/uvicorn apps.api.main:app --port 8000 --host 127.0.0.1
```

Or if you only need the dashboard surface (no `/validate` / `/execute`), skip it.

## Launch sequence (verified)

Run these in order. Every step has a single command — no `cd` chains, no manual env juggling.

### 1. Make sure the execution package is built

```bash
cd /home/nourmaatouk/Solana-AI-Trading-System/services/execution
npm run build
```

This regenerates `services/execution/dist/`. Skip this and the backend will type-check fine but run yesterday's code at runtime. If you've only touched files under `apps/`, you can skip step 1.

### 2. Start the backend in the background

```bash
cd /home/nourmaatouk/Solana-AI-Trading-System/apps/trading-dashboard/backend
PORT=3001 npx ts-node-dev -r tsconfig-paths/register src/server.ts
```

Wait for: `[server] Trading dashboard backend listening on port 3001`.
You should also see two banners that prove the wiring is alive:

```
[signal-poller] Started, polling every 10 s
[auto-executor] Started — event-driven + poll fallback every 30s, dedup window 300s, wallet=phantom
```

If you see `EADDRINUSE` on 3001, an old `ts-node-dev` is still running — `pkill -f "ts-node-dev.*server.ts"` and retry.

### 3. Start the frontend in the background

```bash
cd /home/nourmaatouk/Solana-AI-Trading-System/apps/trading-dashboard/frontend
npx vite --port 5173 --host 127.0.0.1
```

Wait for: `VITE v5.x ready` and `Local: http://127.0.0.1:5173/`.

### 4. Smoke-test the chain

These are the calls I trust to prove "the app is actually running" — they exercise the path from Vite → proxy → backend → state files → Guardian. Run them and check the table below.

```bash
curl -s http://localhost:3001/health
curl -s http://127.0.0.1:5173/api/execution/auto-execute/status
curl -s http://127.0.0.1:5173/api/guardian/policy | python3 -m json.tool
```

| What | Expected |
| --- | --- |
| `/health` (direct backend) | `{"status":"ok",...}` |
| `/api/execution/auto-execute/status` (via Vite) | `{enabled, user_risk_level, max_trade_size_sol}` — three keys present, types correct |
| `/api/guardian/policy` | must include `allowHighRiskTrades` and `defaultUserRiskLevel` — if either is missing, the dist is stale; go back to step 1 |

### 5. (Optional) Drive the UI

```bash
chromium --headless --disable-gpu --no-sandbox --hide-scrollbars \
  --window-size=1400,1100 --virtual-time-budget=4000 \
  --screenshot=/tmp/dashboard.png "http://127.0.0.1:5173/"

chromium --headless --disable-gpu --no-sandbox --hide-scrollbars \
  --window-size=1400,1100 --virtual-time-budget=4000 \
  --screenshot=/tmp/execution.png "http://127.0.0.1:5173/execution"
```

Read the screenshots with the Read tool. **A blank or all-dark frame is a failure** — the dev server probably hasn't finished compiling yet; wait another 2 seconds and retry.

What you should see on `/execution`:
- "Execution" header + paper/devnet/mainnet mode badge
- "System Status" card with your risk limit + trading mode (driven by Settings, defaults to MEDIUM + Manual)
- "Auto-execution" card with eight fields: Selected mode, Backend enabled, Backend risk, Backend trade size, Executing, Last signal key, Last validated, Last auto-executed, Current signal status
- "Wallet — Not connected" + "Connect Wallet" button
- Empty "Trade history" with "No executions yet" placeholder

## Exercising the new auto-execution endpoints

The endpoints below are what the Settings page calls when the user clicks Save. Worth driving when verifying changes to the auto-execution chain:

```bash
# Set the user's risk preference (LOW | MEDIUM | HIGH)
curl -s -X POST http://localhost:3001/api/execution/user-risk-level \
  -H 'Content-Type: application/json' -d '{"level":"LOW"}'

# Set the per-trade size cap (must be > 0 and ≤ 10)
curl -s -X POST http://localhost:3001/api/execution/max-trade-size \
  -H 'Content-Type: application/json' -d '{"amount_sol":0.07}'

# Toggle backend auto-execution
curl -s -X POST http://localhost:3001/api/execution/auto-execute/enable
curl -s -X POST http://localhost:3001/api/execution/auto-execute/disable

# Verify they all landed
curl -s http://localhost:3001/api/execution/auto-execute/status
```

These write to `services/models/classification/data/realtime/{user_risk_level,max_trade_size_sol,auto_execute}.json`. The whole `**/data/realtime/` tree is gitignored.

Invalid input is rejected with HTTP 400 + a zod issue list (e.g. `{"level":"EXTREME"}` → 400, `{"amount_sol":50}` → 400). If those return 500 or pass, something has regressed in the route validation.

## Running tests

```bash
cd services/execution && npm test                          # 63 tests
cd apps/trading-dashboard/backend && npm test              # 9 tests (uses vitest.config.ts alias)
```

`backend/npm test` exercises the SignalPoller -> AutoExecutor wire end-to-end
against a temp REALTIME_DIR. Set REALTIME_DIR in env to point any state-file
reads (`readUserRiskLevel`, `readDailyVolume`, etc.) at a different directory.

## Shutdown

```bash
pkill -f "ts-node-dev.*server.ts"
pkill -f "vite"
pkill -f "uvicorn.*apps.api"
```

`pkill` may exit 144 — that's fine, it just means the process was killed.

## Known quirks

- `AbortSignal.timeout(...)` in [frontend/src/services/executionApi.ts](apps/trading-dashboard/frontend/src/services/executionApi.ts) needs Chrome 103+ / Firefox 100+. Older headless chromium throws "AbortSignal.timeout is not a function" — the UI degrades gracefully (shows "(server default)" / "(env fallback)" placeholders) but no real fetch happens. Real browsers work.
- The backend emits `execution.mainnet.constructed_but_disabled` at startup — that's the `MainnetExecutor` being instantiated as a no-op stub. Not a problem; mainnet stays gated behind `ALLOW_MAINNET_EXECUTION=true`.
- `validate` / `execute` require the FastAPI service at `:8000`. Without it: `{error:"validation_failed", detail:"fetch failed"}` — that's the signal client, not a Guardian failure.

## When to update this skill

- If `services/execution/dist/` ever stops being the runtime source of truth (e.g. project moves to a bundler, or paths alias changes), step 1 may become unnecessary.
- If new state files appear under `**/data/realtime/`, add them to the smoke-test table.
- If a new endpoint joins `/api/execution/`, add a curl line under "Exercising the new auto-execution endpoints" so the next run validates it too.
