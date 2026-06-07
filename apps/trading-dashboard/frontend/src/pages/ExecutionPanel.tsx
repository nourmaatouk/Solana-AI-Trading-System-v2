import { useEffect, useState } from 'react';
import { useWallet, useConnection } from '@solana/wallet-adapter-react';
import { VersionedTransaction } from '@solana/web3.js';
import {
  type ExecutionStatusDto,
  type ExecutionWorkflowDto,
  type GuardianPolicyDto,
  type AutoExecuteStatusDto,
  enableEmergencyStop,
  executeTrade,
  prepareTrade,
  fetchExecutionStatus,
  fetchAutoExecuteStatus,
  fetchGuardianPolicy,
  resumeFromEmergencyStop,
  validateExecution,
} from '../services/executionApi';
import { GuardianBadge } from '../components/GuardianBadge';
import { ExecutionHistory } from '../components/ExecutionHistory';
import { WalletStatusCard } from '../components/WalletStatusCard';
import { DevnetBalanceCard } from '../components/DevnetBalanceCard';
import { EXECUTION_PANEL_REFRESH_MS, EXECUTION_TRADE_DEFAULT_SOL, WALLET_MODE } from '../config';
import { useSettings } from '../hooks/useSettings';
import { useAutoExecution } from '../hooks/useAutoExecution';
import { explorerTxUrl } from '../utils/solanaExplorer';

const card: React.CSSProperties = {
  background: 'var(--bg-panel)',
  border: '1px solid var(--border)',
  borderRadius: 12,
  padding: 20,
};

const MODE_STYLES: Record<string, { bg: string; fg: string; label: string }> = {
  paper:   { bg: 'var(--bg-sub)',  fg: 'var(--text-1)', label: 'PAPER' },
  devnet:  { bg: 'var(--accent-bg)', fg: 'var(--accent)', label: 'DEVNET' },
  mainnet: { bg: 'var(--red-bg)',  fg: 'var(--red)',     label: 'MAINNET' },
};

export function ExecutionPanel() {
  const { publicKey, connected, sendTransaction } = useWallet();
  const { connection } = useConnection();
  // "Offer both": sign with the connected wallet when one is present; otherwise
  // the backend keypair signs. Controls are never gated — backend signing is the
  // default, so connecting a wallet is optional.
  const phantomKey = connected ? (publicKey?.toBase58() ?? null) : null;
  const walletGated = false;
  const userSettings = useSettings();

  const [status, setStatus] = useState<ExecutionStatusDto | null>(null);
  const [policy, setPolicy] = useState<GuardianPolicyDto | null>(null);
  const [autoStatus, setAutoStatus] = useState<AutoExecuteStatusDto | null>(null);
  const [amount, setAmount] = useState<number>(
    () => userSettings.maxTradeSizeSol ?? EXECUTION_TRADE_DEFAULT_SOL,
  );
  const [busy, setBusy] = useState<'idle' | 'validate' | 'execute' | 'signing'>('idle');
  const [confirmResume, setConfirmResume] = useState(false);
  const [last, setLast] = useState<ExecutionWorkflowDto | null>(null);
  const [error, setError] = useState<string | null>(null);
  const autoExec = useAutoExecution();

  async function refresh() {
    try {
      const [s, p, a] = await Promise.all([
        fetchExecutionStatus(),
        fetchGuardianPolicy(),
        fetchAutoExecuteStatus(),
      ]);
      setStatus(s);
      setPolicy(p);
      setAutoStatus(a);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, EXECUTION_PANEL_REFRESH_MS);
    return () => clearInterval(id);
  }, []);

  async function onValidate() {
    setBusy('validate');
    setError(null);
    try {
      const r = await validateExecution({
        amount_sol: amount,
        phantom_public_key: phantomKey ?? undefined,
        user_risk_level: userSettings.riskLevel,
      });
      setLast({
        request_received_at_utc: new Date().toISOString(),
        request_completed_at_utc: new Date().toISOString(),
        guardian: r.guardian,
        execution: null,
        narration: null,
      });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy('idle');
    }
  }

  async function onExecute() {
    setBusy('execute');
    setError(null);
    try {
      const mode = status?.execution_mode ?? 'paper';
      const usePhantomSigning = connected && mode !== 'paper';

      if (!usePhantomSigning) {
        // Paper mode or non-phantom wallet: backend signs and submits directly.
        const r = await executeTrade({
          amount_sol: amount,
          phantom_public_key: phantomKey ?? undefined,
          user_risk_level: userSettings.riskLevel,
        });
        setLast(r);
        refresh();
        return;
      }

      // ── Phantom signing flow ─────────────────────────────────────────────
      // 1. Guardian-validate + get unsigned transaction from backend.
      const prepared = await prepareTrade({
        amount_sol: amount,
        phantom_public_key: phantomKey ?? undefined,
        user_risk_level: userSettings.riskLevel,
      });

      if (!prepared.guardian.approved || !prepared.transaction_base64) {
        // Blocked by Guardian — show the verdict, nothing to sign.
        setLast({
          request_received_at_utc: new Date().toISOString(),
          request_completed_at_utc: new Date().toISOString(),
          guardian: prepared.guardian,
          execution: null,
          narration: null,
        });
        return;
      }

      // 2. Deserialize and send to Phantom for signing + submission.
      setBusy('signing');
      const txBuffer = Buffer.from(prepared.transaction_base64, 'base64');
      const transaction = VersionedTransaction.deserialize(new Uint8Array(txBuffer));
      const sig = await sendTransaction(transaction, connection);

      // 3. Confirm on-chain.
      const { blockhash, lastValidBlockHeight } = await connection.getLatestBlockhash();
      await connection.confirmTransaction({ signature: sig, blockhash, lastValidBlockHeight });

      // 4. Show result.
      setLast({
        request_received_at_utc: new Date().toISOString(),
        request_completed_at_utc: new Date().toISOString(),
        guardian: prepared.guardian,
        execution: {
          status: 'confirmed',
          mode: prepared.execution_mode,
          signature: sig,
          block_height: null,
          asset: prepared.asset,
          side: prepared.side,
          amount_sol: prepared.amount_sol,
          rpc_url: null,
          submitted_at_utc: new Date().toISOString(),
          confirmed_at_utc: new Date().toISOString(),
          error: null,
        },
        narration: null,
      });
      refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy('idle');
    }
  }

  async function onEmergencyStop() {
    await enableEmergencyStop('manual');
    refresh();
  }

  async function onResume() {
    if (!confirmResume) {
      setConfirmResume(true);
      return;
    }
    setConfirmResume(false);
    await resumeFromEmergencyStop();
    refresh();
  }

  const mode = status?.execution_mode ?? 'paper';
  const modeStyle = MODE_STYLES[mode] ?? MODE_STYLES.paper;
  const eStop = status?.emergency_stop.enabled ?? false;

  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: 'var(--bg-page)' }}>
      <div className="max-w-3xl mx-auto space-y-5">
        {/* Header */}
        <div>
          <h1 className="text-xl font-bold mb-1" style={{ color: 'var(--text-1)' }}>
            Execution
          </h1>
          <p className="text-sm" style={{ color: 'var(--text-3)' }}>
            Validated trading signals → Guardian → Solana. Mainnet is disabled by default.
          </p>
        </div>

        {error && (
          <div
            className="px-4 py-3 text-sm rounded-xl"
            style={{ background: 'var(--red-bg)', border: '1px solid var(--red-bd)', color: 'var(--red)' }}
          >
            {error}
          </div>
        )}

        {/* Devnet test-network disclaimer */}
        {mode === 'devnet' && (
          <div className="px-4 py-2 text-xs rounded-xl"
            style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)', color: 'var(--accent)' }}>
            <strong>DEVNET</strong> — these are real on-chain transactions on Solana's <strong>test network</strong>.
            No real money is involved. Mainnet stays disabled unless explicitly enabled in backend config.
          </div>
        )}

        {/* Phantom wallet connection + balance */}
        <WalletStatusCard />
        <DevnetBalanceCard />

        {/* Signer info — backend signs by default; connecting a wallet is optional */}
        {mode !== 'paper' && (
          <div
            className="px-4 py-2 text-xs rounded-xl"
            style={{ background: 'var(--bg-sub)', border: '1px solid var(--border)', color: 'var(--text-3)' }}
          >
            {connected
              ? 'A wallet is connected — manual trades will be signed by you in the wallet popup.'
              : 'No wallet connected — the backend keypair signs trades by default. Connect a wallet above to sign them yourself instead.'}
          </div>
        )}

        {/* Mode + System status */}
        <div style={card}>
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-semibold" style={{ color: 'var(--text-1)' }}>System Status</div>
            <span
              className="text-xs font-bold px-2 py-1 rounded-full"
              style={{ background: modeStyle.bg, color: modeStyle.fg }}
            >
              MODE · {modeStyle.label}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <Field label="Cluster" value={status?.wallet.cluster ?? '—'} />
            <Field
              label="Wallet"
              value={
                phantomKey
                  ? `${phantomKey.slice(0, 6)}…${phantomKey.slice(-6)} (Phantom)`
                  : (status?.wallet.public_key ?? '(no key configured)')
              }
              mono
            />
            <Field
              label="Balance"
              value={status?.wallet.balance_sol != null ? `${status.wallet.balance_sol.toFixed(4)} SOL` : '—'}
            />
            <Field label="Signing" value={status?.wallet.signing_capable ? 'enabled' : 'read-only'} />
            <Field label="Allow mainnet" value={status?.allow_mainnet ? 'YES' : 'no'} />
            <Field label="Emergency stop" value={eStop ? 'ACTIVE' : 'off'} highlight={eStop} />
            <Field label="Your risk limit" value={userSettings.riskLevel} />
            <Field label="Trading mode" value={userSettings.tradingMode} />
          </div>
        </div>

        {/* Trade controls */}
        <div style={card}>
          <div className="text-sm font-semibold mb-3" style={{ color: 'var(--text-1)' }}>Request a trade</div>
          <div className="flex items-end gap-3">
            <label className="flex-1">
              <div className="text-xs mb-1" style={{ color: 'var(--text-3)' }}>Amount (SOL)</div>
              <input
                type="number"
                step={0.01}
                min={0}
                max={policy?.policy.maxTradeSizeSol}
                value={amount}
                onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
                className="w-full px-3 py-2 rounded-lg text-sm font-mono"
                style={{ background: 'var(--bg-sub)', border: '1px solid var(--border)', color: 'var(--text-1)' }}
              />
            </label>
            <button
              onClick={onValidate}
              disabled={busy !== 'idle' || eStop || walletGated}
              className="px-4 py-2 rounded-lg text-sm font-semibold disabled:opacity-50"
              style={{ background: 'var(--bg-sub)', border: '1px solid var(--border)', color: 'var(--text-1)' }}
            >
              {busy === 'validate' ? 'Validating…' : 'Validate'}
            </button>
            <button
              onClick={onExecute}
              disabled={busy !== 'idle' || eStop || walletGated}
              className="px-4 py-2 rounded-lg text-sm font-semibold disabled:opacity-50"
              style={{ background: 'var(--accent)', color: 'var(--bg-page)' }}
            >
              {busy === 'execute' ? 'Preparing…' : busy === 'signing' ? 'Sign in Phantom…' : 'Execute'}
            </button>
          </div>
          {policy && (
            <div className="text-[11px] mt-2" style={{ color: 'var(--text-3)' }}>
              Max trade size: <span className="font-mono">{policy.policy.maxTradeSizeSol} SOL</span>
              {' · '}Min confidence: <span className="font-mono">{policy.policy.confidenceThreshold}</span>
              {' · '}Max spoofing: <span className="font-mono">{policy.policy.maxSpoofingConfidence}</span>
              {' · '}Allowed assets: <span className="font-mono">{policy.policy.allowedAssets.join(', ')}</span>
              {' · '}Your risk limit: <span className="font-mono">{userSettings.riskLevel}</span>
            </div>
          )}
        </div>

        {/* Latest result */}
        {last && (
          <div style={card}>
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm font-semibold" style={{ color: 'var(--text-1)' }}>Latest result</div>
              <GuardianBadge result={last.guardian} />
            </div>
            {last.guardian.blocked_by === 'USER_RISK_LEVEL_EXCEEDED' && (
              <div
                className="text-xs px-3 py-2 rounded-lg mb-2"
                style={{ background: 'var(--red-bg)', color: 'var(--red)', border: '1px solid var(--red-bd)' }}
              >
                Blocked by your risk preference ({userSettings.riskLevel}). Change it in Settings to allow higher-risk signals.
              </div>
            )}
            <ul className="text-xs space-y-1" style={{ color: 'var(--text-2)' }}>
              {last.guardian.reasons.map((r, i) => (
                <li key={i}>· {r}</li>
              ))}
            </ul>
            {last.execution && (
              <div className="mt-3 rounded-lg p-3 text-xs space-y-1"
                style={{ background: 'var(--bg-sub)', color: 'var(--text-2)' }}>
                <Field label="Status" value={last.execution.status} />
                <Field label="Mode" value={last.execution.mode} />
                {last.execution.signature && (
                  <div className="flex items-center justify-between gap-3">
                    <span style={{ color: 'var(--text-3)' }}>Signature</span>
                    <a
                      href={explorerTxUrl(last.execution.signature, last.execution.mode)}
                      target="_blank"
                      rel="noreferrer"
                      className="font-mono hover:underline truncate"
                      style={{ color: 'var(--accent)' }}
                      title="View on Solana Explorer"
                    >
                      {last.execution.signature.slice(0, 8)}…{last.execution.signature.slice(-8)} ↗
                    </a>
                  </div>
                )}
                {last.execution.error && <Field label="Error" value={last.execution.error} />}
              </div>
            )}
            {last.narration && (
              <div className="mt-3 rounded-lg p-3" style={{ background: 'var(--bg-sub)', border: '1px solid var(--border)' }}>
                <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider mb-1" style={{ color: 'var(--accent)' }}>
                  <span>✦</span> AI Narrator · Sentinel
                </div>
                <p className="text-xs italic" style={{ color: 'var(--text-2)' }}>"{last.narration}"</p>
              </div>
            )}
          </div>
        )}

        {/* Auto-execution status */}
        <div style={card}>
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-semibold" style={{ color: 'var(--text-1)' }}>Auto-execution</div>
            <span
              className="text-xs font-bold px-2 py-1 rounded-full"
              style={{
                background: userSettings.tradingMode === 'Manual' ? 'var(--bg-sub)' : 'var(--accent-bg)',
                color: userSettings.tradingMode === 'Manual' ? 'var(--text-3)' : 'var(--accent)',
              }}
            >
              {userSettings.tradingMode.toUpperCase()}
            </span>
          </div>

          {/* Arm toggle — Phantom auto-signing only fires after arming */}
          {WALLET_MODE === 'phantom' && userSettings.tradingMode === 'Auto' && (
            <div
              className="flex items-center justify-between gap-3 mb-3 px-3 py-2 rounded-lg"
              style={{
                background: autoExec.armed ? 'var(--green-bg)' : 'var(--bg-sub)',
                border: `1px solid ${autoExec.armed ? 'var(--green-bd)' : 'var(--border)'}`,
              }}
            >
              <div className="text-xs" style={{ color: autoExec.armed ? 'var(--green)' : 'var(--text-2)' }}>
                {autoExec.armed
                  ? 'ARMED — next BUY/SELL signal will trigger a Phantom signing prompt'
                  : 'Disarmed — Phantom signing will NOT be triggered automatically'}
              </div>
              <button
                onClick={() => (autoExec.armed ? autoExec.disarm() : autoExec.arm())}
                disabled={!connected}
                className="px-3 py-1 rounded-lg text-xs font-semibold disabled:opacity-50"
                style={{
                  background: autoExec.armed ? 'var(--red)' : 'var(--green)',
                  color: 'var(--bg-page)',
                }}
              >
                {autoExec.armed ? 'Disarm' : 'Arm'}
              </button>
            </div>
          )}

          <div className="grid grid-cols-2 gap-2 text-xs">
            <Field label="Selected mode" value={userSettings.tradingMode} />
            <Field
              label="Backend enabled"
              value={autoStatus?.enabled ? 'yes' : 'no'}
              highlight={autoStatus?.enabled && userSettings.tradingMode !== 'Auto'}
            />
            <Field
              label="Backend risk"
              value={autoStatus?.user_risk_level ?? '(server default)'}
            />
            <Field
              label="Backend trade size"
              value={
                autoStatus?.max_trade_size_sol != null
                  ? `${autoStatus.max_trade_size_sol} SOL`
                  : '(env fallback)'
              }
            />
            <Field
              label="Executing"
              value={autoExec.executing ? 'yes' : 'no'}
              highlight={autoExec.executing}
            />
            <Field
              label="Last signal key"
              value={autoExec.lastProcessedKey ? `${autoExec.lastProcessedKey.slice(0, 28)}…` : '—'}
              mono
            />
            <Field label="Last validated" value={autoExec.lastValidatedAt ?? '—'} />
            <Field label="Last auto-executed" value={autoExec.lastAutoExecutedAt ?? '—'} />
            <Field
              label="Current signal status"
              value={autoExec.skippedReason ?? (autoExec.executing ? 'processing…' : '—')}
            />
          </div>

          {/* Warn when frontend + backend are out of sync */}
          {autoStatus && userSettings.tradingMode === 'Auto' && !autoStatus.enabled && (
            <div className="mt-3 text-xs px-3 py-2 rounded-lg"
              style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)', color: 'var(--accent)' }}>
              Trading Mode is Auto but the backend AutoExecutor is disabled. Save Settings to enable it.
            </div>
          )}
          {autoStatus &&
            autoStatus.user_risk_level &&
            autoStatus.user_risk_level !== userSettings.riskLevel && (
            <div className="mt-2 text-xs px-3 py-2 rounded-lg"
              style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)', color: 'var(--accent)' }}>
              Your selected risk ({userSettings.riskLevel}) doesn't match the backend's ({autoStatus.user_risk_level}). Save Settings to sync.
            </div>
          )}

          {autoExec.validationResult && (
            <div className="mt-3">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs" style={{ color: 'var(--text-3)' }}>Last validation</span>
                <GuardianBadge result={autoExec.validationResult} />
              </div>
              {autoExec.validationResult.blocked_by === 'USER_RISK_LEVEL_EXCEEDED' && (
                <div
                  className="text-xs px-3 py-2 rounded-lg mb-2"
                  style={{ background: 'var(--red-bg)', color: 'var(--red)', border: '1px solid var(--red-bd)' }}
                >
                  Auto-validation blocked by your risk preference ({userSettings.riskLevel}).
                </div>
              )}
              <ul className="text-xs space-y-1 mt-1" style={{ color: 'var(--text-2)' }}>
                {autoExec.validationResult.reasons.map((r, i) => (
                  <li key={i}>· {r}</li>
                ))}
              </ul>
            </div>
          )}
          {autoExec.autoExecResult && (
            <div className="mt-3 rounded-lg p-3 text-xs space-y-1"
              style={{ background: 'var(--bg-sub)', color: 'var(--text-2)' }}>
              <Field label="Status" value={autoExec.autoExecResult.status} />
              <Field label="Mode" value={autoExec.autoExecResult.mode} />
              {autoExec.autoExecResult.signature && (
                <Field label="Signature" value={autoExec.autoExecResult.signature} mono />
              )}
            </div>
          )}
        </div>

        {/* Execution history */}
        <div style={card}>
          <div className="text-sm font-semibold mb-3" style={{ color: 'var(--text-1)' }}>Trade history</div>
          <ExecutionHistory />
        </div>

        {/* Emergency stop controls */}
        <div style={card}>
          <div className="text-sm font-semibold mb-3" style={{ color: 'var(--text-1)' }}>Emergency stop</div>
          <p className="text-xs mb-3" style={{ color: 'var(--text-3)' }}>
            When active, every Guardian validation immediately returns BLOCK with reason EMERGENCY_STOP.
            The flag is persisted across restarts.
          </p>
          {eStop ? (
            <div className="flex items-center gap-3 flex-wrap">
              <button
                onClick={onResume}
                className="px-4 py-2 rounded-lg text-sm font-semibold"
                style={{ background: confirmResume ? 'var(--red)' : 'var(--green)', color: 'var(--bg-page)' }}
              >
                {confirmResume ? 'Confirm resume?' : 'Resume execution'}
              </button>
              {confirmResume && (
                <button
                  onClick={() => setConfirmResume(false)}
                  className="px-4 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-sub)', border: '1px solid var(--border)', color: 'var(--text-2)' }}
                >
                  Cancel
                </button>
              )}
              {confirmResume && (
                <span className="text-xs" style={{ color: 'var(--text-3)' }}>
                  Guardian rules stay active after resume.
                </span>
              )}
            </div>
          ) : (
            <button
              onClick={onEmergencyStop}
              className="px-4 py-2 rounded-lg text-sm font-semibold"
              style={{ background: 'var(--red)', color: 'var(--bg-page)' }}
            >
              Stop all execution
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function Field({ label, value, mono, highlight }: { label: string; value: string; mono?: boolean; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span style={{ color: 'var(--text-3)' }}>{label}</span>
      <span
        className={mono ? 'font-mono' : ''}
        style={{ color: highlight ? 'var(--red)' : 'var(--text-1)', overflow: 'hidden', textOverflow: 'ellipsis' }}
      >
        {value}
      </span>
    </div>
  );
}
