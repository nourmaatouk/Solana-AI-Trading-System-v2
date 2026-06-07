import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useWallet } from '@solana/wallet-adapter-react';
import { useSystemWallet } from '../hooks/useSystemWallet';
import { fetchExecutionStatus, type ExecutionStatusDto } from '../services/executionApi';
import { useTradingStore } from '../store';
import { EXECUTION_PANEL_REFRESH_MS } from '../config';

/**
 * Thin always-visible bar that tells the user which mode they're in. The label
 * reflects the user's chosen journey (Explore = Paper, Connect Wallet = Devnet);
 * the safety chips (mainnet lock, emergency stop, cluster) come live from
 * /api/execution/status. Includes a control to switch modes.
 */
export function ModeBanner() {
  const uiMode = useTradingStore((s) => s.mode);
  const setMode = useTradingStore((s) => s.setMode);
  const navigate = useNavigate();
  const { connected: phantomConnected } = useWallet();
  const { connected: systemConnected } = useSystemWallet();
  const connected = phantomConnected || systemConnected;
  const [status, setStatus] = useState<ExecutionStatusDto | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const s = await fetchExecutionStatus();
        if (alive) { setStatus(s); setOffline(false); }
      } catch {
        if (alive) setOffline(true);
      }
    };
    load();
    const id = setInterval(load, EXECUTION_PANEL_REFRESH_MS);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const execMode = uiMode;
  const switchTo: 'paper' | 'devnet' = uiMode === 'paper' ? 'devnet' : 'paper';
  // Switching to devnet is only allowed when a wallet is connected.
  const devnetLocked = switchTo === 'devnet' && !connected;
  const doSwitch = () => {
    if (devnetLocked) return;
    setMode(switchTo);
    navigate(switchTo === 'devnet' ? '/execution' : '/trade');
  };
  const mainnetDisabled = status ? !status.allow_mainnet : true;
  const eStop = status?.emergency_stop.enabled ?? false;
  const cluster = status?.wallet.cluster ?? null;

  const meta =
    execMode === 'devnet'
      ? { label: 'DEVNET MODE', color: '#8B5CF6', bg: 'rgba(139,92,246,0.10)',
          explain: 'Real transactions on Solana devnet — a test network, no real money.' }
      : { label: 'PAPER MODE', color: '#0ea5e9', bg: 'rgba(14,165,233,0.10)',
          explain: 'Simulated trading. No wallet and no real funds required.' };

  return (
    <div className="flex items-center gap-3 px-4 py-1.5 text-xs flex-wrap"
      style={{ background: meta.bg, borderBottom: '1px solid var(--border)' }}>
      <span className="font-bold px-2 py-0.5 rounded"
        style={{ color: meta.color, border: `1px solid ${meta.color}`, letterSpacing: '0.04em' }}>
        {meta.label}
      </span>
      <span style={{ color: 'var(--text-2)' }}>{meta.explain}</span>

      {cluster && execMode !== 'paper' && (
        <span className="font-mono px-1.5 py-0.5 rounded" style={{ color: 'var(--text-3)', background: 'var(--bg-sub)' }}>
          {cluster}
        </span>
      )}

      <span className="ml-auto flex items-center gap-2">
        {/* Wallet-required hint — shown only when the user is in paper mode without a wallet */}
        {devnetLocked && (
          <span className="flex items-center gap-1 text-xs"
            style={{ color: 'var(--text-3)' }}>
            <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            Connect your wallet to access Devnet mode
          </span>
        )}
        <button
          onClick={doSwitch}
          disabled={devnetLocked}
          className="px-2 py-0.5 rounded font-semibold transition-colors"
          style={{
            color:      devnetLocked ? 'var(--text-3)' : 'var(--text-2)',
            border:     `1px solid ${devnetLocked ? 'var(--border)' : 'var(--border)'}`,
            background: 'var(--bg-sub)',
            cursor:     devnetLocked ? 'not-allowed' : 'pointer',
            opacity:    devnetLocked ? 0.5 : 1,
          }}
          title={devnetLocked
            ? 'Connect a Solana wallet to unlock Devnet mode'
            : switchTo === 'devnet'
              ? 'Switch to Devnet — real transactions on Solana test network'
              : 'Switch to simulated paper trading (no wallet required)'}>
          {devnetLocked ? '🔒 ' : ''}Switch to {switchTo === 'devnet' ? 'Devnet' : 'Paper'} →
        </button>
        {mainnetDisabled && (
          <span className="px-2 py-0.5 rounded font-semibold"
            style={{ color: 'var(--green)', background: 'var(--green-bg)', border: '1px solid var(--green-bd)' }}
            title="Mainnet execution is hard-disabled by configuration (ALLOW_MAINNET_EXECUTION=false).">
            🔒 MAINNET DISABLED
          </span>
        )}
        {eStop && (
          <span className="px-2 py-0.5 rounded font-semibold"
            style={{ color: 'var(--red)', background: 'var(--red-bg)', border: '1px solid var(--red-bd)' }}>
            ⛔ EMERGENCY STOP
          </span>
        )}
        {offline && (
          <span className="px-2 py-0.5 rounded" style={{ color: 'var(--text-3)' }}
            title="Execution backend not reachable — showing your selected UI mode.">
            backend offline
          </span>
        )}
      </span>
    </div>
  );
}
