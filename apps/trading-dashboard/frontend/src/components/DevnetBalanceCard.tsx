import { useCallback, useEffect, useState } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import {
  type WalletBalanceDto,
  fetchWalletBalance,
  requestDevnetAirdrop,
} from '../services/executionApi';
import { SOLANA_CLUSTER, WALLET_MODE } from '../config';

const card: React.CSSProperties = {
  background: 'var(--bg-panel)',
  border: '1px solid var(--border)',
  borderRadius: 12,
  padding: 20,
};

/**
 * Shows the connected wallet's SOL balance (fetched from the backend RPC proxy)
 * and, on devnet only, a one-click airdrop button.
 *
 * Renders nothing when WALLET_MODE !== 'phantom' or wallet is disconnected.
 */
export function DevnetBalanceCard() {
  const { publicKey, connected } = useWallet();
  const [balance, setBalance] = useState<WalletBalanceDto | null>(null);
  const [loadingBalance, setLoadingBalance] = useState(false);
  const [airdropBusy, setAirdropBusy] = useState(false);
  const [airdropResult, setAirdropResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const keyStr = publicKey?.toBase58() ?? null;
  const isDevnet = SOLANA_CLUSTER === 'devnet';

  const loadBalance = useCallback(async () => {
    if (!keyStr) return;
    setLoadingBalance(true);
    setError(null);
    try {
      const data = await fetchWalletBalance(keyStr);
      setBalance(data);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoadingBalance(false);
    }
  }, [keyStr]);

  useEffect(() => {
    if (connected && keyStr) {
      loadBalance();
    } else {
      setBalance(null);
    }
  }, [connected, keyStr, loadBalance]);

  async function onAirdrop() {
    if (!keyStr) return;
    setAirdropBusy(true);
    setAirdropResult(null);
    setError(null);
    try {
      const result = await requestDevnetAirdrop(keyStr);
      setAirdropResult(`Airdropped ${result.amount_sol} SOL — tx: ${result.signature.slice(0, 12)}…`);
      await loadBalance();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setAirdropBusy(false);
    }
  }

  if (WALLET_MODE !== 'phantom' || !connected || !keyStr) return null;

  return (
    <div style={card}>
      <div className="flex items-center justify-between mb-3">
        <div className="text-sm font-semibold" style={{ color: 'var(--text-1)' }}>Balance</div>
        <button
          onClick={loadBalance}
          disabled={loadingBalance}
          className="text-xs px-2 py-1 rounded-lg transition-colors disabled:opacity-40"
          style={{ color: 'var(--text-3)', border: '1px solid var(--border)' }}
          onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-sub)')}
          onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          title="Refresh balance"
        >
          {loadingBalance ? '…' : '↻ Refresh'}
        </button>
      </div>

      {error && (
        <div
          className="mb-3 px-3 py-2 rounded-lg text-xs"
          style={{ background: 'var(--red-bg)', border: '1px solid var(--red-bd)', color: 'var(--red)' }}
        >
          {error}
        </div>
      )}

      <div className="flex items-baseline gap-2 mb-3">
        <span className="text-2xl font-bold font-mono" style={{ color: 'var(--text-1)' }}>
          {balance != null ? balance.balance_sol.toFixed(4) : '—'}
        </span>
        <span className="text-sm font-semibold" style={{ color: 'var(--text-3)' }}>SOL</span>
        {balance && (
          <span className="text-xs ml-auto" style={{ color: 'var(--text-3)' }}>
            {balance.cluster}
          </span>
        )}
      </div>

      {airdropResult && (
        <div
          className="mb-3 px-3 py-2 rounded-lg text-xs"
          style={{ background: 'var(--green-bg)', border: '1px solid var(--green-bd)', color: 'var(--green)' }}
        >
          {airdropResult}
        </div>
      )}

      {isDevnet && (
        <button
          onClick={onAirdrop}
          disabled={airdropBusy}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold disabled:opacity-50 transition-colors"
          style={{
            background: 'var(--bg-sub)',
            border: '1px solid var(--border)',
            color: 'var(--text-1)',
          }}
          onMouseEnter={(e) => !airdropBusy && (e.currentTarget.style.background = 'var(--border)')}
          onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--bg-sub)')}
        >
          {airdropBusy ? 'Requesting…' : '⬇ Devnet Airdrop'}
        </button>
      )}
    </div>
  );
}
