import { useState, useRef, useEffect } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { useWalletModal } from '@solana/wallet-adapter-react-ui';
import { WALLET_MODE } from '../config';

function shortKey(key: string): string {
  return `${key.slice(0, 4)}…${key.slice(-4)}`;
}

/**
 * Shows "Connect Wallet" when disconnected, or the wallet name + shortened key
 * with a disconnect option when connected.
 *
 * Renders nothing when WALLET_MODE !== 'phantom'.
 */
export function WalletConnectionButton() {
  const { publicKey, connected, disconnect, wallet, connecting } = useWallet();
  const { setVisible } = useWalletModal();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    function onOutsideClick(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', onOutsideClick);
    return () => document.removeEventListener('mousedown', onOutsideClick);
  }, [menuOpen]);

  if (WALLET_MODE !== 'phantom') return null;

  if (connecting) {
    return (
      <div
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs"
        style={{ background: 'var(--bg-sub)', border: '1px solid var(--border)', color: 'var(--text-3)' }}
      >
        <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: 'var(--accent)' }} />
        Connecting…
      </div>
    );
  }

  if (connected && publicKey) {
    const keyStr = publicKey.toBase58();
    return (
      <div className="relative" ref={menuRef}>
        <button
          onClick={() => setMenuOpen((v) => !v)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
          style={{
            background: 'var(--accent-bg)',
            border: '1px solid var(--accent-bd)',
            color: 'var(--accent)',
          }}
          title={keyStr}
        >
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: 'var(--accent)', flexShrink: 0 }}
          />
          {wallet?.adapter.name ?? 'Wallet'} · {shortKey(keyStr)}
          <svg className="w-3 h-3 ml-0.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {menuOpen && (
          <div
            className="absolute right-0 top-9 w-52 rounded-xl shadow-lg z-50 py-1"
            style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}
          >
            <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--border)' }}>
              <div className="text-xs font-semibold mb-0.5" style={{ color: 'var(--text-1)' }}>
                {wallet?.adapter.name}
              </div>
              <div
                className="text-[11px] font-mono break-all"
                style={{ color: 'var(--text-3)' }}
              >
                {keyStr}
              </div>
            </div>
            <button
              onClick={() => {
                navigator.clipboard.writeText(keyStr).catch(() => undefined);
                setMenuOpen(false);
              }}
              className="w-full text-left px-3 py-2 text-xs transition-colors"
              style={{ color: 'var(--text-2)' }}
              onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-sub)')}
              onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
            >
              Copy address
            </button>
            <button
              onClick={() => { disconnect(); setMenuOpen(false); }}
              className="w-full text-left px-3 py-2 text-xs transition-colors"
              style={{ color: 'var(--red)' }}
              onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--red-bg)')}
              onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
            >
              Disconnect
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <button
      onClick={() => setVisible(true)}
      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors"
      style={{ background: 'var(--accent)', color: 'var(--bg-page)' }}
    >
      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z" />
      </svg>
      Connect Wallet
    </button>
  );
}
