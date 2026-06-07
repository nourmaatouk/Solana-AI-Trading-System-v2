import { useState, useRef, useEffect } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { useSystemWallet } from '../hooks/useSystemWallet';

function shortKey(key: string): string {
  return `${key.slice(0, 4)}…${key.slice(-4)}`;
}

/** Self-contained dropdown menu — owns its own ref and outside-click handler. */
function WalletMenu({
  label,
  fullKey,
  onDisconnect,
  onClose,
}: {
  label: string;
  fullKey: string;
  onDisconnect: () => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    }
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  return (
    <div
      ref={ref}
      className="absolute right-0 top-9 w-56 rounded-xl shadow-lg z-50 py-1"
      style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}
    >
      <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--border)' }}>
        <div className="text-xs font-semibold mb-0.5" style={{ color: 'var(--text-1)' }}>
          {label}
        </div>
        <div className="text-[11px] font-mono break-all" style={{ color: 'var(--text-3)' }}>
          {fullKey}
        </div>
      </div>
      <button
        onClick={() => { navigator.clipboard.writeText(fullKey).catch(() => undefined); onClose(); }}
        className="w-full text-left px-3 py-2 text-xs transition-colors"
        style={{ color: 'var(--text-2)' }}
        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-sub)')}
        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
      >
        Copy address
      </button>
      <button
        onClick={() => { onDisconnect(); onClose(); }}
        className="w-full text-left px-3 py-2 text-xs font-semibold transition-colors"
        style={{ color: 'var(--red)' }}
        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--red-bg)')}
        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
      >
        Disconnect
      </button>
    </div>
  );
}

/**
 * Wallet connection button that supports two connection paths:
 *
 *   • Phantom  — real browser wallet (WALLET_MODE=phantom)
 *   • Dev Wallet — backend system keypair, no browser extension required
 *
 * When connected, shows a chip with the shortened public key and a dropdown
 * containing Copy Address and Disconnect.
 */
export function WalletConnectionButton() {
  const { publicKey, connected: phantomConnected, disconnect: phantomDisconnect,
          wallet, connecting: phantomConnecting } = useWallet();
  const {
    connected: systemConnected, publicKey: systemKey,
    connecting: systemConnecting, disconnect: systemDisconnect,
  } = useSystemWallet();

  const [menuOpen, setMenuOpen] = useState(false);

  // ── Phantom connected ───────────────────────────────────────────────────────
  if (phantomConnected && publicKey) {
    const keyStr = publicKey.toBase58();
    return (
      <div className="relative">
        <button
          onClick={() => setMenuOpen((v) => !v)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
          style={{ background: 'var(--accent-bg)', border: '1px solid var(--accent-bd)', color: 'var(--accent)' }}
          title={keyStr}
        >
          <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: 'var(--accent)' }} />
          {wallet?.adapter.name ?? 'Phantom'} · {shortKey(keyStr)}
          <svg className="w-3 h-3 ml-0.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {menuOpen && (
          <WalletMenu
            label={wallet?.adapter.name ?? 'Phantom Wallet'}
            fullKey={keyStr}
            onDisconnect={phantomDisconnect}
            onClose={() => setMenuOpen(false)}
          />
        )}
      </div>
    );
  }

  // ── System (Dev) wallet connected ───────────────────────────────────────────
  if (systemConnected && systemKey) {
    return (
      <div className="relative">
        <button
          onClick={() => setMenuOpen((v) => !v)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
          style={{ background: 'rgba(139,92,246,0.12)', border: '1px solid rgba(139,92,246,0.35)', color: '#a78bfa' }}
          title={systemKey}
        >
          <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: '#a78bfa' }} />
          Dev Wallet · {shortKey(systemKey)}
          <svg className="w-3 h-3 ml-0.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {menuOpen && (
          <WalletMenu
            label="Devnet System Wallet"
            fullKey={systemKey}
            onDisconnect={systemDisconnect}
            onClose={() => setMenuOpen(false)}
          />
        )}
      </div>
    );
  }

  // ── Connecting spinner ──────────────────────────────────────────────────────
  if (phantomConnecting || systemConnecting) {
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

  // ── Disconnected ────────────────────────────────────────────────────────────
  // No connect button here. Wallet connection is initiated only from the landing
  // page chooser ("Connect Wallet & Trade"); the TopBar just reflects the
  // connected state and offers Disconnect.
  return null;
}
