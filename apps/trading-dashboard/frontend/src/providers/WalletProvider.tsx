import React, { useMemo } from 'react';
import {
  ConnectionProvider as _ConnectionProvider,
  WalletProvider as _SolanaWalletProvider,
} from '@solana/wallet-adapter-react';
import { WalletModalProvider as _WalletModalProvider } from '@solana/wallet-adapter-react-ui';
import {
  PhantomWalletAdapter,
  SolflareWalletAdapter,
  CoinbaseWalletAdapter,
  LedgerWalletAdapter,
} from '@solana/wallet-adapter-wallets';
import { ENABLE_PHANTOM_WALLET, SOLANA_RPC_URL } from '../config';

// The wallet adapter packages are compiled against @types/react@19 (which
// includes `bigint` and `Promise<ReactNode>` in ReactNode / FunctionComponent).
// This project targets React 18. The cast bridges the type gap at the boundary
// without changing React or runtime behaviour.
type AnyProvider = React.ComponentType<{ children: React.ReactNode; [k: string]: unknown }>;
const ConnectionProvider   = _ConnectionProvider   as unknown as AnyProvider;
const SolanaWalletProvider = _SolanaWalletProvider as unknown as AnyProvider;
const WalletModalProvider  = _WalletModalProvider  as unknown as AnyProvider;

// Wallet adapter CSS — scoped to .wallet-adapter-* class names.
import '@solana/wallet-adapter-react-ui/styles.css';

// Start disconnected on every app open. The wallet-adapter persists the last
// selected wallet under 'walletName' and `autoConnect` would silently reconnect
// it on reload. Clearing it at module load (before the provider mounts) means
// reopening the app always requires an explicit "Connect Wallet" — selecting a
// wallet in-session still connects via autoConnect; only reconnect-on-reload is
// suppressed.
try {
  window.localStorage.removeItem('walletName');
} catch {
  /* no localStorage (SSR / privacy mode) — nothing to clear */
}

interface Props {
  children: React.ReactNode;
}

/**
 * Wraps the React tree with the Solana wallet context.
 *
 * `autoConnect` is enabled so that selecting a wallet in the modal triggers the
 * connection popup directly, and a previously-authorised wallet reconnects on
 * reload. This replaces an earlier custom "connect on select" component whose
 * name-change guard silently swallowed retries after a rejected attempt or a
 * disconnect (the user could not reconnect once the first attempt failed).
 *
 * Adapters are registered whenever ENABLE_PHANTOM_WALLET is true, independent of
 * WALLET_MODE — so a user can connect a wallet and sign manually even while the
 * backend keypair (WALLET_MODE=system) auto-signs by default ("offer both").
 * Modern wallets (Phantom, Solflare, Coinbase) are also auto-detected via the
 * Wallet Standard, so they work even if their explicit adapter is unavailable.
 * Set ENABLE_PHANTOM_WALLET=false to mount an empty provider (no wallet UI).
 */
export function WalletProvider({ children }: Props) {
  // Register the wallet adapters explicitly so they always appear in the modal.
  // Phantom/Solflare/Coinbase also register as Standard Wallets, which logs a
  // harmless "registered as a Standard Wallet" console notice — that message is
  // advisory only and does not affect connection. Keeping the explicit adapters
  // guarantees Phantom is listed even when standard auto-detection doesn't
  // surface it in the modal.
  const wallets = useMemo(() => {
    if (!ENABLE_PHANTOM_WALLET) return [];
    return [
      new PhantomWalletAdapter(),
      new SolflareWalletAdapter(),
      new CoinbaseWalletAdapter(),
      new LedgerWalletAdapter(),
    ];
  }, []);

  return (
    <ConnectionProvider endpoint={SOLANA_RPC_URL}>
      {/* onError prevents wallet connection failures from becoming unhandled
          exceptions. Common causes: extension not installed, user rejected the
          connection prompt, or extension locked. All are handled in the UI. */}
      <SolanaWalletProvider
        wallets={wallets}
        autoConnect
        onError={(err: { name: string; message: string }) => console.warn('[WalletAdapter]', err.name, err.message)}
      >
        <WalletModalProvider>
          {children}
        </WalletModalProvider>
      </SolanaWalletProvider>
    </ConnectionProvider>
  );
}
