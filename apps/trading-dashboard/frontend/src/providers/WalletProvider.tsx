import React, { useEffect, useMemo, useRef } from 'react';
import {
  ConnectionProvider as _ConnectionProvider,
  WalletProvider as _SolanaWalletProvider,
  useWallet,
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
// Double-cast bridges the React 18 / React 19 @types mismatch without changing
// React version or runtime behaviour.
type AnyProvider = React.ComponentType<{ children: React.ReactNode; [k: string]: unknown }>;
const ConnectionProvider   = _ConnectionProvider   as unknown as AnyProvider;
const SolanaWalletProvider = _SolanaWalletProvider as unknown as AnyProvider;
const WalletModalProvider  = _WalletModalProvider  as unknown as AnyProvider;

// Wallet adapter CSS — scoped to .wallet-adapter-* class names.
import '@solana/wallet-adapter-react-ui/styles.css';

/**
 * Calls connect() the moment the user selects a wallet from the modal.
 * The WalletModal only calls adapter.select() — with autoConnect=false
 * nothing else triggers the actual connection popup. This component fills
 * that gap globally so every "Connect Wallet" button in the app works
 * without each one needing its own connect() logic.
 *
 * Safe to call twice: connect() is a no-op when already connecting/connected.
 */
function ConnectOnSelect() {
  const { wallet, connected, connecting, connect } = useWallet();
  const prevWalletName = useRef<string | null>(null);

  useEffect(() => {
    const name = wallet?.adapter.name ?? null;
    if (name && name !== prevWalletName.current && !connected && !connecting) {
      connect().catch(() => {});
    }
    prevWalletName.current = name;
  }, [wallet, connected, connecting, connect]);

  return null;
}

interface Props {
  children: React.ReactNode;
}

/**
 * Wraps the React tree with the Solana wallet context.
 *
 * Adapters are registered whenever ENABLE_PHANTOM_WALLET is true, independent of
 * WALLET_MODE — so a user can connect a wallet and sign manually even while the
 * backend keypair (WALLET_MODE=system) auto-signs by default ("offer both").
 * Set ENABLE_PHANTOM_WALLET=false to mount an empty provider (no wallet UI).
 */
export function WalletProvider({ children }: Props) {
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
      <SolanaWalletProvider wallets={wallets} autoConnect={false}>
        <WalletModalProvider>
          <ConnectOnSelect />
          {children}
        </WalletModalProvider>
      </SolanaWalletProvider>
    </ConnectionProvider>
  );
}
