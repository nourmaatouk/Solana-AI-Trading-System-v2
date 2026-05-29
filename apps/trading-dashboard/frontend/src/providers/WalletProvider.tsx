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
import { ENABLE_PHANTOM_WALLET, SOLANA_RPC_URL, WALLET_MODE } from '../config';

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

interface Props {
  children: React.ReactNode;
}

/**
 * Wraps the React tree with the Solana wallet context.
 *
 * When WALLET_MODE !== 'phantom' or ENABLE_PHANTOM_WALLET is false, the
 * provider is still mounted but the wallet list is empty — components that call
 * useWallet() will always see connected=false, which is the correct state for
 * paper/system modes.
 */
export function WalletProvider({ children }: Props) {
  const wallets = useMemo(() => {
    if (!ENABLE_PHANTOM_WALLET || WALLET_MODE !== 'phantom') return [];
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
          {children}
        </WalletModalProvider>
      </SolanaWalletProvider>
    </ConnectionProvider>
  );
}
