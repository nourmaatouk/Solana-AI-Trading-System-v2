/**
 * useSystemWallet — simulates a connected wallet using the backend's
 * Devnet keypair (SOLANA_PRIVATE_KEY in services/execution/.env).
 *
 * This lets the platform enter Devnet mode without a real Phantom extension.
 * The private key never touches the frontend; the public key is fetched from
 * GET /api/execution/status (primary source) and stored in sessionStorage for
 * the session. SYSTEM_WALLET_PUBLIC_KEY (from VITE_SYSTEM_WALLET_PUBLIC_KEY) is
 * an optional offline fallback only — empty by default.
 */
import React, { createContext, useContext, useState, useCallback } from 'react';
import { fetchExecutionStatus } from '../services/executionApi';
import { SYSTEM_WALLET_PUBLIC_KEY } from '../config';

const SESSION_KEY = 'solvanta_system_wallet';

interface SystemWalletState {
  connected: boolean;
  publicKey: string | null;
  connecting: boolean;
  connect: () => Promise<void>;
  disconnect: () => void;
}

const SystemWalletContext = createContext<SystemWalletState>({
  connected: false,
  publicKey: null,
  connecting: false,
  connect: async () => {},
  disconnect: () => {},
});

export function SystemWalletProvider({ children }: { children: React.ReactNode }) {
  const [publicKey, setPublicKey] = useState<string | null>(
    () => sessionStorage.getItem(SESSION_KEY) ?? null,
  );
  const [connecting, setConnecting] = useState(false);

  const connect = useCallback(async () => {
    setConnecting(true);
    try {
      // Primary source: the backend reports the active system wallet pubkey.
      const status = await fetchExecutionStatus();
      const pk = status.wallet.public_key || SYSTEM_WALLET_PUBLIC_KEY;
      if (!pk) throw new Error('no_system_wallet');
      setPublicKey(pk);
      sessionStorage.setItem(SESSION_KEY, pk);
    } catch {
      // Backend unreachable — fall back to the configured env key if one is set.
      // When VITE_SYSTEM_WALLET_PUBLIC_KEY is empty, stay disconnected rather
      // than connecting with an unknown key.
      if (SYSTEM_WALLET_PUBLIC_KEY) {
        setPublicKey(SYSTEM_WALLET_PUBLIC_KEY);
        sessionStorage.setItem(SESSION_KEY, SYSTEM_WALLET_PUBLIC_KEY);
      }
    } finally {
      setConnecting(false);
    }
  }, []);

  const disconnect = useCallback(() => {
    setPublicKey(null);
    sessionStorage.removeItem(SESSION_KEY);
  }, []);

  return (
    <SystemWalletContext.Provider
      value={{ connected: publicKey !== null, publicKey, connecting, connect, disconnect }}
    >
      {children}
    </SystemWalletContext.Provider>
  );
}

export function useSystemWallet(): SystemWalletState {
  return useContext(SystemWalletContext);
}
