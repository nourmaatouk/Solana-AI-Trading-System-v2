import { useCallback, useEffect, useRef, useState } from 'react';
import { useWallet, useConnection } from '@solana/wallet-adapter-react';
import { VersionedTransaction } from '@solana/web3.js';
import { useTradingStore } from '../store';
import type { AISignalData } from '../types';
import { loadSettings } from './useSettings';
import {
  validateExecution,
  prepareTrade,
  type GuardianResultDto,
  type ExecutionResultDto,
} from '../services/executionApi';
import { WALLET_MODE } from '../config';

function computeSignalKey(s: AISignalData): string {
  return [
    s.action ?? '',
    s.generated_at_utc ?? s.signal_timestamp ?? '',
    String(s.fused_score ?? ''),
    String(s.confidence ?? ''),
  ].join('|');
}

export interface AutoExecutionState {
  /** True only after the user has explicitly armed auto-execution this session.
   *  Phantom auto-signing never fires while armed=false, so simply opening the
   *  dashboard cannot trigger a wallet popup. */
  armed: boolean;
  lastProcessedKey: string | null;
  lastValidatedAt: string | null;
  lastAutoExecutedAt: string | null;
  /** Always reflects the most recent signal received: cleared whenever a new
   *  signal arrives so the UI never shows a stale reason from an earlier one. */
  skippedReason: string | null;
  validationResult: GuardianResultDto | null;
  autoExecResult: ExecutionResultDto | null;
  executing: boolean;
}

export interface AutoExecutionControls extends AutoExecutionState {
  arm: () => void;
  disarm: () => void;
}

const INITIAL_STATE: AutoExecutionState = {
  armed: false,
  lastProcessedKey: null,
  lastValidatedAt: null,
  lastAutoExecutedAt: null,
  skippedReason: null,
  validationResult: null,
  autoExecResult: null,
  executing: false,
};

/**
 * Watches incoming AI signals and acts based on the user's trading mode:
 *
 * Manual    → no-op (always skipped).
 * Semi-Auto → auto-validates each new BUY/SELL signal; stores the Guardian
 *             verdict so ExecutionPanel can display it and let the user confirm.
 * Auto      → phantom+non-paper: only fires when armed=true; prepares the
 *             unsigned tx and requests Phantom signing.
 *             paper/system: backend AutoExecutor handles it; frontend skips.
 *
 * Safety properties:
 *   1. Phantom auto-signing requires an explicit user arm() this session,
 *      so opening the page cannot trigger a wallet popup.
 *   2. Each unique signal key is processed at most once.
 *   3. State updates after the component unmounts are suppressed.
 *   4. skippedReason is cleared on every fresh signal — never leaks across.
 */
export function useAutoExecution(): AutoExecutionControls {
  const signal = useTradingStore((s) => s.signal);
  const { publicKey, connected, sendTransaction } = useWallet();
  const { connection } = useConnection();

  const lastProcessedKeyRef = useRef<string | null>(null);
  const executingRef = useRef(false);
  const armedRef = useRef(false);
  const isMountedRef = useRef(true);
  const walletRef = useRef({ publicKey, connected, sendTransaction });
  walletRef.current = { publicKey, connected, sendTransaction };

  const [state, setState] = useState<AutoExecutionState>(INITIAL_STATE);

  // Suppress state updates after unmount (e.g. user navigates away during
  // prepareTrade/sendTransaction).
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const safeSetState = useCallback(
    (updater: (prev: AutoExecutionState) => AutoExecutionState) => {
      if (!isMountedRef.current) return;
      setState(updater);
    },
    [],
  );

  const arm = useCallback(() => {
    armedRef.current = true;
    safeSetState((prev) => ({ ...prev, armed: true }));
  }, [safeSetState]);

  const disarm = useCallback(() => {
    armedRef.current = false;
    safeSetState((prev) => ({ ...prev, armed: false }));
  }, [safeSetState]);

  // Auto-disarm whenever the wallet disconnects — re-arming should be a
  // deliberate act after reconnect.
  useEffect(() => {
    if (!connected && armedRef.current) {
      armedRef.current = false;
      safeSetState((prev) => ({ ...prev, armed: false }));
    }
  }, [connected, safeSetState]);

  useEffect(() => {
    if (!signal) return;

    const settings = loadSettings();
    const { tradingMode, riskLevel, maxTradeSizeSol } = settings;

    // Always clear the previous signal's skip reason — skippedReason should
    // refer to the current signal only.
    safeSetState((prev) => ({ ...prev, skippedReason: null }));

    if (tradingMode === 'Manual') {
      safeSetState((prev) => ({ ...prev, skippedReason: 'Manual mode — auto-execution disabled' }));
      return;
    }

    const action = (signal.action ?? '').toUpperCase();
    if (action !== 'BUY' && action !== 'SELL') {
      safeSetState((prev) => ({
        ...prev,
        skippedReason: `Signal action "${action || 'HOLD'}" — skipped`,
      }));
      return;
    }

    const key = computeSignalKey(signal);
    if (key === lastProcessedKeyRef.current) return;

    if (executingRef.current) {
      safeSetState((prev) => ({
        ...prev,
        skippedReason: 'Previous auto-execution still in progress',
      }));
      return;
    }

    lastProcessedKeyRef.current = key;
    executingRef.current = true;
    safeSetState((prev) => ({
      ...prev,
      lastProcessedKey: key,
      executing: true,
      skippedReason: null,
    }));

    const { publicKey: pk, connected: isConnected, sendTransaction: signTx } = walletRef.current;
    const phantomKey =
      WALLET_MODE === 'phantom' && isConnected ? (pk?.toBase58() ?? null) : null;
    const baseReq = {
      amount_sol: maxTradeSizeSol,
      user_risk_level: riskLevel,
      phantom_public_key: phantomKey ?? undefined,
    };

    if (tradingMode === 'Semi-Auto') {
      validateExecution(baseReq)
        .then((r) => {
          safeSetState((prev) => ({
            ...prev,
            validationResult: r.guardian,
            lastValidatedAt: new Date().toISOString(),
            executing: false,
          }));
        })
        .catch((err: Error) => {
          // Release the key so a retry on the same signal is allowed after a
          // network/API error (as opposed to a Guardian block, which is final).
          lastProcessedKeyRef.current = null;
          safeSetState((prev) => ({
            ...prev,
            skippedReason: `Validation error: ${err.message}`,
            executing: false,
          }));
        })
        .finally(() => {
          executingRef.current = false;
        });
      return;
    }

    // Auto mode — Phantom must be explicitly armed; paper/system handled by backend.
    if (WALLET_MODE === 'phantom' && isConnected) {
      if (!armedRef.current) {
        safeSetState((prev) => ({
          ...prev,
          skippedReason: 'Auto-execution is not armed — click "Arm" to enable Phantom auto-signing',
          executing: false,
        }));
        executingRef.current = false;
        return;
      }

      prepareTrade(baseReq)
        .then(async (prepared) => {
          if (!prepared.guardian.approved || !prepared.transaction_base64) {
            safeSetState((prev) => ({
              ...prev,
              validationResult: prepared.guardian,
              lastValidatedAt: new Date().toISOString(),
              skippedReason: `Guardian blocked: ${prepared.guardian.blocked_by ?? 'unknown'}`,
              executing: false,
            }));
            return;
          }
          const txBuffer = Buffer.from(prepared.transaction_base64, 'base64');
          const transaction = VersionedTransaction.deserialize(new Uint8Array(txBuffer));
          const sig = await signTx(transaction, connection);
          const { blockhash, lastValidBlockHeight } = await connection.getLatestBlockhash();
          await connection.confirmTransaction({ signature: sig, blockhash, lastValidBlockHeight });
          const now = new Date().toISOString();
          // SAFETY: auto-disarm after a successful trade. Re-arming requires a
          // deliberate user click — this caps uncontrolled repeated signing.
          armedRef.current = false;
          safeSetState((prev) => ({
            ...prev,
            armed: false,
            validationResult: prepared.guardian,
            autoExecResult: {
              status: 'confirmed',
              mode: prepared.execution_mode,
              signature: sig,
              block_height: null,
              asset: prepared.asset,
              side: prepared.side,
              amount_sol: prepared.amount_sol,
              rpc_url: null,
              submitted_at_utc: now,
              confirmed_at_utc: now,
              error: null,
            },
            lastAutoExecutedAt: now,
            lastValidatedAt: now,
            skippedReason: 'Auto-disarmed after successful trade — click Arm to enable the next one',
            executing: false,
          }));
        })
        .catch((err: Error) => {
          // Release the key so the next signal poll can retry after a transient
          // network error. A Guardian BLOCK is a final verdict and stays consumed.
          lastProcessedKeyRef.current = null;
          safeSetState((prev) => ({
            ...prev,
            skippedReason: `Auto-exec error: ${err.message}`,
            executing: false,
          }));
        })
        .finally(() => {
          executingRef.current = false;
        });
    } else {
      safeSetState((prev) => ({
        ...prev,
        skippedReason: 'Auto mode (paper/system) — handled by backend AutoExecutor',
        executing: false,
      }));
      executingRef.current = false;
    }
  }, [signal, safeSetState, connection]);

  return { ...state, arm, disarm };
}
