import { Router } from 'express';
import {
  getWalletBalance,
  isValidPublicKey,
  requestDevnetAirdrop,
} from '../services/walletRpc.service';
import {
  SOLANA_CLUSTER,
  DEVNET_AIRDROP_AMOUNT_SOL,
  MAX_DEVNET_AIRDROP_SOL,
} from '../config';

export const walletRouter = Router();

// POST /api/wallet/session
// Validates a public key and records which wallet is making requests.
// Never requests or stores a private key.
walletRouter.post('/session', (req, res) => {
  const { publicKey } = req.body as { publicKey?: unknown };
  if (typeof publicKey !== 'string' || !publicKey.trim()) {
    return res.status(400).json({ error: 'publicKey is required' });
  }
  if (!isValidPublicKey(publicKey)) {
    return res.status(400).json({ error: 'Invalid Solana public key format' });
  }
  return res.json({
    publicKey: publicKey.trim(),
    cluster: SOLANA_CLUSTER,
    timestamp: new Date().toISOString(),
  });
});

// GET /api/wallet/balance/:publicKey
// Returns the SOL balance for the given public key via the configured RPC.
walletRouter.get('/balance/:publicKey', async (req, res) => {
  const { publicKey } = req.params;
  if (!isValidPublicKey(publicKey)) {
    return res.status(400).json({ error: 'Invalid Solana public key format' });
  }
  try {
    const result = await getWalletBalance(publicKey);
    return res.json(result);
  } catch (err) {
    return res.status(502).json({ error: (err as Error).message });
  }
});

// POST /api/wallet/airdrop-devnet
// Requests a devnet SOL airdrop. Blocked on mainnet and testnet.
walletRouter.post('/airdrop-devnet', async (req, res) => {
  if (SOLANA_CLUSTER !== 'devnet') {
    return res.status(403).json({
      error: `Airdrop is only available on devnet (current cluster: ${SOLANA_CLUSTER}).`,
    });
  }

  const { publicKey, amount_sol } = req.body as {
    publicKey?: unknown;
    amount_sol?: unknown;
  };

  if (typeof publicKey !== 'string' || !publicKey.trim()) {
    return res.status(400).json({ error: 'publicKey is required' });
  }
  if (!isValidPublicKey(publicKey)) {
    return res.status(400).json({ error: 'Invalid Solana public key format' });
  }

  const requested =
    typeof amount_sol === 'number' && Number.isFinite(amount_sol) && amount_sol > 0
      ? amount_sol
      : DEVNET_AIRDROP_AMOUNT_SOL;
  const clamped = Math.min(requested, MAX_DEVNET_AIRDROP_SOL);

  try {
    const result = await requestDevnetAirdrop(publicKey.trim(), clamped);
    return res.json(result);
  } catch (err) {
    return res.status(502).json({ error: (err as Error).message });
  }
});
