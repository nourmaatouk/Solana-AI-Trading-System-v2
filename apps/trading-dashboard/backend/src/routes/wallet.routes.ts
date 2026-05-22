import { Router } from 'express';
import {
  getWalletBalance,
  isValidPublicKey,
  requestDevnetAirdrop,
} from '../services/walletRpc.service';

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
    cluster: process.env.SOLANA_CLUSTER ?? 'devnet',
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
  const cluster = process.env.SOLANA_CLUSTER ?? 'devnet';
  if (cluster !== 'devnet') {
    return res.status(403).json({
      error: `Airdrop is only available on devnet (current cluster: ${cluster}).`,
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

  const defaultAmount = parseFloat(process.env.DEVNET_AIRDROP_AMOUNT_SOL ?? '1');
  const maxAmount = parseFloat(process.env.MAX_DEVNET_AIRDROP_SOL ?? '2');
  const requested =
    typeof amount_sol === 'number' && Number.isFinite(amount_sol) && amount_sol > 0
      ? amount_sol
      : defaultAmount;
  const clamped = Math.min(requested, maxAmount);

  try {
    const result = await requestDevnetAirdrop(publicKey.trim(), clamped);
    return res.json(result);
  } catch (err) {
    return res.status(502).json({ error: (err as Error).message });
  }
});
