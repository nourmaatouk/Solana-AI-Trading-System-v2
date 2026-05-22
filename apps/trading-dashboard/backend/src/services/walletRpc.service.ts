import { Connection, PublicKey, LAMPORTS_PER_SOL } from '@solana/web3.js';

// Base58 chars only, 32–44 characters (32-byte Solana public key base58-encoded).
const SOLANA_PUBKEY_RE = /^[1-9A-HJ-NP-Za-km-z]{32,44}$/;

function getRpcUrl(): string {
  return process.env.SOLANA_RPC_URL ?? 'https://api.devnet.solana.com';
}

function getCluster(): string {
  return process.env.SOLANA_CLUSTER ?? 'devnet';
}

function getConnection(): Connection {
  return new Connection(getRpcUrl(), 'confirmed');
}

export function isValidPublicKey(pk: string): boolean {
  return SOLANA_PUBKEY_RE.test(pk.trim());
}

export async function getWalletBalance(publicKeyStr: string): Promise<{
  publicKey: string;
  balance_sol: number;
  cluster: string;
  rpc_url: string;
  timestamp: string;
}> {
  const pk = new PublicKey(publicKeyStr.trim());
  const conn = getConnection();
  const lamports = await conn.getBalance(pk);
  return {
    publicKey: publicKeyStr.trim(),
    balance_sol: lamports / LAMPORTS_PER_SOL,
    cluster: getCluster(),
    rpc_url: getRpcUrl(),
    timestamp: new Date().toISOString(),
  };
}

export async function requestDevnetAirdrop(
  publicKeyStr: string,
  amountSol: number,
): Promise<{ signature: string; amount_sol: number; cluster: string }> {
  const pk = new PublicKey(publicKeyStr.trim());
  const conn = getConnection();
  const lamports = Math.floor(amountSol * LAMPORTS_PER_SOL);
  const signature = await conn.requestAirdrop(pk, lamports);
  const latestBlockhash = await conn.getLatestBlockhash();
  await conn.confirmTransaction({ signature, ...latestBlockhash }, 'confirmed');
  return { signature, amount_sol: amountSol, cluster: getCluster() };
}
