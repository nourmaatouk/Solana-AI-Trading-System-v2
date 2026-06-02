/**
 * One-time setup script — initializes the GuardianConfig PDA on devnet.
 *
 * Run once after deployment:
 *   npx ts-node scripts/initialize-guardian.ts
 *
 * Required env (or .env file):
 *   SOLANA_PRIVATE_KEY   - base58 admin keypair (same as the executor wallet)
 *   GUARDIAN_PROGRAM_ID  - DVQ3uopMVLznw3tLJahTkfE1VwzNSPHo1Q8tTvUbjdrG
 *   SOLANA_RPC_URL       - https://api.devnet.solana.com (default)
 *
 * Optional overrides:
 *   MAX_TRADE_SOL              - default 0.1
 *   MAX_DAILY_LOSS_SOL         - default 0.5
 *   CONFIDENCE_THRESHOLD       - default 0.69
 *   ALLOWED_EXECUTOR_PUBKEY    - defaults to the admin's own pubkey
 */

import 'dotenv/config';
import crypto from 'crypto';

const LAMPORTS_PER_SOL      = 1_000_000_000;
const BPS_DIVISOR           = 10_000;
const GUARDIAN_CONFIG_SEED  = Buffer.from('guardian-config');
// Instruction discriminator for initialize_config from the IDL.
const INIT_DISCRIMINATOR    = Buffer.from([208, 127, 21, 1, 194, 190, 196, 70]);

async function main() {
  const web3   = require('@solana/web3.js');
  let   bs58   = require('bs58');
  if (!bs58.decode && bs58.default) bs58 = bs58.default;

  const rpcUrl          = process.env.SOLANA_RPC_URL    ?? 'https://api.devnet.solana.com';
  const privateKeyB58   = process.env.SOLANA_PRIVATE_KEY!;
  const programIdStr    = process.env.GUARDIAN_PROGRAM_ID!;

  if (!privateKeyB58)  { console.error('SOLANA_PRIVATE_KEY not set'); process.exit(1); }
  if (!programIdStr)   { console.error('GUARDIAN_PROGRAM_ID not set'); process.exit(1); }

  const maxTradeSol       = parseFloat(process.env.MAX_TRADE_SOL        ?? '0.1');
  const maxDailyLossSol   = parseFloat(process.env.MAX_DAILY_LOSS_SOL   ?? '0.5');
  const confidenceGate    = parseFloat(process.env.CONFIDENCE_THRESHOLD ?? '0.69');

  const keypair        = web3.Keypair.fromSecretKey(bs58.decode(privateKeyB58));
  const adminPubkey    = keypair.publicKey;
  const executorPubkey = process.env.ALLOWED_EXECUTOR_PUBKEY
    ? new web3.PublicKey(process.env.ALLOWED_EXECUTOR_PUBKEY)
    : adminPubkey;

  const programId = new web3.PublicKey(programIdStr);

  // Derive GuardianConfig PDA
  const [configPda] = web3.PublicKey.findProgramAddressSync(
    [GUARDIAN_CONFIG_SEED, adminPubkey.toBuffer()],
    programId,
  );

  console.log(`Program ID:       ${programIdStr}`);
  console.log(`Admin:            ${adminPubkey.toBase58()}`);
  console.log(`Executor:         ${executorPubkey.toBase58()}`);
  console.log(`GuardianConfig:   ${configPda.toBase58()}`);
  console.log(`Max trade:        ${maxTradeSol} SOL`);
  console.log(`Max daily loss:   ${maxDailyLossSol} SOL`);
  console.log(`Confidence gate:  ${Math.round(confidenceGate * BPS_DIVISOR)} bps (${(confidenceGate * 100).toFixed(1)}%)`);
  console.log('');

  const connection = new web3.Connection(rpcUrl, 'confirmed');

  // Check if already initialized
  const existing = await connection.getAccountInfo(configPda);
  if (existing) {
    console.log('GuardianConfig PDA already exists — already initialized.');
    console.log(`Add to your .env:\n  GUARDIAN_PROGRAM_ID=${programIdStr}\n  GUARDIAN_ADMIN_PUBKEY=${adminPubkey.toBase58()}\n  ENABLE_ONCHAIN_GUARDIAN=true`);
    return;
  }

  // Build instruction data:
  //   [0..8]   discriminator (8 bytes)
  //   [8..16]  max_trade_lamports u64 LE
  //   [16..24] max_daily_loss_lamports u64 LE
  //   [24..26] confidence_threshold_bps u16 LE
  //   [26..58] allowed_executor pubkey (32 bytes)
  const data = Buffer.allocUnsafe(8 + 8 + 8 + 2 + 32);
  INIT_DISCRIMINATOR.copy(data, 0);
  data.writeBigUInt64LE(BigInt(Math.floor(maxTradeSol     * LAMPORTS_PER_SOL)), 8);
  data.writeBigUInt64LE(BigInt(Math.floor(maxDailyLossSol * LAMPORTS_PER_SOL)), 16);
  data.writeUInt16LE(Math.round(confidenceGate * BPS_DIVISOR), 24);
  executorPubkey.toBuffer().copy(data, 26);

  const instruction = new web3.TransactionInstruction({
    programId,
    keys: [
      { pubkey: configPda,                 isSigner: false, isWritable: true  },
      { pubkey: adminPubkey,               isSigner: true,  isWritable: true  },
      { pubkey: web3.SystemProgram.programId, isSigner: false, isWritable: false },
    ],
    data,
  });

  const { blockhash, lastValidBlockHeight } = await connection.getLatestBlockhash();
  const tx = new web3.Transaction({ blockhash, lastValidBlockHeight, feePayer: adminPubkey })
    .add(instruction);

  console.log('Sending initialize_config transaction…');
  const sig = await web3.sendAndConfirmTransaction(connection, tx, [keypair], { commitment: 'confirmed' });

  console.log(`\n✓ GuardianConfig initialized! Signature: ${sig}`);
  console.log(`  View on explorer: https://explorer.solana.com/tx/${sig}?cluster=devnet`);
  console.log('');
  console.log('Add these to services/execution/.env:');
  console.log(`  GUARDIAN_PROGRAM_ID=${programIdStr}`);
  console.log(`  GUARDIAN_ADMIN_PUBKEY=${adminPubkey.toBase58()}`);
  console.log(`  ENABLE_ONCHAIN_GUARDIAN=true`);
}

main().catch((err) => { console.error(err); process.exit(1); });
