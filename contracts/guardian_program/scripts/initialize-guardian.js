/**
 * One-time setup — initializes the GuardianConfig PDA on devnet.
 *
 * Run from the repo root:
 *   node contracts/guardian_program/scripts/initialize-guardian.js
 *
 * Required env:
 *   SOLANA_PRIVATE_KEY   - base58 admin keypair
 *   GUARDIAN_PROGRAM_ID  - deployed program ID
 *
 * Optional:
 *   SOLANA_RPC_URL            - default https://api.devnet.solana.com
 *   MAX_TRADE_SOL             - default 0.1
 *   MAX_DAILY_LOSS_SOL        - default 0.5
 *   CONFIDENCE_THRESHOLD      - default 0.69
 *   ALLOWED_EXECUTOR_PUBKEY   - defaults to the admin's own pubkey
 */

'use strict';

require('dotenv/config');

const { Connection, Keypair, PublicKey, Transaction, TransactionInstruction, SystemProgram, sendAndConfirmTransaction } = require('@solana/web3.js');
const bs58Module = require('bs58');
const bs58 = bs58Module.default ?? bs58Module;
const crypto = require('crypto');

const LAMPORTS_PER_SOL     = 1_000_000_000n;
const BPS_DIVISOR          = 10_000;
const GUARDIAN_CONFIG_SEED = Buffer.from('guardian-config');
// Discriminator for initialize_config from the IDL.
const INIT_DISCRIMINATOR   = Buffer.from([208, 127, 21, 1, 194, 190, 196, 70]);

async function main() {
  const rpcUrl         = process.env.SOLANA_RPC_URL    ?? 'https://api.devnet.solana.com';
  const privateKeyB58  = process.env.SOLANA_PRIVATE_KEY;
  const programIdStr   = process.env.GUARDIAN_PROGRAM_ID;

  if (!privateKeyB58) { console.error('Error: SOLANA_PRIVATE_KEY is not set.'); process.exit(1); }
  if (!programIdStr)  { console.error('Error: GUARDIAN_PROGRAM_ID is not set.'); process.exit(1); }

  const maxTradeSol      = parseFloat(process.env.MAX_TRADE_SOL        ?? '0.1');
  const maxDailyLossSol  = parseFloat(process.env.MAX_DAILY_LOSS_SOL   ?? '0.5');
  const confidenceGate   = parseFloat(process.env.CONFIDENCE_THRESHOLD ?? '0.69');

  const keypair       = Keypair.fromSecretKey(bs58.decode(privateKeyB58));
  const adminPubkey   = keypair.publicKey;
  const executorPubkey = process.env.ALLOWED_EXECUTOR_PUBKEY
    ? new PublicKey(process.env.ALLOWED_EXECUTOR_PUBKEY)
    : adminPubkey;
  const programId = new PublicKey(programIdStr);

  // Derive GuardianConfig PDA: seeds = ["guardian-config", admin]
  const [configPda] = PublicKey.findProgramAddressSync(
    [GUARDIAN_CONFIG_SEED, adminPubkey.toBuffer()],
    programId,
  );

  console.log('='.repeat(60));
  console.log('Guardian On-chain Setup');
  console.log('='.repeat(60));
  console.log(`RPC:              ${rpcUrl}`);
  console.log(`Program ID:       ${programIdStr}`);
  console.log(`Admin:            ${adminPubkey.toBase58()}`);
  console.log(`Executor:         ${executorPubkey.toBase58()}`);
  console.log(`GuardianConfig:   ${configPda.toBase58()}`);
  console.log(`Max trade:        ${maxTradeSol} SOL`);
  console.log(`Max daily loss:   ${maxDailyLossSol} SOL`);
  console.log(`Confidence gate:  ${Math.round(confidenceGate * BPS_DIVISOR)} bps (${(confidenceGate * 100).toFixed(1)}%)`);
  console.log('');

  const connection = new Connection(rpcUrl, 'confirmed');

  // Check admin balance
  const balance = await connection.getBalance(adminPubkey);
  console.log(`Admin balance:    ${(balance / 1e9).toFixed(4)} SOL`);
  if (balance < 0.01 * 1e9) {
    console.error('Error: insufficient balance. Fund the admin wallet first.');
    process.exit(1);
  }

  // Check if already initialized
  const existing = await connection.getAccountInfo(configPda);
  if (existing) {
    console.log('\nGuardianConfig PDA already exists — already initialized.\n');
    printEnvInstructions(programIdStr, adminPubkey.toBase58());
    return;
  }

  // Build instruction data (58 bytes):
  //   [0..8]   discriminator        (8 bytes)
  //   [8..16]  max_trade_lamports   u64 LE  (8 bytes)
  //   [16..24] max_daily_loss_lamports u64 LE (8 bytes)
  //   [24..26] confidence_threshold_bps u16 LE (2 bytes)
  //   [26..58] allowed_executor     pubkey (32 bytes)
  const data = Buffer.allocUnsafe(8 + 8 + 8 + 2 + 32);
  INIT_DISCRIMINATOR.copy(data, 0);
  data.writeBigUInt64LE(BigInt(Math.floor(maxTradeSol     * 1e9)), 8);
  data.writeBigUInt64LE(BigInt(Math.floor(maxDailyLossSol * 1e9)), 16);
  data.writeUInt16LE(Math.round(confidenceGate * BPS_DIVISOR), 24);
  executorPubkey.toBuffer().copy(data, 26);

  const instruction = new TransactionInstruction({
    programId,
    keys: [
      { pubkey: configPda,               isSigner: false, isWritable: true  },
      { pubkey: adminPubkey,             isSigner: true,  isWritable: true  },
      { pubkey: SystemProgram.programId, isSigner: false, isWritable: false },
    ],
    data,
  });

  const { blockhash, lastValidBlockHeight } = await connection.getLatestBlockhash();
  const tx = new Transaction({ blockhash, lastValidBlockHeight, feePayer: adminPubkey }).add(instruction);

  console.log('Sending initialize_config transaction…');
  const sig = await sendAndConfirmTransaction(connection, tx, [keypair], { commitment: 'confirmed' });

  console.log(`\n✓ GuardianConfig initialized!`);
  console.log(`  Signature: ${sig}`);
  console.log(`  Explorer:  https://explorer.solana.com/tx/${sig}?cluster=devnet`);
  console.log('');
  printEnvInstructions(programIdStr, adminPubkey.toBase58());
}

function printEnvInstructions(programId, adminPubkey) {
  console.log('Add these to services/execution/.env:');
  console.log('─'.repeat(50));
  console.log(`GUARDIAN_PROGRAM_ID=${programId}`);
  console.log(`GUARDIAN_ADMIN_PUBKEY=${adminPubkey}`);
  console.log(`ENABLE_ONCHAIN_GUARDIAN=true`);
  console.log('─'.repeat(50));
}

main().catch((err) => { console.error('\nError:', err.message ?? err); process.exit(1); });
