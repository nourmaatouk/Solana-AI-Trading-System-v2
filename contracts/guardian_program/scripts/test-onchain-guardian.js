/**
 * Direct on-chain Guardian smoke test.
 *
 * Tests three cases against the live devnet program:
 *   1. PASS — valid trade (should create an ExecutionRecord PDA)
 *   2. BLOCK — confidence too low (should fail with LowConfidence 6002)
 *   3. BLOCK — amount too large (should fail with TradeSizeExceeded 6001)
 *   4. REPLAY — same signal key as case 1 (should fail with account-already-in-use)
 *
 * Run from repo root:
 *   GUARDIAN_PRIVATE_KEY=<base58> node contracts/guardian_program/scripts/test-onchain-guardian.js
 */
'use strict';

require('dotenv/config');
const { Connection, Keypair, PublicKey, Transaction, TransactionInstruction, SystemProgram, sendAndConfirmTransaction } = require('@solana/web3.js');
const bs58Module = require('bs58');
const bs58 = bs58Module.default ?? bs58Module;
const crypto = require('crypto');

const LAMPORTS_PER_SOL     = 1_000_000_000;
const BPS_DIVISOR          = 10_000;
const GUARDIAN_CONFIG_SEED = Buffer.from('guardian-config');
const EXECUTION_RECORD_SEED = Buffer.from('execution-record');
const AUTHORIZE_DISCRIMINATOR = Buffer.from([20, 171, 124, 206, 150, 135, 226, 97]);

const rpcUrl      = process.env.SOLANA_RPC_URL    ?? 'https://api.devnet.solana.com';
const privateKeyB58 = process.env.GUARDIAN_PRIVATE_KEY ?? process.env.SOLANA_PRIVATE_KEY;
const programIdStr  = process.env.GUARDIAN_PROGRAM_ID  ?? 'DVQ3uopMVLznw3tLJahTkfE1VwzNSPHo1Q8tTvUbjdrG';
const adminPubkeyStr = process.env.GUARDIAN_ADMIN_PUBKEY ?? process.env.SOLANA_PUBLIC_KEY;

if (!privateKeyB58) { console.error('Set GUARDIAN_PRIVATE_KEY'); process.exit(1); }
if (!adminPubkeyStr){ console.error('Set GUARDIAN_ADMIN_PUBKEY'); process.exit(1); }

const keypair   = Keypair.fromSecretKey(bs58.decode(privateKeyB58));
const programId = new PublicKey(programIdStr);
const adminKey  = new PublicKey(adminPubkeyStr);
const connection = new Connection(rpcUrl, 'confirmed');

function hashKey(key) {
  return crypto.createHash('sha256').update(key, 'utf8').digest();
}

function findConfigPda() {
  const [pda] = PublicKey.findProgramAddressSync(
    [GUARDIAN_CONFIG_SEED, adminKey.toBuffer()], programId,
  );
  return pda;
}

function findRecordPda(hash) {
  const [pda] = PublicKey.findProgramAddressSync(
    [EXECUTION_RECORD_SEED, hash], programId,
  );
  return pda;
}

async function callAuthorize(label, amountSol, confidenceRatio, signalKey) {
  const amountLamports = BigInt(Math.floor(amountSol * LAMPORTS_PER_SOL));
  const confidenceBps  = Math.round(confidenceRatio * BPS_DIVISOR);
  const hash           = hashKey(signalKey);
  const configPda      = findConfigPda();
  const recordPda      = findRecordPda(hash);

  const data = Buffer.allocUnsafe(50);
  AUTHORIZE_DISCRIMINATOR.copy(data, 0);
  data.writeBigUInt64LE(amountLamports, 8);
  data.writeUInt16LE(confidenceBps, 16);
  hash.copy(data, 18);

  const ix = new TransactionInstruction({
    programId,
    keys: [
      { pubkey: configPda,               isSigner: false, isWritable: false },
      { pubkey: recordPda,               isSigner: false, isWritable: true  },
      { pubkey: keypair.publicKey,       isSigner: true,  isWritable: true  },
      { pubkey: SystemProgram.programId, isSigner: false, isWritable: false },
    ],
    data,
  });

  const { blockhash, lastValidBlockHeight } = await connection.getLatestBlockhash();
  const tx = new Transaction({ blockhash, lastValidBlockHeight, feePayer: keypair.publicKey }).add(ix);

  try {
    const sig = await sendAndConfirmTransaction(connection, tx, [keypair], { commitment: 'confirmed' });
    console.log(`  ✓ APPROVED  ${label}`);
    console.log(`    → amount=${amountSol} SOL  confidence=${(confidenceRatio*100).toFixed(0)}%  bps=${confidenceBps}`);
    console.log(`    → ExecutionRecord PDA: ${recordPda.toBase58()}`);
    console.log(`    → Signature: ${sig}`);
    console.log(`    → Explorer: https://explorer.solana.com/tx/${sig}?cluster=devnet`);
    return { ok: true, sig };
  } catch (err) {
    const msg = err.message ?? String(err);
    const code = extractErrorCode(msg);
    const label2 = errorLabel(code);
    console.log(`  ✗ REJECTED  ${label}`);
    console.log(`    → amount=${amountSol} SOL  confidence=${(confidenceRatio*100).toFixed(0)}%  bps=${confidenceBps}`);
    console.log(`    → On-chain error: ${label2 ?? msg.slice(0, 80)}`);
    return { ok: false, code, msg };
  }
}

function extractErrorCode(msg) {
  const m = msg.match(/custom program error: 0x([0-9a-fA-F]+)/);
  if (m) return parseInt(m[1], 16);
  return null;
}

function errorLabel(code) {
  const map = {
    6000: 'EmergencyStop (6000)',
    6001: 'TradeSizeExceeded (6001)',
    6002: 'LowConfidence (6002)',
    6003: 'NotAdmin (6003)',
    6004: 'NotExecutor (6004)',
    6005: 'InvalidAmount (6005)',
    6006: 'InvalidBps (6006)',
  };
  return map[code] ?? null;
}

async function main() {
  console.log('\n' + '='.repeat(60));
  console.log('On-chain Guardian smoke test');
  console.log('='.repeat(60));
  console.log(`Program:  ${programIdStr}`);
  console.log(`Executor: ${keypair.publicKey.toBase58()}`);
  console.log(`Config:   ${findConfigPda().toBase58()}`);
  console.log('');

  const bal = await connection.getBalance(keypair.publicKey);
  console.log(`Executor balance: ${(bal / 1e9).toFixed(4)} SOL\n`);

  // ── Test 1: valid trade — should PASS ──────────────────────────────────────
  console.log('Test 1: valid trade (0.01 SOL, 75% confidence)');
  const t1key = `test-valid-${Date.now()}`;
  const r1 = await callAuthorize('valid trade', 0.01, 0.75, t1key);
  console.log('');

  // ── Test 2: confidence too low — should BLOCK with 6002 ───────────────────
  console.log('Test 2: confidence below threshold (0.01 SOL, 50% confidence — gate is 69%)');
  await callAuthorize('low confidence', 0.01, 0.50, `test-lowconf-${Date.now()}`);
  console.log('');

  // ── Test 3: amount too large — should BLOCK with 6001 ─────────────────────
  console.log('Test 3: trade size exceeds max (99 SOL — max is 0.1 SOL)');
  await callAuthorize('oversized trade', 99.0, 0.80, `test-bigsize-${Date.now()}`);
  console.log('');

  // ── Test 4: replay — same key as test 1 — should BLOCK ────────────────────
  if (r1.ok) {
    console.log('Test 4: replay attack (same signal key as test 1)');
    await callAuthorize('replay attack', 0.01, 0.75, t1key);
    console.log('');
  }

  console.log('='.repeat(60));
  console.log('Done. Verify approved transactions on Solana Explorer (devnet).');
  console.log('='.repeat(60) + '\n');
}

main().catch((err) => { console.error('\nFatal:', err.message ?? err); process.exit(1); });
