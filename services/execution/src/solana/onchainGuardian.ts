/**
 * On-chain Guardian client.
 *
 * Calls `authorize_execution` on the deployed Guardian Anchor program
 * (contracts/guardian_program) BEFORE the actual Solana transaction so the
 * swap can only land if the program approves it on-chain.
 *
 * No Anchor TypeScript SDK needed — we build the instruction manually from
 * the IDL discriminator and argument layout so the execution service stays
 * lean (@solana/web3.js is already a dependency).
 *
 * Rules enforced on-chain (mirrors the off-chain TypeScript Guardian):
 *   1. Emergency stop flag (stored in GuardianConfig PDA)
 *   2. Executor key must match the configured allowed_executor
 *   3. Amount ≤ max_trade_lamports
 *   4. Confidence ≥ confidence_threshold_bps
 *
 * Replay protection: every approved signal creates an ExecutionRecord PDA
 * seeded by the signal hash. A second call with the same hash reverts
 * automatically because the account already exists.
 */
import crypto from 'crypto';
import { log } from '../logger';

const LAMPORTS_PER_SOL = 1_000_000_000;
const BPS_DIVISOR = 10_000;

// Seed constants — must match the Rust program exactly.
const GUARDIAN_CONFIG_SEED   = Buffer.from('guardian-config');
const EXECUTION_RECORD_SEED  = Buffer.from('execution-record');

// Instruction discriminator from the IDL (authorize_execution).
// Anchor computes this as sha256("global:authorize_execution")[0..8].
const AUTHORIZE_DISCRIMINATOR = Buffer.from([20, 171, 124, 206, 150, 135, 226, 97]);

export interface OnChainGuardianParams {
  programId:        string;
  adminPubkey:      string;   // owner of the GuardianConfig PDA
  rpcUrl:           string;
  privateKeyBase58: string;
  amountSol:        number;
  confidenceRatio:  number;   // [0, 1]
  signalKey:        string;   // unique per signal — hashed to [u8;32] for PDA + replay guard
}

export interface OnChainGuardianResult {
  authorized: boolean;
  signature:  string | null;
  error:      string | null;
}

/** SHA-256 of the signal key → [u8; 32] used as the ExecutionRecord seed. */
function hashSignalKey(key: string): Buffer {
  return crypto.createHash('sha256').update(key, 'utf8').digest();
}

type PublicKeyClass = {
  new(s: string): { toBuffer(): Buffer };
  findProgramAddressSync(seeds: Buffer[], programId: unknown): [unknown, number];
};

/** Derive the GuardianConfig PDA (seeds: ["guardian-config", admin]). */
function guardianConfigPda(web3: Record<string, unknown>, programId: string, adminPubkey: string) {
  const PK = web3['PublicKey'] as PublicKeyClass;
  const findProgramAddressSync = PK.findProgramAddressSync.bind(PK);
  const [pda] = findProgramAddressSync(
    [GUARDIAN_CONFIG_SEED, new PK(adminPubkey).toBuffer()],
    new PK(programId),
  );
  return pda;
}

/** Derive the ExecutionRecord PDA (seeds: ["execution-record", signal_hash]). */
function executionRecordPda(web3: Record<string, unknown>, programId: string, hash: Buffer) {
  const PK = web3['PublicKey'] as PublicKeyClass;
  const findProgramAddressSync = PK.findProgramAddressSync.bind(PK);
  const [pda] = findProgramAddressSync(
    [EXECUTION_RECORD_SEED, hash],
    new PK(programId),
  );
  return pda;
}

/**
 * Submit an `authorize_execution` transaction to the Guardian program.
 *
 * Returns { authorized: true, signature } when the program approves.
 * Returns { authorized: false, error } when any on-chain rule fails — the
 * error message contains the program error code so callers can surface it.
 */
export async function authorizeOnChain(
  params: OnChainGuardianParams,
): Promise<OnChainGuardianResult> {
  const {
    programId, adminPubkey, rpcUrl, privateKeyBase58,
    amountSol, confidenceRatio, signalKey,
  } = params;

  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const web3 = require('@solana/web3.js') as Record<string, unknown>;
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    let bs58 = require('bs58') as { decode?: (s: string) => Uint8Array; default?: { decode(s: string): Uint8Array } };
    if (!bs58.decode && bs58.default) bs58 = bs58.default as typeof bs58;

    const Connection  = web3['Connection']  as new (url: string, c: string) => unknown;
    const PublicKey   = web3['PublicKey']   as new (s: string) => { toBuffer(): Buffer };
    const Transaction = web3['Transaction'] as new (opts: object) => { add(ix: unknown): unknown };
    const Keypair     = web3['Keypair']     as { fromSecretKey(sk: Uint8Array): { publicKey: unknown; secretKey: Uint8Array } };
    const SystemProg  = (web3['SystemProgram'] as Record<string, unknown>)['programId'];
    const sendAndConfirm = web3['sendAndConfirmTransaction'] as (
      conn: unknown, tx: unknown, signers: unknown[], opts: object,
    ) => Promise<string>;
    const getLatestBlockhash = (conn: unknown) =>
      (conn as { getLatestBlockhash(): Promise<{ blockhash: string; lastValidBlockHeight: number }> })
        .getLatestBlockhash();

    const connection = new Connection(rpcUrl, 'confirmed');
    const keypair    = Keypair.fromSecretKey(bs58.decode!(privateKeyBase58));

    const amountLamports = BigInt(Math.floor(amountSol * LAMPORTS_PER_SOL));
    const confidenceBps  = Math.round(Math.min(1, Math.max(0, confidenceRatio)) * BPS_DIVISOR);
    const hash           = hashSignalKey(signalKey);

    const configPda = guardianConfigPda(web3, programId, adminPubkey);
    const recordPda = executionRecordPda(web3, programId, hash);

    // Instruction data layout (50 bytes total):
    //   [0..8]  discriminator  (8 bytes)
    //   [8..16] amount_lamports u64 LE (8 bytes)
    //   [16..18] confidence_bps u16 LE (2 bytes)
    //   [18..50] signal_hash   [u8;32] (32 bytes)
    const data = Buffer.allocUnsafe(8 + 8 + 2 + 32);
    AUTHORIZE_DISCRIMINATOR.copy(data, 0);
    data.writeBigUInt64LE(amountLamports, 8);
    data.writeUInt16LE(confidenceBps, 16);
    hash.copy(data, 18);

    const TxInstruction = web3['TransactionInstruction'] as new (opts: {
      programId: unknown; keys: unknown[]; data: Buffer;
    }) => unknown;

    const instruction = new TxInstruction({
      programId: new PublicKey(programId),
      keys: [
        { pubkey: configPda,           isSigner: false, isWritable: false },
        { pubkey: recordPda,           isSigner: false, isWritable: true  },
        { pubkey: keypair.publicKey,   isSigner: true,  isWritable: true  },
        { pubkey: SystemProg,          isSigner: false, isWritable: false },
      ],
      data,
    });

    const { blockhash, lastValidBlockHeight } = await getLatestBlockhash(connection);
    const tx = new Transaction({ blockhash, lastValidBlockHeight, feePayer: keypair.publicKey })
      .add(instruction);

    const signature = await sendAndConfirm(connection, tx, [keypair], { commitment: 'confirmed' });

    log.info('onchain_guardian.authorized', {
      signature,
      amount_sol: amountSol,
      confidence_bps: confidenceBps,
      signal_key_prefix: signalKey.slice(0, 20),
    });
    return { authorized: true, signature, error: null };

  } catch (err) {
    const msg = (err as Error).message ?? String(err);
    // Surface human-readable on-chain error names where possible.
    const reason = parseGuardianError(msg);
    log.warn('onchain_guardian.rejected', { reason });
    return { authorized: false, signature: null, error: reason };
  }
}

/** Map on-chain error codes to human-readable names. */
function parseGuardianError(msg: string): string {
  const codeMap: Record<string, string> = {
    '6000': 'EMERGENCY_STOP — Guardian kill-switch is active',
    '6001': 'TRADE_SIZE_EXCEEDED — amount exceeds on-chain maximum',
    '6002': 'LOW_CONFIDENCE — signal confidence below on-chain threshold',
    '6003': 'NOT_ADMIN — caller is not the program admin',
    '6004': 'NOT_EXECUTOR — wallet key is not the configured executor',
    '6005': 'INVALID_AMOUNT — amount must be > 0',
    '6006': 'INVALID_BPS — confidence basis-points out of range',
  };
  for (const [code, label] of Object.entries(codeMap)) {
    if (msg.includes(code)) return label;
  }
  // Replay protection: if the ExecutionRecord PDA already exists Solana
  // returns an "already in use" error — surface it clearly.
  if (msg.toLowerCase().includes('already in use') || msg.includes('0x0')) {
    return 'REPLAY_BLOCKED — this signal was already executed on-chain';
  }
  return msg;
}
