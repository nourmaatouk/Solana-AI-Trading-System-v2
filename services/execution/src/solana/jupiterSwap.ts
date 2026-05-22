/**
 * Jupiter v6 REST client — pure HTTP, no SDK dependency.
 *
 * Flow:
 *   1. fetchJupiterQuote  — GET /quote  →  best route for the trade
 *   2. fetchJupiterSwapTx — POST /swap  →  base64-encoded VersionedTransaction
 *
 * Trade direction:
 *   SELL  → ExactIn:  spend exactly `amount_sol` SOL, receive quote currency
 *   BUY   → ExactOut: receive exactly `amount_sol` SOL, spend quote currency
 *
 * Important constraint:
 *   Jupiter's quote/swap API is mainnet-only. Submitting the returned
 *   transaction to a devnet RPC will fail because the Jupiter program
 *   accounts do not exist on devnet. Use SOLANA_RPC_URL pointing at a
 *   mainnet endpoint when JUPITER_SWAP_ENABLED=true.
 */

const WRAPPED_SOL_MINT = 'So11111111111111111111111111111111111111112';

// ── Jupiter API shapes ───────────────────────────────────────────────────────

export interface JupiterQuoteResponse {
  inputMint: string;
  inAmount: string;
  outputMint: string;
  outAmount: string;
  otherAmountThreshold: string;
  swapMode: 'ExactIn' | 'ExactOut';
  slippageBps: number;
  priceImpactPct: string;
  routePlan: unknown[];
  [key: string]: unknown;
}

interface JupiterSwapApiResponse {
  swapTransaction: string;
  lastValidBlockHeight: number;
  prioritizationFeeLamports: number;
}

export interface JupiterSwapPackage {
  quoteResponse: JupiterQuoteResponse;
  /** base64-encoded VersionedTransaction ready to be deserialized and signed */
  swapTransaction: string;
  inAmountLamports: bigint;
  outAmountLamports: bigint;
  priceImpactPct: string;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

async function jupiterGet<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text().catch(() => '(unreadable)');
    throw new Error(`Jupiter GET ${res.status} — ${body.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

async function jupiterPost<T>(url: string, payload: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '(unreadable)');
    throw new Error(`Jupiter POST ${res.status} — ${body.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Build and return a Jupiter swap package for the given trade.
 *
 * @param apiUrl           Jupiter quote API base URL (no trailing slash)
 * @param side             BUY = receive SOL (ExactOut), SELL = spend SOL (ExactIn)
 * @param amountSol        SOL quantity to buy or sell
 * @param userPublicKey    Signer's base58 public key (fee payer + authority)
 * @param quoteCurrencyMint  Mint of the counter-currency (USDC or USDT)
 * @param slippageBps      Slippage tolerance in basis points (e.g. 50 = 0.5 %)
 */
export async function fetchJupiterSwap(
  apiUrl: string,
  side: 'BUY' | 'SELL',
  amountSol: number,
  userPublicKey: string,
  quoteCurrencyMint: string,
  slippageBps: number,
): Promise<JupiterSwapPackage> {
  const lamports = Math.floor(amountSol * 1_000_000_000);
  if (lamports <= 0) throw new Error(`Invalid amountSol: ${amountSol}`);

  const inputMint  = side === 'SELL' ? WRAPPED_SOL_MINT : quoteCurrencyMint;
  const outputMint = side === 'SELL' ? quoteCurrencyMint : WRAPPED_SOL_MINT;
  // ExactIn for SELL (we know exactly how much SOL we give up).
  // ExactOut for BUY  (we know exactly how much SOL we want to receive).
  const swapMode   = side === 'SELL' ? 'ExactIn' : 'ExactOut';

  const quoteUrl =
    `${apiUrl}/quote` +
    `?inputMint=${inputMint}` +
    `&outputMint=${outputMint}` +
    `&amount=${lamports}` +
    `&slippageBps=${slippageBps}` +
    `&swapMode=${swapMode}` +
    `&onlyDirectRoutes=false`;

  const quoteResponse = await jupiterGet<JupiterQuoteResponse>(quoteUrl);

  const swapApiResponse = await jupiterPost<JupiterSwapApiResponse>(`${apiUrl}/swap`, {
    quoteResponse,
    userPublicKey,
    wrapAndUnwrapSol: true,
    dynamicComputeUnitLimit: true,
    prioritizationFeeLamports: 'auto',
  });

  return {
    quoteResponse,
    swapTransaction: swapApiResponse.swapTransaction,
    inAmountLamports: BigInt(quoteResponse.inAmount),
    outAmountLamports: BigInt(quoteResponse.outAmount),
    priceImpactPct: quoteResponse.priceImpactPct,
  };
}
