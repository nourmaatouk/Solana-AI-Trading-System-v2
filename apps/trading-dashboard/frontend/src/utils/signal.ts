/** Minutes after which a signal is considered stale — matches SIGNAL_STALE_MINUTES in main.py */
export const SIGNAL_STALE_MINUTES = 20;

/** Spoofing thresholds — must match SPOOFING_SEVERITY_HIGH/MEDIUM in fusion/config.py */
export const SPOOFING_HIGH   = 0.6;
export const SPOOFING_MEDIUM = 0.3;

/**
 * Extract a sentiment label from the FinBERT reasoning string.
 * Reasoning format: "FinBERT NEGATIVE (-0.360), event=news"
 */
export function parseSentimentLabel(reasoning: string): string {
  const upper = reasoning.toUpperCase();
  if (upper.includes('POSITIVE')) return 'POSITIVE';
  if (upper.includes('NEGATIVE')) return 'NEGATIVE';
  if (upper.includes('BULLISH'))  return 'BULLISH';
  if (upper.includes('BEARISH'))  return 'BEARISH';
  return 'NEUTRAL';
}

/** Minutes since the signal was generated, or null if timestamp is missing/invalid. */
export function signalAgeMinutes(generated_at_utc: string | undefined): number | null {
  if (!generated_at_utc) return null;
  try {
    return Math.floor((Date.now() - new Date(generated_at_utc).getTime()) / 60000);
  } catch {
    return null;
  }
}
