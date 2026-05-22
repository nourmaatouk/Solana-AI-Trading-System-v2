import { ACTIVE_SYMBOL, QUOTE_ASSETS } from '../config';

export function parseMarketSymbol(raw: string = ACTIVE_SYMBOL): { base: string; quote: string } {
  const symbol = raw.replace('/', '').toUpperCase();
  const quote = QUOTE_ASSETS.find((asset) => symbol.endsWith(asset));
  if (!quote) return { base: symbol, quote: '' };
  return { base: symbol.slice(0, -quote.length), quote };
}

export function formatMarketPair(raw: string = ACTIVE_SYMBOL): string {
  const { base, quote } = parseMarketSymbol(raw);
  return quote ? `${base}/${quote}` : base;
}
