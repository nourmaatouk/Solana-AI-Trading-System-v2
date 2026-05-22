/**
 * Drop-in replacement for `AbortSignal.timeout(ms)` for older browsers.
 *
 * `AbortSignal.timeout` is only available in Chrome 103+ / Firefox 100+ /
 * Safari 16+. Using it directly would throw "AbortSignal.timeout is not a
 * function" on older browsers and break every page that calls into the API.
 *
 * The setTimeout handle is intentionally not cleared on success: aborting a
 * signal whose fetch already resolved is a no-op, so leaking the timer for
 * a few seconds is harmless and keeps the call site as terse as the native
 * API would be.
 */
export function timeoutSignal(ms: number): AbortSignal {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), ms);
  return controller.signal;
}
