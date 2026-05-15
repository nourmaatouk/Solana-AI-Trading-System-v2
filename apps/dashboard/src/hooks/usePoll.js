import { useState, useEffect, useCallback } from 'react'

export function usePoll(fetchFn, intervalMs = 30_000) {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  const run = useCallback(async (signal) => {
    try {
      const result = await fetchFn({ signal })
      setData(result)
      setError(null)
    } catch (e) {
      if (e.name !== 'AbortError') setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [fetchFn])

  useEffect(() => {
    const ctrl = new AbortController()
    run(ctrl.signal)
    const id = setInterval(() => run(ctrl.signal), intervalMs)
    return () => { ctrl.abort(); clearInterval(id) }
  }, [run, intervalMs])

  return { data, loading, error }
}
