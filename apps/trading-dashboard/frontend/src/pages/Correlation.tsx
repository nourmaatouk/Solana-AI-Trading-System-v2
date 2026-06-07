import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, type IChartApi, type ISeriesApi, type LineData, type UTCTimestamp } from 'lightweight-charts';
import { fetchCorrelation, type CorrelationData, type CorrelationCandle } from '../services/api';
import { EXECUTION_PANEL_REFRESH_MS } from '../config';

const card: React.CSSProperties = { background: 'var(--bg-panel)', border: '1px solid var(--border)' };

const FEATURE_META: Record<string, { label: string; description: string; unit: string }> = {
  btc_return_4:  { label: 'BTC Return (1h)',   description: 'BTC price change over the last 4 candles (1 hour).', unit: '%' },
  btc_return_16: { label: 'BTC Return (4h)',   description: 'BTC price change over the last 16 candles (4 hours).', unit: '%' },
  btc_vol_12:    { label: 'BTC Volatility (3h)', description: 'Standard deviation of BTC returns over 12 candles.', unit: '' },
  btc_vs_sma20:  { label: 'BTC vs SMA20',      description: 'Distance between BTC price and its 20-period moving average. Positive = above trend.', unit: '' },
  btc_rsi_14:    { label: 'BTC RSI (14)',       description: 'Momentum oscillator — above 70 is overbought, below 30 is oversold.', unit: '' },
};

function fmt(val: number | null, unit: string): string {
  if (val === null || val === undefined) return '—';
  const v = unit === '%' ? val * 100 : val;
  return `${v.toFixed(3)}${unit}`;
}

function fmtCorr(r: number | null): string {
  if (r === null) return '—';
  return r.toFixed(3);
}

function corrColor(r: number | null): string {
  if (r === null) return 'var(--text-3)';
  if (r >= 0.7) return 'var(--accent)';
  if (r >= 0.4) return '#f59e0b';
  return 'var(--red)';
}

function corrLabel(r: number | null): string {
  if (r === null) return 'Unknown';
  if (r >= 0.7) return 'Strong positive';
  if (r >= 0.4) return 'Moderate';
  if (r >= 0) return 'Weak positive';
  if (r >= -0.4) return 'Weak negative';
  return 'Strong negative';
}

function toLineData(candles: CorrelationCandle[]): LineData<UTCTimestamp>[] {
  const seen = new Set<number>();
  return candles
    .map((c) => ({ time: Math.floor(c.timestamp / 1000) as UTCTimestamp, value: c.close }))
    .filter((d) => { if (seen.has(d.time)) return false; seen.add(d.time); return true; })
    .sort((a, b) => a.time - b.time);
}

function normalise(data: LineData<UTCTimestamp>[]): LineData<UTCTimestamp>[] {
  if (!data.length) return [];
  const base = data[0].value;
  if (!base) return data;
  return data.map((d) => ({ ...d, value: ((d.value / base) - 1) * 100 }));
}

export function Correlation() {
  const [data, setData] = useState<CorrelationData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const chartRef = useRef<HTMLDivElement>(null);
  const chartApi = useRef<IChartApi | null>(null);
  const btcSeries = useRef<ISeriesApi<'Line'> | null>(null);
  const solSeries = useRef<ISeriesApi<'Line'> | null>(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const d = await fetchCorrelation(60);
        if (alive) { setData(d); setError(null); }
      } catch (e) {
        if (alive) setError(String(e));
      } finally {
        if (alive) setLoading(false);
      }
    };
    load();
    const id = setInterval(load, EXECUTION_PANEL_REFRESH_MS);
    return () => { alive = false; clearInterval(id); };
  }, []);

  // Build chart once on mount, update series data when data changes
  useEffect(() => {
    if (!chartRef.current) return;
    const text   = getComputedStyle(document.documentElement).getPropertyValue('--text-3').trim()  || '#848e9c';
    const border = getComputedStyle(document.documentElement).getPropertyValue('--border').trim()   || '#2b3139';
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()   || '#00c2ff';

    const chart = createChart(chartRef.current, {
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: text, attributionLogo: false },
      grid: { vertLines: { color: border }, horzLines: { color: border } },
      rightPriceScale: { borderColor: border },
      timeScale: { borderColor: border, timeVisible: true, secondsVisible: false },
      width: chartRef.current.clientWidth,
      height: 260,
    });
    btcSeries.current = chart.addLineSeries({ color: '#f59e0b', lineWidth: 2, priceLineVisible: false, lastValueVisible: true, title: 'BTC %' });
    solSeries.current = chart.addLineSeries({ color: accent,    lineWidth: 2, priceLineVisible: false, lastValueVisible: true, title: 'SOL %' });
    chartApi.current = chart;

    const ro = new ResizeObserver(() => {
      if (chartRef.current) chart.applyOptions({ width: chartRef.current.clientWidth });
    });
    ro.observe(chartRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartApi.current = null;
      btcSeries.current = null;
      solSeries.current = null;
    };
  }, []);

  useEffect(() => {
    if (!data || !btcSeries.current || !solSeries.current) return;
    btcSeries.current.setData(normalise(toLineData(data.btc_candles)));
    solSeries.current.setData(normalise(toLineData(data.sol_candles)));
    chartApi.current?.timeScale().fitContent();
  }, [data]);

  const feat = data?.btc_features ?? {};

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5" style={{ background: 'var(--bg-page)' }}>
      <div>
        <h1 className="text-xl font-bold" style={{ color: 'var(--text-1)' }}>BTC / SOL Correlation</h1>
        <p className="text-sm mt-1" style={{ color: 'var(--text-3)' }}>
          Bitcoin acts as a market barometer for Solana. The AI model uses 5 BTC-derived features
          alongside SOL technicals — this page shows what those features look like right now.
        </p>
      </div>

      {error && (
        <div className="text-sm px-4 py-3 rounded-lg"
          style={{ color: 'var(--red)', background: 'var(--red-bg)', border: '1px solid var(--red-bd)' }}>
          {error}
        </div>
      )}

      {/* Correlation coefficient */}
      <div className="rounded-xl p-5" style={card}>
        <div className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: 'var(--text-2)' }}>
          Rolling correlation (last 60 candles · 15 min)
        </div>
        <div className="flex items-end gap-4">
          <span className="text-5xl font-bold tabular-nums" style={{ color: corrColor(data?.rolling_correlation ?? null) }}>
            {loading ? '…' : fmtCorr(data?.rolling_correlation ?? null)}
          </span>
          <div className="mb-1.5 space-y-0.5">
            <div className="text-sm font-semibold" style={{ color: corrColor(data?.rolling_correlation ?? null) }}>
              {loading ? '' : corrLabel(data?.rolling_correlation ?? null)}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-3)' }}>
              Pearson r between BTC and SOL 15-min close returns
            </div>
          </div>
        </div>
        <div className="mt-3 text-xs" style={{ color: 'var(--text-3)' }}>
          A value close to <strong style={{ color: 'var(--text-2)' }}>+1.0</strong> means BTC and SOL move together — when BTC falls, SOL likely follows.
          The model uses this relationship to filter out false SOL signals that go against the broader market trend.
        </div>
      </div>

      {/* BTC co-features */}
      <div>
        <div className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: 'var(--text-2)' }}>
          Live BTC co-features fed to XGBoost
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {Object.entries(FEATURE_META).map(([key, meta]) => {
            const val = (feat as Record<string, number | null>)[key] ?? null;
            const isReturn = key.startsWith('btc_return');
            const trend = isReturn && val !== null ? (val > 0 ? 'up' : val < 0 ? 'down' : 'neutral') : 'neutral';
            const trendColor = trend === 'up' ? 'var(--accent)' : trend === 'down' ? 'var(--red)' : 'var(--text-1)';
            return (
              <div key={key} className="rounded-xl p-4" style={card} title={meta.description}>
                <div className="text-[10px] font-semibold uppercase tracking-wide mb-1" style={{ color: 'var(--text-3)' }}>
                  {meta.label}
                </div>
                <div className="text-xl font-bold tabular-nums" style={{ color: loading ? 'var(--text-3)' : trendColor }}>
                  {loading ? '…' : fmt(val, meta.unit)}
                </div>
                <div className="text-[10px] mt-1 leading-tight" style={{ color: 'var(--text-3)' }}>
                  {meta.description}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Normalised price chart */}
      <div className="rounded-xl p-4" style={card}>
        <div className="flex items-center justify-between mb-3">
          <div className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-2)' }}>
            Price change since open — BTC vs SOL (normalised %)
          </div>
          <div className="flex items-center gap-4 text-[11px]" style={{ color: 'var(--text-3)' }}>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-0.5 rounded" style={{ background: '#f59e0b' }} /> BTC
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-0.5 rounded" style={{ background: 'var(--accent)' }} /> SOL
            </span>
          </div>
        </div>
        {(data?.btc_candles.length ?? 0) === 0 && !loading ? (
          <div className="text-xs text-center py-10" style={{ color: 'var(--text-3)' }}>
            No candle data yet — run the pipeline first.
          </div>
        ) : (
          <div ref={chartRef} />
        )}
        <div className="mt-2 text-[10px]" style={{ color: 'var(--text-3)' }}>
          Both series rebased to 0% at the first candle so moves are directly comparable.
        </div>
      </div>

      {/* Explainer */}
      <div className="rounded-xl p-5" style={card}>
        <div className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: 'var(--text-2)' }}>
          Why does the model watch BTC?
        </div>
        <div className="space-y-2 text-xs" style={{ color: 'var(--text-3)' }}>
          <p>
            BTC is the crypto market's leading indicator. When BTC drops sharply, almost every altcoin —
            including SOL — follows within 1–4 candles. Without this context, the model might see SOL
            technicals looking bullish while BTC is crashing, and issue a wrong BUY signal.
          </p>
          <p>
            During training, XGBoost learned the historical correlation between BTC movements and SOL
            price direction. At inference time it uses the 5 live BTC features above to filter out
            signals that go against the broader market trend.
          </p>
          <p>
            <strong style={{ color: 'var(--text-2)' }}>You are always trading SOL</strong> — BTC is
            used as context only, never as a trade target.
          </p>
        </div>
      </div>
    </div>
  );
}
