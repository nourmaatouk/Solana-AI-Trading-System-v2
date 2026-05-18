import { useEffect, useRef, useCallback } from 'react';
import {
  createChart,
  ColorType,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type IPriceLine,
  type CandlestickData,
  type HistogramData,
  type LineData,
  type Time,
} from 'lightweight-charts';
import { useTradingStore } from '../store';
import { socketService } from '../services/socket';

const TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1D'];

const TZ_OFFSET_S = -new Date().getTimezoneOffset() * 60;
function localTime(utcSeconds: number): Time {
  return (utcSeconds + TZ_OFFSET_S) as Time;
}

function calcSMA(closes: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  for (let i = 0; i < closes.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else {
      const sum = closes.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
  }
  return result;
}

export function PriceChart() {
  const klines = useTradingStore((s) => s.klines);
  const selectedInterval = useTradingStore((s) => s.selectedInterval);
  const setSelectedInterval = useTradingStore((s) => s.setSelectedInterval);
  const ticker = useTradingStore((s) => s.ticker);

  const theme = useTradingStore((s) => s.theme);

  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const ma7Ref = useRef<ISeriesApi<'Line'> | null>(null);
  const ma25Ref = useRef<ISeriesApi<'Line'> | null>(null);
  const ma99Ref = useRef<ISeriesApi<'Line'> | null>(null);
  const priceLineRef = useRef<IPriceLine | null>(null);

  const prevLenRef = useRef(0);
  const prevFirstTimeRef = useRef<number>(0);
  const hasFittedRef = useRef(false);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1e2329' },
        textColor: '#848e9c',
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: '#2b3139' },
        horzLines: { color: '#2b3139' },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#2b3139', textColor: '#848e9c' },
      timeScale: { borderColor: '#2b3139', timeVisible: true, secondsVisible: false },
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#0ecb81', downColor: '#f6465d',
      borderUpColor: '#0ecb81', borderDownColor: '#f6465d',
      wickUpColor: '#0ecb81', wickDownColor: '#f6465d',
    });

    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });
    chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

    const ma7Series = chart.addLineSeries({ color: '#8B5CF6', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    const ma25Series = chart.addLineSeries({ color: '#14B8A6', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    const ma99Series = chart.addLineSeries({ color: '#f6465d', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });

    chartRef.current = chart;
    candleRef.current = candleSeries;
    volumeRef.current = volumeSeries;
    ma7Ref.current = ma7Series;
    ma25Ref.current = ma25Series;
    ma99Ref.current = ma99Series;

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      volumeRef.current = null;
      ma7Ref.current = null;
      ma25Ref.current = null;
      ma99Ref.current = null;
    };
  }, []);

  useEffect(() => {
    if (!candleRef.current || !volumeRef.current || klines.length === 0) {
      prevLenRef.current = 0;
      return;
    }

    const prev = prevLenRef.current;
    const curr = klines.length;
    const currFirstTime = klines[0]?.time ?? 0;
    const isSnapshot = curr - prev > 1 || prev === 0 || currFirstTime !== prevFirstTimeRef.current;
    prevLenRef.current = curr;
    prevFirstTimeRef.current = currFirstTime;

    const last = klines[curr - 1];

    if (isSnapshot) {
      const candleData: CandlestickData[] = klines.map((k) => ({
        time: localTime(k.time),
        open: parseFloat(k.open),
        high: parseFloat(k.high),
        low: parseFloat(k.low),
        close: parseFloat(k.close),
      }));
      const volumeData: HistogramData[] = klines.map((k) => ({
        time: localTime(k.time),
        value: parseFloat(k.volume),
        color: parseFloat(k.close) >= parseFloat(k.open) ? '#0ecb8140' : '#f6465d40',
      }));
      const closes = klines.map((k) => parseFloat(k.close));
      const ma7 = calcSMA(closes, 7);
      const ma25 = calcSMA(closes, 25);
      const ma99 = calcSMA(closes, 99);
      const ma7Data: LineData[] = klines.map((k, i) => ({ time: localTime(k.time), value: ma7[i] })).filter((d): d is LineData => d.value !== null);
      const ma25Data: LineData[] = klines.map((k, i) => ({ time: localTime(k.time), value: ma25[i] })).filter((d): d is LineData => d.value !== null);
      const ma99Data: LineData[] = klines.map((k, i) => ({ time: localTime(k.time), value: ma99[i] })).filter((d): d is LineData => d.value !== null);

      candleRef.current.setData(candleData);
      volumeRef.current.setData(volumeData);
      ma7Ref.current?.setData(ma7Data);
      ma25Ref.current?.setData(ma25Data);
      ma99Ref.current?.setData(ma99Data);

      if (!hasFittedRef.current) {
        chartRef.current?.timeScale().fitContent();
        hasFittedRef.current = true;
      }
    } else {
      candleRef.current.update({
        time: localTime(last.time),
        open: parseFloat(last.open),
        high: parseFloat(last.high),
        low: parseFloat(last.low),
        close: parseFloat(last.close),
      });
      volumeRef.current.update({
        time: localTime(last.time),
        value: parseFloat(last.volume),
        color: parseFloat(last.close) >= parseFloat(last.open) ? '#0ecb8140' : '#f6465d40',
      });
      const closes = klines.map((k) => parseFloat(k.close));
      const i = curr - 1;
      const ma7v = i >= 6  ? closes.slice(i - 6,  i + 1).reduce((a, b) => a + b, 0) / 7  : null;
      const ma25v = i >= 24 ? closes.slice(i - 24, i + 1).reduce((a, b) => a + b, 0) / 25 : null;
      const ma99v = i >= 98 ? closes.slice(i - 98, i + 1).reduce((a, b) => a + b, 0) / 99 : null;
      if (ma7v !== null)  ma7Ref.current?.update({ time: localTime(last.time), value: ma7v });
      if (ma25v !== null) ma25Ref.current?.update({ time: localTime(last.time), value: ma25v });
      if (ma99v !== null) ma99Ref.current?.update({ time: localTime(last.time), value: ma99v });
    }
  }, [klines]);

  useEffect(() => {
    if (!ticker || !candleRef.current) return;
    const price = parseFloat(ticker.price);

    if (priceLineRef.current) {
      candleRef.current.removePriceLine(priceLineRef.current);
    }
    priceLineRef.current = candleRef.current.createPriceLine({
      price,
      color: '#8B5CF6',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      axisLabelVisible: true,
      title: `$${price.toFixed(2)}`,
    });
  }, [ticker]);

  // Re-apply chart colors when theme switches
  useEffect(() => {
    if (!chartRef.current) return;
    const isDark = theme === 'dark';
    chartRef.current.applyOptions({
      layout: {
        background: { type: ColorType.Solid, color: isDark ? '#1e2329' : '#ffffff' },
        textColor: isDark ? '#848e9c' : '#6b7280',
      },
      grid: {
        vertLines: { color: isDark ? '#2b3139' : '#f3f4f6' },
        horzLines: { color: isDark ? '#2b3139' : '#f3f4f6' },
      },
      rightPriceScale: { borderColor: isDark ? '#2b3139' : '#e5e7eb', textColor: isDark ? '#848e9c' : '#6b7280' },
      timeScale: { borderColor: isDark ? '#2b3139' : '#e5e7eb' },
    });
  }, [theme]);

  const setKlines = useTradingStore((s) => s.setKlines);

  const handleIntervalChange = useCallback((interval: string) => {
    setSelectedInterval(interval);
    setKlines([]);
    hasFittedRef.current = false;
    prevFirstTimeRef.current = 0;
    socketService.send({ type: 'changeInterval', interval });
  }, [setSelectedInterval, setKlines]);

  return (
    <div className="flex flex-col h-full" style={{ background: 'var(--bg-panel)' }}>
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-3 py-1.5" style={{ borderBottom: '1px solid var(--border)' }}>
        <div className="flex items-center gap-1">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf}
              onClick={() => handleIntervalChange(tf)}
              className="px-2 py-0.5 rounded text-xs transition-colors"
              style={selectedInterval === tf
                ? { background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--accent-bd)' }
                : { color: 'var(--text-2)', border: '1px solid transparent' }
              }
            >
              {tf}
            </button>
          ))}
        </div>
        <div className="h-3 w-px" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-3 text-[10px]">
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 inline-block" style={{ background: '#8B5CF6' }} />
            <span style={{ color: 'var(--text-3)' }}>MA7</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 inline-block" style={{ background: '#14B8A6' }} />
            <span style={{ color: 'var(--text-3)' }}>MA25</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 inline-block" style={{ background: '#f6465d' }} />
            <span style={{ color: 'var(--text-3)' }}>MA99</span>
          </span>
        </div>
      </div>

      {/* Chart canvas */}
      <div ref={containerRef} className="flex-1 relative">
        {klines.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-sm" style={{ color: 'var(--text-3)' }}>Waiting for market data...</div>
          </div>
        )}
      </div>
    </div>
  );
}
