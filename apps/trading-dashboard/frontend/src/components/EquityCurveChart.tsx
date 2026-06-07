import { useEffect, useRef } from 'react';
import {
  createChart,
  ColorType,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from 'lightweight-charts';

export interface EquityPoint {
  time: number;   // UTC unix seconds
  value: number;  // equity multiple (1.0 = starting balance)
}

interface Props {
  /** Strategy equity over time (1.0 = start). */
  strategy: EquityPoint[];
  /** Buy-and-hold benchmark over the same period (1.0 = start). */
  benchmark: EquityPoint[];
  height?: number;
}

function cssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

// lightweight-charts requires strictly ascending, unique timestamps and throws
// otherwise — dedupe by time (keep last) and sort, so a duplicated timestamp
// can never crash the page.
function toLineData(points: EquityPoint[]) {
  const byTime = new Map<number, number>();
  for (const p of points) {
    if (p && Number.isFinite(p.time) && Number.isFinite(p.value)) byTime.set(p.time, p.value);
  }
  return Array.from(byTime.keys())
    .sort((a, b) => a - b)
    .map((t) => ({ time: t as UTCTimestamp, value: byTime.get(t)! }));
}

export function EquityCurveChart({ strategy, benchmark, height = 240 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const stratRef = useRef<ISeriesApi<'Line'> | null>(null);
  const benchRef = useRef<ISeriesApi<'Line'> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const text = cssVar('--text-3', '#848e9c');
    const border = cssVar('--border', '#2b3139');
    const green = cssVar('--green', '#0ecb81');

    const chart = createChart(containerRef.current, {
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: text, attributionLogo: false },
      grid: { vertLines: { color: border }, horzLines: { color: border } },
      rightPriceScale: { borderColor: border },
      timeScale: { borderColor: border, timeVisible: false, secondsVisible: false },
      crosshair: { horzLine: { labelVisible: true }, vertLine: { labelVisible: true } },
      width: containerRef.current.clientWidth,
      height,
    });

    stratRef.current = chart.addLineSeries({
      color: green, lineWidth: 2, priceLineVisible: false, lastValueVisible: true,
    });
    benchRef.current = chart.addLineSeries({
      color: text, lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false, lastValueVisible: true,
    });
    chartRef.current = chart;

    const ro = new ResizeObserver(() => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth });
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      stratRef.current = null;
      benchRef.current = null;
    };
  }, [height]);

  useEffect(() => {
    if (!stratRef.current || !benchRef.current) return;
    stratRef.current.setData(toLineData(strategy));
    benchRef.current.setData(toLineData(benchmark));
    chartRef.current?.timeScale().fitContent();
  }, [strategy, benchmark]);

  return <div ref={containerRef} style={{ width: '100%', height }} />;
}
