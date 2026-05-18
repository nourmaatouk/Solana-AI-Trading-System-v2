interface HistoryEntry {
  run_utc: string;
  equity: number;
  action: string;
}

interface PortfolioChartProps {
  history: HistoryEntry[];
  loading: boolean;
}

function formatDate(utcStr: string): string {
  try {
    const d = new Date(utcStr);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return '';
  }
}

export function PortfolioChart({ history, loading }: PortfolioChartProps) {
  const hasData = history.length >= 2;

  if (loading) {
    return (
      <div className="rounded-xl p-5 h-full flex flex-col" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
        <div className="h-4 w-48 skeleton mb-4" />
        <div className="flex-1 skeleton rounded-lg" />
      </div>
    );
  }

  if (!hasData) {
    return (
      <div className="rounded-xl p-5 h-full flex flex-col" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
        <div className="mb-3">
          <div className="text-sm font-semibold" style={{ color: 'var(--text-1)' }}>Portfolio Performance</div>
          <div className="text-xs" style={{ color: 'var(--text-3)' }}>30-day equity curve</div>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-2">
          <svg className="w-10 h-10" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24" style={{ color: 'var(--border)' }}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p className="text-sm" style={{ color: 'var(--text-3)' }}>Waiting for data</p>
          <p className="text-xs" style={{ color: 'var(--text-3)' }}>(scheduler runs every 15 min)</p>
        </div>
      </div>
    );
  }

  const equities = history.map((h) => h.equity);
  const minEq = Math.min(...equities);
  const maxEq = Math.max(...equities);
  const range = maxEq - minEq || 0.001;

  const WIDTH = 500;
  const HEIGHT = 180;
  const PAD_L = 48;
  const PAD_R = 12;
  const PAD_T = 12;
  const PAD_B = 24;

  const chartW = WIDTH - PAD_L - PAD_R;
  const chartH = HEIGHT - PAD_T - PAD_B;

  const points = history.map((h, i) => {
    const x = PAD_L + (i / (history.length - 1)) * chartW;
    const y = PAD_T + chartH - ((h.equity - minEq) / range) * chartH;
    return { x, y, entry: h };
  });

  const linePath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)},${p.y.toFixed(1)}`)
    .join(' ');

  const areaPath = `${linePath} L ${points[points.length - 1].x.toFixed(1)},${(PAD_T + chartH).toFixed(1)} L ${PAD_L},${(PAD_T + chartH).toFixed(1)} Z`;

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => ({
    y: PAD_T + chartH - t * chartH,
    label: (minEq + t * range).toFixed(3),
  }));

  const step = Math.max(1, Math.floor(history.length / 4));
  const xTicks = history
    .filter((_, i) => i % step === 0 || i === history.length - 1)
    .map((h) => {
      const idx = history.indexOf(h);
      const x = PAD_L + (idx / (history.length - 1)) * chartW;
      return { x, label: formatDate(h.run_utc) };
    });

  const seenLabels = new Set<string>();
  const filteredXTicks = xTicks.filter(t => {
    if (seenLabels.has(t.label)) return false;
    seenLabels.add(t.label);
    return true;
  });

  return (
    <div className="rounded-xl p-5 h-full flex flex-col" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
      <div className="mb-3">
        <div className="text-sm font-semibold" style={{ color: 'var(--text-1)' }}>Portfolio Performance</div>
        <div className="text-xs" style={{ color: 'var(--text-3)' }}>30-day equity curve · Paper Trading</div>
      </div>

      <div className="flex-1 min-h-0">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" className="w-full h-full" style={{ minHeight: 120 }}>
          <defs>
            <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#8B5CF6" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#14B8A6" stopOpacity="0.02" />
            </linearGradient>
            <clipPath id="chartClip">
              <rect x={PAD_L} y={PAD_T} width={chartW} height={chartH} />
            </clipPath>
          </defs>

          {yTicks.map((tick, i) => (
            <g key={i}>
              <line x1={PAD_L} y1={tick.y} x2={PAD_L + chartW} y2={tick.y} stroke="#2b3139" strokeWidth={1} />
              <text x={PAD_L - 4} y={tick.y + 3} fontSize={8} fill="#5e6673" textAnchor="end">{tick.label}</text>
            </g>
          ))}

          <path d={areaPath} fill="url(#equityGrad)" clipPath="url(#chartClip)" />
          <path d={linePath} fill="none" stroke="#8B5CF6" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" clipPath="url(#chartClip)" />

          {filteredXTicks.map((tick, i) => (
            <text key={i} x={tick.x} y={HEIGHT - 4} fontSize={8} fill="#5e6673" textAnchor="middle">{tick.label}</text>
          ))}
        </svg>
      </div>
    </div>
  );
}
