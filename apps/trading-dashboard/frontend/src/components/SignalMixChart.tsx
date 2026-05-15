interface HistoryEntry {
  action: string;
}

interface SignalMixChartProps {
  history: HistoryEntry[];
  loading: boolean;
}

const COLORS = {
  BUY: '#22c55e',
  HOLD: '#a855f7',
  SELL: '#ef4444',
};

interface Slice {
  label: 'BUY' | 'HOLD' | 'SELL';
  count: number;
  color: string;
}

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number): string {
  const start = polarToCartesian(cx, cy, r, endAngle);
  const end = polarToCartesian(cx, cy, r, startAngle);
  const largeArc = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}

export function SignalMixChart({ history, loading }: SignalMixChartProps) {
  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 h-full flex flex-col">
        <div className="h-4 w-32 skeleton mb-4" />
        <div className="flex-1 flex items-center justify-center">
          <div className="w-32 h-32 rounded-full skeleton" />
        </div>
      </div>
    );
  }

  const counts = { BUY: 0, HOLD: 0, SELL: 0 };
  for (const entry of history) {
    const action = (entry.action || 'HOLD').toUpperCase() as 'BUY' | 'HOLD' | 'SELL';
    if (action in counts) counts[action]++;
    else counts.HOLD++;
  }

  const total = counts.BUY + counts.HOLD + counts.SELL;

  const allSlices: Slice[] = [
    { label: 'BUY', count: counts.BUY, color: COLORS.BUY },
    { label: 'HOLD', count: counts.HOLD, color: COLORS.HOLD },
    { label: 'SELL', count: counts.SELL, color: COLORS.SELL },
  ];
  const slices: Slice[] = allSlices.filter((s) => s.count > 0);

  const cx = 80;
  const cy = 80;
  const r = 55;
  const innerR = 32;

  // Build arcs
  let currentAngle = 0;
  const arcs = slices.map((slice) => {
    const sweep = total > 0 ? (slice.count / total) * 360 : 0;
    const startAngle = currentAngle;
    const endAngle = currentAngle + sweep;
    currentAngle = endAngle;
    return { ...slice, startAngle, endAngle, sweep };
  });

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 h-full flex flex-col">
      <div className="mb-3">
        <div className="text-sm font-semibold text-gray-800">Signal Mix</div>
        <div className="text-xs text-gray-400">{total} total signals</div>
      </div>

      <div className="flex-1 flex flex-col items-center gap-4">
        {total === 0 ? (
          <div className="flex-1 flex items-center justify-center text-xs text-gray-400">
            No signal history yet
          </div>
        ) : (
          <>
            {/* Donut SVG */}
            <svg viewBox="0 0 160 160" className="w-36 h-36 flex-shrink-0">
              {arcs.map((arc, i) => {
                if (arc.sweep < 0.5) return null;
                // For full circle (100%), handle as special case
                if (arc.sweep >= 359.9) {
                  return (
                    <circle
                      key={i}
                      cx={cx}
                      cy={cy}
                      r={r}
                      fill="none"
                      stroke={arc.color}
                      strokeWidth={r - innerR}
                    />
                  );
                }
                const outerPath = describeArc(cx, cy, r, arc.startAngle, arc.endAngle);
                return (
                  <path
                    key={i}
                    d={outerPath}
                    fill="none"
                    stroke={arc.color}
                    strokeWidth={r - innerR}
                    strokeLinecap="butt"
                  />
                );
              })}
              {/* Inner circle white hole */}
              <circle cx={cx} cy={cy} r={innerR} fill="white" />
              {/* Center text */}
              <text x={cx} y={cy - 4} textAnchor="middle" fontSize={9} fill="#6b7280" fontWeight="500">
                Total
              </text>
              <text x={cx} y={cy + 9} textAnchor="middle" fontSize={13} fill="#111827" fontWeight="700">
                {total}
              </text>
            </svg>

            {/* Legend */}
            <div className="w-full space-y-1.5">
              {slices.map((slice) => {
                const pct = total > 0 ? ((slice.count / total) * 100).toFixed(0) : '0';
                return (
                  <div key={slice.label} className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ background: slice.color }} />
                      <span className="text-gray-600 font-medium">{slice.label}</span>
                    </div>
                    <div className="flex items-center gap-2 text-gray-500">
                      <span>{slice.count}</span>
                      <span className="text-gray-300">·</span>
                      <span>{pct}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
