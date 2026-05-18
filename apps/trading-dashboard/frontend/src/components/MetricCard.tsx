import { ReactNode } from 'react';

interface MetricCardProps {
  title: string;
  value: ReactNode;
  subtitle?: ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  badge?: ReactNode;
  icon?: ReactNode;
}

export function MetricCard({ title, value, subtitle, trend, badge, icon }: MetricCardProps) {
  const trendColor =
    trend === 'up' ? '#16a34a' :
    trend === 'down' ? '#dc2626' :
    undefined;

  return (
    <div className="rounded-xl p-5 flex flex-col gap-2" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
      <div className="flex items-start justify-between">
        <span className="text-xs font-medium uppercase tracking-wide" style={{ color: 'var(--text-2)' }}>{title}</span>
        {icon && <div style={{ color: 'var(--text-3)' }}>{icon}</div>}
        {badge && <div>{badge}</div>}
      </div>
      <div className="text-2xl font-bold tracking-tight" style={{ color: trendColor ?? 'var(--text-1)' }}>{value}</div>
      {subtitle && <div className="text-xs" style={{ color: 'var(--text-3)' }}>{subtitle}</div>}
    </div>
  );
}
