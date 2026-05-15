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
    trend === 'up' ? 'text-[#16a34a]' :
    trend === 'down' ? 'text-[#dc2626]' :
    'text-gray-700';

  return (
    <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100 flex flex-col gap-2">
      <div className="flex items-start justify-between">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">{title}</span>
        {icon && <div className="text-gray-400">{icon}</div>}
        {badge && <div>{badge}</div>}
      </div>
      <div className={`text-2xl font-bold tracking-tight ${trendColor}`}>{value}</div>
      {subtitle && <div className="text-xs text-gray-400">{subtitle}</div>}
    </div>
  );
}
