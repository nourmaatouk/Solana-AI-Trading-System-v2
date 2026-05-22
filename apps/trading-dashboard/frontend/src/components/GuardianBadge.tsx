import type { GuardianResultDto } from '../services/executionApi';

interface GuardianBadgeProps {
  result: GuardianResultDto | null;
}

const STYLES: Record<string, { bg: string; border: string; fg: string; label: string }> = {
  APPROVE: { bg: 'var(--green-bg)', border: 'var(--green-bd)', fg: 'var(--green)', label: 'APPROVED' },
  BLOCK:   { bg: 'var(--red-bg)',   border: 'var(--red-bd)',   fg: 'var(--red)',   label: 'BLOCKED'   },
  PENDING: { bg: 'var(--bg-sub)',   border: 'var(--border)',   fg: 'var(--text-3)', label: 'NOT EVALUATED' },
};

export function GuardianBadge({ result }: GuardianBadgeProps) {
  const key = result?.decision ?? 'PENDING';
  const style = STYLES[key] ?? STYLES.PENDING;
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold border"
      style={{ background: style.bg, borderColor: style.border, color: style.fg }}
      title={result?.blocked_by ?? ''}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: style.fg }} />
      Guardian · {style.label}
      {result?.blocked_by && <span className="font-mono ml-1">({result.blocked_by})</span>}
    </span>
  );
}
