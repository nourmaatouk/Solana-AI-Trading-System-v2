import { useTradingStore } from '../store';

export function StatusIndicator() {
  const connectionStatus = useTradingStore((s) => s.connectionStatus);
  const exchange = useTradingStore((s) => s.exchange);

  const configs = {
    connected: {
      dot: 'bg-green-400 animate-pulse',
      text: 'text-green-400',
      label: 'Live',
      source: `${exchange.toUpperCase()} WebSocket`,
    },
    reconnecting: {
      dot: 'bg-amber-400 animate-pulse',
      text: 'text-amber-400',
      label: 'Reconnecting',
      source: `${exchange.toUpperCase()} WebSocket`,
    },
    disconnected: {
      dot: 'bg-red-500',
      text: 'text-red-400',
      label: 'Offline',
      source: 'No feed',
    },
  };

  const cfg = configs[connectionStatus];

  return (
    <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#2b3139]/80">
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      <span className={`text-[10px] font-semibold ${cfg.text}`}>{cfg.label}</span>
      <span className="text-[#848e9c] text-[10px]">· {cfg.source}</span>
    </div>
  );
}
