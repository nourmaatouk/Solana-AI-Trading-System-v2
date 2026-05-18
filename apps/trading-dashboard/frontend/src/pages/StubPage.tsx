interface StubPageProps {
  title: string;
  description?: string;
}

export function StubPage({ title, description }: StubPageProps) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
      <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5" style={{ background: 'linear-gradient(135deg, #6D28D9 0%, #14B8A6 100%)' }}>
        <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
        </svg>
      </div>
      <h1 className="text-xl font-bold mb-2" style={{ color: 'var(--text-1)' }}>{title}</h1>
      <p className="text-sm max-w-sm" style={{ color: 'var(--text-3)' }}>
        {description ?? 'This section is coming soon. Stay tuned for updates.'}
      </p>
      <div className="mt-6 px-4 py-2 rounded-full text-xs font-semibold" style={{ background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--accent-bd)' }}>
        Coming soon
      </div>
    </div>
  );
}
