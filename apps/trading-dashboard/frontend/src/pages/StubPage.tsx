interface StubPageProps {
  title: string;
  description?: string;
}

export function StubPage({ title, description }: StubPageProps) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
      <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5" style={{ background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)' }}>
        <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
        </svg>
      </div>
      <h1 className="text-xl font-bold text-gray-800 mb-2">{title}</h1>
      <p className="text-sm text-gray-400 max-w-sm">
        {description ?? 'This section is coming soon. Stay tuned for updates.'}
      </p>
      <div className="mt-6 px-4 py-2 rounded-full bg-purple-50 text-purple-600 text-xs font-semibold">
        Coming soon
      </div>
    </div>
  );
}
