import { useEffect, useState } from 'react';
import { fetchNews, fetchSentiment, type NewsArticle } from '../services/api';
import { FUSION_WEIGHT_SENTIMENT } from '../config';

const card = { background: 'var(--bg-panel)', border: '1px solid var(--border)' } as React.CSSProperties;
const wpct = (w: number) => `${Math.round(w * 100)}%`;

function SentPill({ label }: { label: string | null }) {
  const u = (label ?? '').toUpperCase();
  const map: Record<string, { c: string; b: string; bd: string }> = {
    POSITIVE: { c: 'var(--green)',  b: 'var(--green-bg)',  bd: 'var(--green-bd)'  },
    BULLISH:  { c: 'var(--green)',  b: 'var(--green-bg)',  bd: 'var(--green-bd)'  },
    NEGATIVE: { c: 'var(--red)',    b: 'var(--red-bg)',    bd: 'var(--red-bd)'    },
    BEARISH:  { c: 'var(--red)',    b: 'var(--red-bg)',    bd: 'var(--red-bd)'    },
    NEUTRAL:  { c: 'var(--text-2)', b: 'var(--bg-sub)',    bd: 'var(--border)'    },
  };
  const s = map[u] ?? map.NEUTRAL;
  return (
    <span className="px-2 py-0.5 rounded-full text-xs font-semibold border whitespace-nowrap"
      style={{ color: s.c, backgroundColor: s.b, borderColor: s.bd }}>
      {u || 'UNKNOWN'}
    </span>
  );
}

function fmtDate(ts: string | null): string {
  if (!ts) return '—';
  const d = new Date(ts);
  return isNaN(d.getTime()) ? ts : d.toLocaleString();
}

export function SentimentNews() {
  const [articles, setArticles] = useState<NewsArticle[] | null>(null);
  const [agg, setAgg] = useState<{ label: string; score: number | null } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true);
      setError(null);
      const [news, sent] = await Promise.allSettled([fetchNews(30), fetchSentiment()]);
      if (!alive) return;
      if (news.status === 'fulfilled') setArticles(news.value.articles);
      else setError('Could not load news from the backend. Is the API running?');
      if (sent.status === 'fulfilled') setAgg({ label: sent.value.label, score: sent.value.score });
      setLoading(false);
    })();
    return () => { alive = false; };
  }, []);

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-5">
      <div>
        <h1 className="text-xl font-bold" style={{ color: 'var(--text-1)' }}>Sentiment &amp; News</h1>
        <p className="text-sm mt-1" style={{ color: 'var(--text-3)' }}>
          Financial headlines from GDELT and RSS feeds, scored by FinBERT. This feed drives the
          sentiment component of the fused trading signal.
        </p>
      </div>

      {/* Contribution-to-signal explainer (real weight from config) */}
      <div className="rounded-xl p-5 flex flex-wrap items-center gap-x-6 gap-y-3" style={card}>
        <div>
          <div className="text-xs uppercase tracking-widest" style={{ color: 'var(--text-3)' }}>Weight in fusion</div>
          <div className="text-2xl font-bold" style={{ color: 'var(--accent)' }}>{wpct(FUSION_WEIGHT_SENTIMENT)}</div>
        </div>
        <div className="h-8 w-px" style={{ background: 'var(--border)' }} />
        <div>
          <div className="text-xs uppercase tracking-widest mb-1" style={{ color: 'var(--text-3)' }}>Current aggregate sentiment</div>
          {agg ? (
            <div className="flex items-center gap-2">
              <SentPill label={agg.label} />
              <span className="text-sm font-mono" style={{ color: 'var(--text-2)' }}>
                {agg.score != null ? agg.score.toFixed(3) : 'n/a'}
              </span>
            </div>
          ) : (
            <span className="text-sm" style={{ color: 'var(--text-3)' }}>{loading ? 'loading…' : 'unavailable'}</span>
          )}
        </div>
        <p className="text-xs flex-1 min-w-[220px]" style={{ color: 'var(--text-3)' }}>
          FinBERT classifies each headline as positive / negative / neutral. The relevance‑weighted
          average feeds the fusion layer at {wpct(FUSION_WEIGHT_SENTIMENT)} of the final score; a
          high‑impact event (hack, outage, listing) can additionally shift the decision.
        </p>
      </div>

      {/* News list with loading / error / empty states */}
      <div className="rounded-xl p-5" style={card}>
        <div className="text-xs font-semibold tracking-widest uppercase mb-4" style={{ color: 'var(--text-3)' }}>
          Recent headlines {articles && `(${articles.length})`}
        </div>

        {loading && (
          <div className="space-y-3">
            {[0, 1, 2].map((i) => (
              <div key={i} className="h-14 rounded-lg animate-pulse" style={{ background: 'var(--bg-sub)' }} />
            ))}
          </div>
        )}

        {!loading && error && (
          <div className="text-sm px-4 py-3 rounded-lg"
            style={{ color: 'var(--red)', background: 'var(--red-bg)', border: '1px solid var(--red-bd)' }}>
            {error}
          </div>
        )}

        {!loading && !error && articles && articles.length === 0 && (
          <div className="text-sm text-center py-8" style={{ color: 'var(--text-3)' }}>
            No news yet. The sentiment daemon will populate this feed on its next fetch.
          </div>
        )}

        {!loading && !error && articles && articles.length > 0 && (
          <div className="space-y-2">
            {articles.map((a, i) => (
              <div key={`${a.url ?? a.title}-${i}`} className="rounded-lg p-3 flex flex-col gap-1.5"
                style={{ background: 'var(--bg-sub)', border: '1px solid var(--border)' }}>
                <div className="flex items-start justify-between gap-3">
                  {a.url ? (
                    <a href={a.url} target="_blank" rel="noreferrer"
                      className="text-sm font-medium hover:underline" style={{ color: 'var(--text-1)' }}>
                      {a.title ?? '(untitled)'}
                    </a>
                  ) : (
                    <span className="text-sm font-medium" style={{ color: 'var(--text-1)' }}>{a.title ?? '(untitled)'}</span>
                  )}
                  <SentPill label={a.sentiment_label} />
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs" style={{ color: 'var(--text-3)' }}>
                  {a.feed_source && (
                    <span className="px-1.5 py-0.5 rounded font-semibold"
                      style={{ background: 'var(--accent-bg)', color: 'var(--accent)' }}>{a.feed_source}</span>
                  )}
                  {a.source && <span>{a.source}</span>}
                  <span>· {fmtDate(a.timestamp)}</span>
                  {a.sentiment_score != null && <span>· score {a.sentiment_score.toFixed(3)}</span>}
                  {a.event_type && a.event_type !== 'none' && <span>· event: {a.event_type}</span>}
                  {a.high_impact_event && (
                    <span className="px-1.5 py-0.5 rounded font-semibold"
                      style={{ background: 'var(--red-bg)', color: 'var(--red)' }}>HIGH IMPACT</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
