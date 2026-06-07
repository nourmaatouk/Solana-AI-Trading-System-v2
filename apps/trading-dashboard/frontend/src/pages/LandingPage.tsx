import { useState, useEffect, Fragment } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTradingStore } from '../store';
import {
  ACTIVE_TIMEFRAME,
  MODEL_FEATURE_COUNT,
  WALK_FORWARD_FOLDS,
  DEFAULT_CONFIDENCE_GATE,
  FUSION_WEIGHT_CLASSIFICATION,
  FUSION_WEIGHT_SENTIMENT,
  FUSION_WEIGHT_SPOOFING,
  FUSION_WEIGHT_GNN,
  FUSION_WEIGHT_RL,
} from '../config';
import { useSystemWallet } from '../hooks/useSystemWallet';
import { formatMarketPair } from '../utils/symbol';
import { fetchFusion, fetchSignal } from '../services/api';
import type { AISignalData } from '../types';
import logoImg from '../assets/logo.png';

// ── Derived from config — no hard-coded marketing numbers ────────────────────
const MARKET_PAIR = formatMarketPair();
const GATE_PCT = Math.round(DEFAULT_CONFIDENCE_GATE * 100);
const MODEL_COUNT = [
  FUSION_WEIGHT_CLASSIFICATION,
  FUSION_WEIGHT_SENTIMENT,
  FUSION_WEIGHT_SPOOFING,
  FUSION_WEIGHT_GNN,
  FUSION_WEIGHT_RL,
].filter((w) => w > 0).length;

/**
 * Public marketing landing page shown at `/` before the user enters the app.
 * Self-contained and theme-independent: it always renders the light "Solvanta"
 * brand look (animated aurora + glass product cards), independent of the
 * dashboard's dark/light toggle. A "Get Started" CTA opens a chooser modal:
 * explore the dashboard (demo) or connect a wallet and trade. No real auth.
 */

const PURPLE = '#7C3AED';
const TEAL = '#0D9488';

const DILEMMAS: { emoji: string; problem: string; title: string; body: string; accent: string }[] = [
  {
    emoji: '😴',
    problem: 'You sleep',
    title: 'We monitor 24/7',
    body: 'Never miss a big opportunity or a danger sign — even at 3 AM.',
    accent: '#7C3AED',
  },
  {
    emoji: '🧠',
    problem: 'Too much information',
    title: 'We filter the noise',
    body: 'We scan price action, order-book moves, and news sentiment every cycle, keeping only what matters.',
    accent: '#2563EB',
  },
  {
    emoji: '🎭',
    problem: 'Market manipulation',
    title: 'We flag suspicious orders',
    body: 'We flag order-book activity that looks like spoofing, so it never catches you off guard.',
    accent: '#0D9488',
  },
];

/** Rotating phrase in the headline — each cycles with its own gradient. */
const PHRASES: { text: string; grad: string }[] = [
  { text: 'while you sleep', grad: 'linear-gradient(90deg,#7C3AED,#6366F1)' },
  { text: 'while you work',  grad: 'linear-gradient(90deg,#2563EB,#0D9488)' },
  { text: 'while you live',  grad: 'linear-gradient(90deg,#0D9488,#F59E0B)' },
  { text: '24/7',            grad: 'linear-gradient(90deg,#9333EA,#EC4899)' },
];

const STEPS: { color: string; title: string; body: string }[] = [
  { color: '#7C3AED', title: 'Connect Wallet', body: 'Connect your Solana wallet securely to get started.' },
  { color: '#2563EB', title: 'Configure Preferences', body: 'Choose your preferred asset, risk level, and trading mode.' },
  { color: '#3B82F6', title: 'AI Market Analysis', body: 'The AI analyzes market data, order-book activity, news, and sentiment in real time.' },
  { color: '#0D9488', title: 'Trading Decision', body: 'The system detects high-probability opportunities and generates smart trading signals.' },
  { color: '#16A34A', title: 'Automated Execution', body: 'Approved trades are executed on the Solana blockchain, after the Guardian checks.' },
];

const STATS: { value: string; label: string }[] = [
  { value: String(MODEL_COUNT), label: 'AI models fused' },
  { value: String(MODEL_FEATURE_COUNT), label: 'features per scan' },
  { value: '24/7', label: 'market monitoring' },
  { value: `${WALK_FORWARD_FOLDS}-fold`, label: 'walk-forward validated' },
];

export function LandingPage() {
  const navigate = useNavigate();
  const { connected: walletConnected, connect: connectSystemWallet } = useSystemWallet();
  const setMode = useTradingStore((s) => s.setMode);
  const [chooserOpen, setChooserOpen] = useState(false);
  const [phraseIdx, setPhraseIdx] = useState(0);
  const [signal, setSignal] = useState<AISignalData | null>(null);

  // Pull the latest live signal so the preview cards show real data, not
  // hard-coded numbers. Falls back silently (cards show the config gate) when
  // the backend/FastAPI isn't reachable from the public landing page.
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const s = await fetchFusion();
        if (alive) setSignal(s);
      } catch {
        try {
          const s = await fetchSignal();
          if (alive) setSignal(s);
        } catch {
          /* backend not running — keep null, cards show the gate */
        }
      }
    };
    load();
    return () => { alive = false; };
  }, []);

  const sigAction = (signal?.action ?? '').toUpperCase() || null;
  const sigConf = signal?.confidence ?? signal?.classification_confidence ?? null;
  const sigConfPct = sigConf != null ? Math.round(sigConf * 100) : null;
  const actionColor = sigAction === 'BUY' ? '#10B981' : sigAction === 'SELL' ? '#EF4444' : '#64748b';

  // Cycle the headline phrase: sleep → work → live → 24/7 → …
  useEffect(() => {
    const id = setInterval(() => setPhraseIdx((i) => (i + 1) % PHRASES.length), 2200);
    return () => clearInterval(id);
  }, []);

  // Connect Wallet & Trade — enters devnet using the backend Dev Wallet (system
  // keypair); no browser extension required. Navigating here (rather than via an
  // effect on the connected flag) keeps the redirect tied to this explicit
  // click, so a pre-existing Dev Wallet session never auto-bounces the landing
  // page on load.
  const startWithWallet = async () => {
    setChooserOpen(false);
    await connectSystemWallet();
    setMode('devnet');
    navigate('/execution');
  };

  // Small illustrative mockup shown under each "how to use" step.
  const stepVisual = (i: number) => {
    switch (i) {
      case 0:
        return <div className="lp-mini-wallet">👛 Connect Wallet</div>;
      case 1:
        return (
          <div className="lp-mini-prefs">
            <div className="lp-mini-row"><span>Asset</span><b>{MARKET_PAIR}</b></div>
            <div className="lp-mini-row"><span>Risk level</span><b>Medium ▾</b></div>
            <div className="lp-mini-row"><span>Trading mode</span><b>Auto ▾</b></div>
          </div>
        );
      case 2:
        return (
          <div className="lp-mini-icons">
            <span>📈</span><span>🌐</span><span>📰</span><span>💬</span>
          </div>
        );
      case 3:
        return (
          <div className="lp-mini-signal">
            <div className="lp-mini-pair">{MARKET_PAIR}</div>
            <div
              className="lp-mini-buy"
              style={{ color: actionColor, background: `${actionColor}1f` }}
            >
              {sigAction ? `${sigAction} SIGNAL` : 'AWAITING SIGNAL'}
            </div>
            <div className="lp-mini-conf">
              <div className="lp-mini-bar"><span style={{ width: `${sigConfPct ?? GATE_PCT}%` }} /></div>
              <span className="lp-mini-num">{sigConfPct != null ? `${sigConfPct}%` : `≥${GATE_PCT}%`}</span>
            </div>
          </div>
        );
      default:
        return <div className="lp-mini-exec">⚡ <b>Solana</b> · on-chain</div>;
    }
  };

  return (
    <div className="lp-root">
      {/* ── Animated background ───────────────────────────────────────────── */}
      <div className="lp-bg" aria-hidden>
        <span className="lp-blob lp-blob-1" />
        <span className="lp-blob lp-blob-2" />
        <span className="lp-blob lp-blob-3" />
        <div className="lp-grid" />
      </div>

      {/* ── Top header: logo + app name ──────────────────────────────────── */}
      <header className="lp-header">
        <div className="lp-brand" onClick={() => navigate('/dashboard')} title="Enter the dashboard">
          <img src={logoImg} alt="Solvanta AI logo" className="lp-logo" />
          <span className="lp-brand-name">Solvanta AI</span>
        </div>
        <div className="lp-live-pill">
          <span className="lp-live-dot" /> Markets live
        </div>
      </header>

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="lp-hero">
        {/* floating product cards (decorative) */}
        <div className="lp-float lp-float-signal">
          <div className="lp-fc-top">
            <span className="lp-live-dot" /> LIVE SIGNAL
          </div>
          <div className="lp-fc-pair">{MARKET_PAIR}</div>
          <div className="lp-fc-action" style={{ color: actionColor }}>{sigAction ?? '—'}</div>
          <div className="lp-fc-conf">
            <div className="lp-fc-conf-bar"><span style={{ width: `${sigConfPct ?? GATE_PCT}%` }} /></div>
            <span className="lp-fc-conf-num">{sigConfPct != null ? `${sigConfPct}%` : `≥${GATE_PCT}%`}</span>
          </div>
        </div>

        <div className="lp-float lp-float-chart">
          <div className="lp-fc-chip">{MARKET_PAIR} · {ACTIVE_TIMEFRAME}</div>
          <svg viewBox="0 0 120 48" preserveAspectRatio="none" className="lp-spark">
            <defs>
              <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={TEAL} stopOpacity="0.35" />
                <stop offset="100%" stopColor={TEAL} stopOpacity="0" />
              </linearGradient>
            </defs>
            <path d="M0,40 L18,34 L34,38 L52,24 L70,28 L88,14 L104,18 L120,6" fill="none" stroke={TEAL} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M0,40 L18,34 L34,38 L52,24 L70,28 L88,14 L104,18 L120,6 L120,48 L0,48 Z" fill="url(#sparkFill)" />
          </svg>
        </div>

        <div className="lp-float lp-float-guard">
          <span className="lp-guard-icon">🛡️</span>
          <div>
            <div className="lp-guard-title">Guardian</div>
            <div className="lp-guard-ok">Trade approved ✓</div>
          </div>
        </div>

        {/* hero content */}
        <div className="lp-badge reveal">
          <span className="lp-live-dot" />
          YOUR SMART TRADING ASSISTANT — ALWAYS ON
        </div>

        <p className="lp-welcome reveal" style={{ animationDelay: '0.05s' }}>
          👋 Welcome to <span className="lp-grad">Solvanta AI</span>
        </p>

        <h1 className="lp-h1 reveal" style={{ animationDelay: '0.1s' }}>
          <span style={{ color: '#0f172a' }}>You're the expert.</span>
          <br />
          <span style={{ color: '#334155' }}>We watch the markets</span>
          <br />
          <span className="lp-rotate-wrap" key={phraseIdx}>
            <span className="lp-grad-anim" style={{ backgroundImage: PHRASES[phraseIdx].grad }}>
              {PHRASES[phraseIdx].text}
            </span>
          </span>
          <span className="lp-cursor" aria-hidden />
        </h1>

        <p className="lp-sub reveal" style={{ animationDelay: '0.2s' }}>
          You know how to trade. But you can't watch every chart, every order book,
          and every news story — <strong>every second</strong>. Solvanta does.
        </p>

        <button className="lp-cta reveal" style={{ animationDelay: '0.3s' }} onClick={() => setChooserOpen(true)}>
          Get Started <span style={{ fontSize: 15 }}>→</span>
        </button>


        {/* stats strip */}
        <div className="lp-stats reveal" style={{ animationDelay: '0.45s' }}>
          {STATS.map((s) => (
            <div key={s.label} className="lp-stat">
              <div className="lp-stat-value">{s.value}</div>
              <div className="lp-stat-label">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── The Trader's Dilemma ─────────────────────────────────────────── */}
      <section className="lp-dilemma">
        <p className="lp-overline">THE TRADER'S DILEMMA</p>
        <h2 className="lp-h2">
          <span style={{ color: '#0f172a' }}>The market never sleeps.</span>
          <br />
          <span style={{ color: '#94a3b8' }}>But you have to.</span>
        </h2>
        <p className="lp-dilemma-sub">
          Every trader knows the frustration: the perfect setup happens while you're
          asleep, at work, or away from your screen. Solvanta AI was built for exactly this.
        </p>

        <div className="lp-cards">
          {DILEMMAS.map((d) => (
            <div key={d.title} className="lp-card" style={{ ['--accent' as string]: d.accent }}>
              <div className="lp-card-bar" />
              <div className="lp-card-emoji">{d.emoji}</div>
              <p className="lp-card-problem">{d.problem}</p>
              <h3 className="lp-card-title">{d.title}</h3>
              <p className="lp-card-body">{d.body}</p>
            </div>
          ))}
        </div>

        <div className="lp-bottom-cta">
          <button className="lp-cta" onClick={() => setChooserOpen(true)}>
            Get Started <span style={{ fontSize: 15 }}>→</span>
          </button>
        </div>
      </section>

      {/* ── How to use Solvanta AI ───────────────────────────────────────── */}
      <section className="lp-howto">
        <div className="lp-howto-titlebox">
          <h2 className="lp-howto-title">How to use <span className="lp-grad">Solvanta AI</span> ?</h2>
        </div>

        <div className="lp-steps">
          {STEPS.map((s, i) => (
            <Fragment key={s.title}>
              <div className="lp-step">
                <div className="lp-step-num" style={{ background: s.color, boxShadow: `0 8px 20px ${s.color}55` }}>
                  {i + 1}
                </div>
                <h3 className="lp-step-title" style={{ color: s.color }}>{s.title}</h3>
                <p className="lp-step-body">{s.body}</p>
                <div className="lp-step-visual">{stepVisual(i)}</div>
              </div>
              {i < STEPS.length - 1 && <div className="lp-step-arrow" aria-hidden>→</div>}
            </Fragment>
          ))}
        </div>
      </section>

      <footer className="lp-footer">
        <img src={logoImg} alt="" className="lp-footer-logo" />
        <span>Solvanta AI — your always-on trading assistant</span>
      </footer>

      {/* ── Choose-how-to-start modal ────────────────────────────────────── */}
      {chooserOpen && (
        <div className="lp-modal-overlay" onClick={() => setChooserOpen(false)}>
          <div className="lp-modal" onClick={(e) => e.stopPropagation()}>
            <button className="lp-modal-close" aria-label="Close" onClick={() => setChooserOpen(false)}>×</button>
            <h2 className="lp-modal-title">How do you want to start?</h2>
            <p className="lp-modal-sub">Explore in demo mode, or connect your wallet to execute Guardian-validated trades on devnet.</p>

            <div className="lp-modal-options">
              <button className="lp-opt" onClick={() => { setChooserOpen(false); setMode('paper'); navigate('/market'); }}>
                <div className="lp-opt-emoji">📊</div>
                <h3>Explore the Dashboard</h3>
                <p>See the live market, AI signals, and paper-trading metrics. No wallet needed.</p>
                <span className="lp-opt-link">View live market →</span>
              </button>

              <button className="lp-opt lp-opt-primary" onClick={startWithWallet}>
                <div className="lp-opt-emoji">👛</div>
                <h3>Connect Wallet &amp; Trade</h3>
                <p>Connect your Solana wallet to execute trades on Solana devnet (a test network — no real funds), each validated by the Guardian before it runs.</p>
                <span className="lp-opt-link">{walletConnected ? 'Start trading →' : 'Connect wallet →'}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Styles + animations ──────────────────────────────────────────── */}
      <style>{`
        .lp-root {
          position: relative;
          min-height: 100vh;
          width: 100%;
          overflow-x: hidden;
          overflow-y: auto;
          background: #f6f7fc;
          color: #0f172a;
          font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
        }

        /* animated background */
        .lp-bg { position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none; }
        .lp-blob { position: absolute; border-radius: 50%; filter: blur(70px); opacity: 0.5; }
        .lp-blob-1 { width: 520px; height: 520px; top: -120px; left: -80px;
          background: radial-gradient(circle, rgba(124,58,237,0.45), transparent 70%); animation: drift1 18s ease-in-out infinite; }
        .lp-blob-2 { width: 460px; height: 460px; top: 10%; right: -120px;
          background: radial-gradient(circle, rgba(13,148,136,0.45), transparent 70%); animation: drift2 22s ease-in-out infinite; }
        .lp-blob-3 { width: 420px; height: 420px; bottom: -120px; left: 30%;
          background: radial-gradient(circle, rgba(59,130,246,0.35), transparent 70%); animation: drift3 26s ease-in-out infinite; }
        .lp-grid { position: absolute; inset: 0;
          background-image: linear-gradient(rgba(15,23,42,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(15,23,42,0.04) 1px, transparent 1px);
          background-size: 46px 46px;
          mask-image: radial-gradient(ellipse 80% 60% at 50% 40%, #000 30%, transparent 75%);
          -webkit-mask-image: radial-gradient(ellipse 80% 60% at 50% 40%, #000 30%, transparent 75%); }

        /* header */
        .lp-header { position: absolute; top: 0; left: 0; right: 0; z-index: 10;
          display: flex; align-items: center; justify-content: space-between; padding: 20px 36px; }
        .lp-brand { display: flex; align-items: center; gap: 12px; cursor: pointer; }
        .lp-logo { width: 40px; height: 40px; object-fit: contain; }
        .lp-brand-name { font-size: 20px; font-weight: 800; letter-spacing: -0.01em; color: #0f172a; }
        .lp-live-pill { display: inline-flex; align-items: center; gap: 8px; padding: 7px 14px; border-radius: 999px;
          background: rgba(255,255,255,0.7); border: 1px solid #e7eaf3; backdrop-filter: blur(8px);
          font-size: 12px; font-weight: 700; color: #475569; letter-spacing: 0.04em; }
        .lp-live-dot { width: 8px; height: 8px; border-radius: 50%; background: ${TEAL};
          box-shadow: 0 0 0 0 rgba(13,148,136,0.5); animation: pulse 1.8s infinite; flex-shrink: 0; }

        /* hero */
        .lp-hero { position: relative; z-index: 1; min-height: 100vh; display: flex; flex-direction: column;
          align-items: center; justify-content: center; text-align: center; padding: 90px 24px 60px; }
        .lp-badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 18px; border-radius: 999px;
          background: rgba(124,58,237,0.08); border: 1px solid rgba(124,58,237,0.22); color: ${PURPLE};
          font-weight: 700; font-size: 13px; letter-spacing: 0.08em; margin-bottom: 28px; }
        .lp-welcome { margin: 0 0 14px; font-size: clamp(18px, 2.4vw, 26px); font-weight: 700; color: #475569; }
        .lp-grad { background: linear-gradient(90deg, ${PURPLE}, ${TEAL}); -webkit-background-clip: text;
          background-clip: text; -webkit-text-fill-color: transparent; }
        .lp-h1 { font-size: clamp(40px, 7vw, 86px); line-height: 1.02; font-weight: 800; margin: 0; letter-spacing: -0.025em; }
        .lp-rotate-wrap { display: inline-block;
          animation: swapIn 0.55s cubic-bezier(0.34, 1.56, 0.64, 1) both; }
        .lp-grad-anim { display: inline-block; background-size: 220% auto;
          -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
          animation: gradientShift 5s linear infinite; }
        .lp-cursor { display: inline-block; width: 4px; height: 0.9em; margin-left: 8px; background: ${PURPLE};
          vertical-align: middle; animation: blink 1.1s step-end infinite; }
        .lp-sub { max-width: 700px; margin: 30px auto 0; font-size: clamp(16px, 2vw, 21px); line-height: 1.6; color: #475569; }
        .lp-sub strong { color: #1e293b; }
        .lp-cta { margin-top: 38px; display: inline-flex; align-items: center; gap: 10px; padding: 17px 40px;
          border-radius: 14px; border: none; cursor: pointer; color: #fff; font-size: 18px; font-weight: 700;
          background: linear-gradient(90deg, ${PURPLE}, #9333EA);
          box-shadow: 0 14px 34px rgba(124,58,237,0.4); transition: transform 0.15s ease, box-shadow 0.15s ease; }
        .lp-cta:hover { transform: translateY(-2px); box-shadow: 0 18px 42px rgba(124,58,237,0.5); }
        .lp-fineprint { margin-top: 20px; font-size: 14px; color: #94a3b8; }

        /* stats */
        .lp-stats { display: flex; flex-wrap: wrap; justify-content: center; gap: 14px; margin-top: 48px; }
        .lp-stat { min-width: 130px; padding: 18px 22px; border-radius: 16px; background: rgba(255,255,255,0.65);
          border: 1px solid #e7eaf3; backdrop-filter: blur(8px); }
        .lp-stat-value { font-size: 30px; font-weight: 800; background: linear-gradient(90deg, ${PURPLE}, ${TEAL});
          -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
        .lp-stat-label { margin-top: 4px; font-size: 13px; color: #64748b; font-weight: 500; }

        /* floating product cards */
        .lp-float { position: absolute; z-index: 2; background: rgba(255,255,255,0.78); backdrop-filter: blur(14px);
          border: 1px solid rgba(231,234,243,0.9); border-radius: 18px; box-shadow: 0 20px 50px rgba(15,23,42,0.12); padding: 16px 18px; }
        .lp-float-signal { top: 20%; left: 6%; width: 200px; animation: floaty 7s ease-in-out infinite; }
        .lp-float-chart { top: 60%; left: 9%; width: 180px; animation: floaty 8s ease-in-out 0.5s infinite; }
        .lp-float-guard { top: 30%; right: 7%; display: flex; align-items: center; gap: 12px; animation: floaty 7.5s ease-in-out 0.3s infinite; }
        .lp-fc-top { display: flex; align-items: center; gap: 7px; font-size: 10px; font-weight: 800; letter-spacing: 0.06em; color: ${TEAL}; }
        .lp-fc-pair { margin-top: 10px; font-size: 13px; font-weight: 700; color: #64748b; }
        .lp-fc-action { font-size: 30px; font-weight: 800; color: #10B981; line-height: 1.1; }
        .lp-fc-conf { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
        .lp-fc-conf-bar { flex: 1; height: 6px; border-radius: 6px; background: #eef1f6; overflow: hidden; }
        .lp-fc-conf-bar span { display: block; height: 100%; border-radius: 6px; background: linear-gradient(90deg, ${PURPLE}, ${TEAL}); }
        .lp-fc-conf-num { font-size: 12px; font-weight: 800; color: #475569; }
        .lp-fc-chip { display: inline-block; font-size: 11px; font-weight: 800; color: ${TEAL}; background: rgba(13,148,136,0.1);
          padding: 3px 9px; border-radius: 999px; margin-bottom: 8px; }
        .lp-spark { width: 100%; height: 48px; display: block; }
        .lp-guard-icon { font-size: 26px; }
        .lp-guard-title { font-size: 12px; font-weight: 700; color: #64748b; }
        .lp-guard-ok { font-size: 14px; font-weight: 800; color: #10B981; }

        /* dilemma */
        .lp-dilemma { position: relative; z-index: 1; padding: 90px 24px; max-width: 1180px; margin: 0 auto; }
        .lp-overline { text-align: center; color: #F59E0B; font-weight: 800; letter-spacing: 0.14em; font-size: 13px; margin: 0; }
        .lp-h2 { text-align: center; margin: 14px 0 0; font-size: clamp(30px, 4.5vw, 52px); font-weight: 800; letter-spacing: -0.02em; }
        .lp-dilemma-sub { text-align: center; max-width: 720px; margin: 22px auto 0; font-size: 18px; line-height: 1.6; color: #475569; }
        .lp-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; margin-top: 56px; }
        .lp-card { position: relative; overflow: hidden; background: #fff; border-radius: 20px; padding: 34px 30px 30px;
          box-shadow: 0 10px 30px rgba(15,23,42,0.06); border: 1px solid #eef1f6;
          transition: transform 0.18s ease, box-shadow 0.18s ease; }
        .lp-card:hover { transform: translateY(-6px); box-shadow: 0 24px 50px rgba(15,23,42,0.12); }
        .lp-card-bar { position: absolute; top: 0; left: 0; right: 0; height: 4px;
          background: linear-gradient(90deg, var(--accent), transparent); }
        .lp-card-emoji { font-size: 34px; margin-bottom: 24px; }
        .lp-card-problem { color: #EF4444; text-decoration: line-through; font-weight: 600; margin: 0 0 8px; font-size: 15px; }
        .lp-card-title { color: #10B981; font-weight: 800; font-size: 22px; margin: 0 0 14px; }
        .lp-card-body { color: #64748b; line-height: 1.6; margin: 0; font-size: 15.5px; }
        .lp-bottom-cta { text-align: center; margin-top: 66px; }

        /* how to use */
        .lp-howto { position: relative; z-index: 1; padding: 40px 24px 80px; max-width: 1240px; margin: 0 auto; }
        .lp-howto-titlebox { display: inline-block; border: 2px solid ${PURPLE}; border-radius: 16px;
          padding: 10px 26px; margin: 0 auto 50px; }
        .lp-howto { text-align: center; }
        .lp-howto-title { margin: 0; font-size: clamp(26px, 4vw, 44px); font-weight: 800; color: #1e3a8a; letter-spacing: -0.01em; }
        .lp-steps { display: flex; align-items: stretch; justify-content: center; gap: 8px; text-align: center; }
        .lp-step { position: relative; flex: 1; min-width: 0; background: #fff; border: 1px solid #eef1f6; border-radius: 18px;
          padding: 40px 18px 22px; margin-top: 26px; box-shadow: 0 10px 28px rgba(15,23,42,0.05);
          display: flex; flex-direction: column; align-items: center;
          transition: transform 0.18s ease, box-shadow 0.18s ease; }
        .lp-step:hover { transform: translateY(-5px); box-shadow: 0 22px 44px rgba(15,23,42,0.1); }
        .lp-step-num { position: absolute; top: -26px; left: 50%; transform: translateX(-50%);
          width: 52px; height: 52px; border-radius: 50%; color: #fff; font-size: 24px; font-weight: 800;
          display: flex; align-items: center; justify-content: center; border: 4px solid #fff; }
        .lp-step-title { margin: 6px 0 10px; font-size: 17px; font-weight: 800; }
        .lp-step-body { margin: 0 0 18px; font-size: 13.5px; line-height: 1.5; color: #64748b; }
        .lp-step-visual { margin-top: auto; width: 100%; display: flex; justify-content: center; }
        .lp-step-arrow { display: flex; align-items: center; color: ${PURPLE}; font-size: 26px; font-weight: 800; flex: 0 0 auto; }

        /* step mini-mockups */
        .lp-mini-wallet { background: ${PURPLE}; color: #fff; font-size: 12px; font-weight: 700; padding: 9px 14px; border-radius: 10px; }
        .lp-mini-prefs { width: 100%; background: #f8f9fd; border: 1px solid #e7eaf3; border-radius: 12px; padding: 12px; }
        .lp-mini-row { display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; color: #64748b; padding: 5px 0; }
        .lp-mini-row b { color: #1e293b; }
        .lp-mini-icons { display: flex; gap: 12px; font-size: 24px; padding: 8px 0; }
        .lp-mini-signal { width: 100%; background: #f8f9fd; border: 1px solid #e7eaf3; border-radius: 12px; padding: 12px; text-align: left; }
        .lp-mini-pair { font-size: 12px; font-weight: 700; color: #64748b; }
        .lp-mini-buy { font-size: 13px; font-weight: 800; color: #10B981; background: rgba(16,185,129,0.12); display: inline-block; padding: 2px 8px; border-radius: 6px; margin: 6px 0 10px; }
        .lp-mini-conf { display: flex; align-items: center; gap: 8px; }
        .lp-mini-bar { flex: 1; height: 6px; border-radius: 6px; background: #e7eaf3; overflow: hidden; }
        .lp-mini-bar span { display: block; height: 100%; border-radius: 6px; background: linear-gradient(90deg, ${PURPLE}, ${TEAL}); }
        .lp-mini-num { font-size: 12px; font-weight: 800; color: #475569; }
        .lp-mini-exec { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; font-weight: 700; color: #1e293b;
          background: linear-gradient(160deg, rgba(124,58,237,0.08), rgba(13,148,136,0.08)); padding: 10px 16px; border-radius: 12px; }

        @media (max-width: 980px) {
          .lp-steps { flex-wrap: wrap; gap: 36px 8px; }
          .lp-step { flex: 1 1 260px; }
          .lp-step-arrow { display: none; }
        }

        /* footer */
        .lp-footer { position: relative; z-index: 1; display: flex; align-items: center; justify-content: center; gap: 10px;
          padding: 40px 24px 56px; color: #94a3b8; font-size: 14px; }
        .lp-footer-logo { width: 22px; height: 22px; object-fit: contain; opacity: 0.8; }

        /* modal */
        .lp-modal-overlay { position: fixed; inset: 0; z-index: 50; background: rgba(15,23,42,0.45); backdrop-filter: blur(5px);
          display: flex; align-items: center; justify-content: center; padding: 24px; animation: fadeIn 0.18s ease-out; }
        .lp-modal { position: relative; width: 100%; max-width: 760px; background: #fff; border-radius: 24px;
          padding: 42px 36px; box-shadow: 0 30px 80px rgba(15,23,42,0.3); animation: popIn 0.2s ease-out; }
        .lp-modal-close { position: absolute; top: 18px; right: 18px; width: 34px; height: 34px; border-radius: 50%;
          border: none; cursor: pointer; background: #f1f5f9; color: #64748b; font-size: 18px; line-height: 1; }
        .lp-modal-title { text-align: center; margin: 0; font-size: 28px; font-weight: 800; color: #0f172a; }
        .lp-modal-sub { text-align: center; margin: 10px 0 0; color: #64748b; font-size: 15px; }
        .lp-modal-options { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-top: 30px; }
        .lp-opt { text-align: left; cursor: pointer; background: #f8f9fd; border: 1px solid #e7eaf3; border-radius: 18px;
          padding: 26px 24px; transition: transform 0.12s ease, box-shadow 0.12s ease; }
        .lp-opt:hover { transform: translateY(-3px); box-shadow: 0 14px 30px rgba(124,58,237,0.16); }
        .lp-opt-primary { background: linear-gradient(160deg, rgba(124,58,237,0.06), rgba(13,148,136,0.06)); border-color: ${PURPLE}; }
        .lp-opt-emoji { font-size: 34px; margin-bottom: 14px; }
        .lp-opt h3 { margin: 0 0 8px; font-size: 19px; font-weight: 800; color: #0f172a; }
        .lp-opt p { margin: 0; color: #64748b; font-size: 14.5px; line-height: 1.55; }
        .lp-opt-link { display: inline-block; margin-top: 16px; color: ${PURPLE}; font-weight: 700; font-size: 14px; }

        /* entrance + keyframes */
        .reveal { opacity: 0; animation: fadeUp 0.7s cubic-bezier(0.22, 1, 0.36, 1) forwards; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes blink { 0%, 100% { opacity: 1 } 50% { opacity: 0 } }
        @keyframes floaty { 0%, 100% { transform: translateY(0) } 50% { transform: translateY(-14px) } }
        @keyframes gradientShift { to { background-position: 250% center; } }
        @keyframes swapIn { 0% { opacity: 0; transform: translateY(18px) scale(0.88); } 100% { opacity: 1; transform: translateY(0) scale(1); } }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(13,148,136,0.5); } 70% { box-shadow: 0 0 0 7px rgba(13,148,136,0); } 100% { box-shadow: 0 0 0 0 rgba(13,148,136,0); } }
        @keyframes drift1 { 0%, 100% { transform: translate(0,0) scale(1); } 50% { transform: translate(60px, 40px) scale(1.1); } }
        @keyframes drift2 { 0%, 100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-50px, 60px) scale(1.08); } }
        @keyframes drift3 { 0%, 100% { transform: translate(0,0) scale(1); } 50% { transform: translate(40px, -50px) scale(1.12); } }
        @keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }
        @keyframes popIn { from { opacity: 0; transform: scale(0.96) translateY(10px); } to { opacity: 1; transform: scale(1) translateY(0); } }

        /* hide floating cards on narrow screens to avoid overlap */
        @media (max-width: 1080px) { .lp-float { display: none; } }
        @media (prefers-reduced-motion: reduce) {
          .reveal, .lp-blob, .lp-float, .lp-grad-anim, .lp-rotate-wrap, .lp-live-dot { animation: none !important; opacity: 1 !important; transform: none !important; }
        }
      `}</style>
    </div>
  );
}
