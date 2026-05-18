function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl p-5" style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
      <div className="font-semibold text-sm mb-3" style={{ color: 'var(--text-1)' }}>{title}</div>
      {children}
    </div>
  );
}

function Step({ n, title, desc }: { n: number; title: string; desc: string }) {
  return (
    <div className="flex gap-3">
      <div className="w-6 h-6 rounded-full text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5"
        style={{ background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--accent-bd)' }}>
        {n}
      </div>
      <div>
        <div className="text-sm font-medium" style={{ color: 'var(--text-1)' }}>{title}</div>
        <div className="text-xs mt-0.5 leading-relaxed" style={{ color: 'var(--text-3)' }}>{desc}</div>
      </div>
    </div>
  );
}

function Model({ name, weight, color, desc }: { name: string; weight: string; color: string; desc: string }) {
  return (
    <div className="flex gap-3 items-start">
      <div className="px-2 py-0.5 rounded text-[10px] font-bold flex-shrink-0 mt-0.5"
        style={{ backgroundColor: color + '20', color, border: `1px solid ${color}40` }}>
        {weight}
      </div>
      <div>
        <div className="text-sm font-medium" style={{ color: 'var(--text-1)' }}>{name}</div>
        <div className="text-xs mt-0.5 leading-relaxed" style={{ color: 'var(--text-3)' }}>{desc}</div>
      </div>
    </div>
  );
}

import { DEFAULT_CONFIDENCE_GATE } from '../config';

const GATE_PCT = Math.round(DEFAULT_CONFIDENCE_GATE * 100);

export function HowItWorks() {
  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: 'var(--bg-page)' }}>
      <div className="max-w-2xl mx-auto space-y-4">
        <div>
          <h1 className="text-xl font-bold mb-1" style={{ color: 'var(--text-1)' }}>How It Works</h1>
          <p className="text-sm" style={{ color: 'var(--text-3)' }}>Architecture and data flow of the Solvanta AI Trading System.</p>
        </div>

        <Section title="System Overview">
          <p className="text-sm leading-relaxed" style={{ color: 'var(--text-2)' }}>
            Solvanta is an AI-powered predictive trading assistant for SOL/USDT. It fuses four independent
            models — a machine learning classifier, a reinforcement learning agent, an NLP sentiment model,
            and a graph neural network — to generate BUY / SELL / HOLD signals every 15 minutes.
            All execution is simulated (paper trading).
          </p>
        </Section>

        <Section title="Data Pipeline (every 15 minutes)">
          <div className="space-y-3">
            <Step n={1} title="Market data collection" desc="Fetches the latest OHLCV candles (15-min) for SOL/USDT from Binance REST API. BTC/USDT data is collected simultaneously for correlation features." />
            <Step n={2} title="Technical feature engineering" desc="Computes 49 features: RSI, MACD, EMA, ATR, Bollinger Bands, Hawkes intensity (market microstructure), and BTC correlation. Designed to be leak-free across the training/test boundary." />
            <Step n={3} title="Order book collection" desc="A separate daemon streams the Binance depth20 WebSocket continuously, saving snapshots used as real-time order book features." />
            <Step n={4} title="News sentiment" desc="FinBERT processes the latest crypto news headlines. The sentiment score (POSITIVE / NEUTRAL / NEGATIVE) and relevance weight are computed and merged into the feature vector." />
            <Step n={5} title="XGBoost inference" desc={`The trained XGBoost model outputs P(UP) and P(DOWN). The ${GATE_PCT}% confidence gate requires one direction to exceed ${DEFAULT_CONFIDENCE_GATE} to produce a non-HOLD signal.`} />
            <Step n={6} title="PPO RL inference" desc="A Proximal Policy Optimization agent trained on historical episodes observes the same 49 features plus portfolio state. Outputs a directional score blended into the final fused signal at 30% weight." />
            <Step n={7} title="Spoofing detection" desc="TEN-GNN (Temporal Event Network GNN) analyzes order book graph structure for abnormal bid/ask patterns, large walls, and imbalance. Output: spoofing_confidence ∈ [0, 1]." />
            <Step n={8} title="Fusion" desc="Base fusion: XGBoost 50% + FinBERT 25% + TEN-GNN 20% + GNN 5%. The PPO RL score is then blended in at 30% weight. If spoofing > 0.7, BUY signals are vetoed regardless." />
            <Step n={9} title="Paper trade execution" desc="The fusion decision is logged and a simulated trade is recorded. Equity is tracked from a normalized baseline of 1.0." />
          </div>
        </Section>

        <Section title="AI Models">
          <div className="space-y-4">
            <Model name="XGBoost Classifier" weight="50%" color="#8B5CF6"
              desc={`Gradient-boosted tree trained on walk-forward validated historical data. Input: 49 technical and microstructure features. Output: probability of price going UP or DOWN in the next 15-minute candle. Confidence gate at ${GATE_PCT}% prevents low-conviction signals.`} />
            <Model name="PPO RL Agent" weight="30% blend" color="#F59E0B"
              desc="Proximal Policy Optimization agent trained with a custom TradingEnv on 188k historical 15-min bars. Observation: 49 market features + 3 portfolio state variables. Action space: HOLD / BUY / SELL. Score blended into the base fused result." />
            <Model name="FinBERT Sentiment" weight="25%" color="#14B8A6"
              desc="BERT-based NLP model fine-tuned on financial text. Processes news headlines from crypto RSS feeds. Outputs POSITIVE / NEUTRAL / NEGATIVE label with a relevance score. Reduces BUY confidence when sentiment is strongly negative." />
            <Model name="TEN-GNN Spoofing Detector" weight="20%" color="#f6465d"
              desc="Temporal Event Network GNN that models the order book as a graph. Detects abnormal patterns: large liquidity walls, rapid imbalance shifts, and coordinated bid/ask pressure. Confidence > 0.7 vetoes all BUY signals." />
            <Model name="GNN Structural" weight="5%" color="#0ecb81"
              desc="Graph neural network encoding the structural properties of the order book (depth, spread, level distribution). Provides a low-weight sanity check on the overall market microstructure." />
          </div>
        </Section>

        <Section title="Real-Time Data (WebSocket)">
          <div className="text-xs space-y-2" style={{ color: 'var(--text-2)' }}>
            {[
              ['Price ticker',     'solusdt@ticker',           'Live price, 24h change, volume — updated every 100ms'],
              ['Candlestick chart','solusdt@kline_{interval}', 'Real-time OHLCV candles — REST backfill of 500 historical candles on connect'],
              ['Order book',       'solusdt@depth20@100ms',    'Top 20 bid/ask levels — refreshed every 100ms'],
              ['Trade stream',     'solusdt@trade',            'Every executed trade with buyer/seller side'],
              ['Watchlist',        '{symbol}@ticker × 6',      'Live prices for SOL, BTC, ETH, BNB, ADA, XRP'],
            ].map(([name, stream, desc]) => (
              <div key={name} className="pb-2 last:pb-0" style={{ borderBottom: '1px solid var(--border)' }}>
                <div className="flex items-center gap-2">
                  <span className="font-medium" style={{ color: 'var(--text-1)' }}>{name}</span>
                  <code className="text-[10px] px-1.5 py-0.5 rounded font-mono" style={{ background: 'var(--bg-sub)', color: 'var(--text-2)' }}>{stream}</code>
                </div>
                <div className="mt-0.5" style={{ color: 'var(--text-3)' }}>{desc}</div>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Infrastructure">
          <div className="grid grid-cols-2 gap-3 text-xs">
            {[
              ['Frontend',         'React 18 + TypeScript + Vite + Tailwind CSS'],
              ['Charts',           'lightweight-charts v4 (Binance-style candlesticks)'],
              ['State',            'Zustand (zero-selector re-render optimization)'],
              ['Backend',          'Node.js 20 + Express + WebSocket server'],
              ['AI API',           'FastAPI (Python) on port 8000'],
              ['Process manager',  'PM2 — 6 daemons always running'],
              ['ML training',      'XGBoost + PPO (stable-baselines3) + walk-forward validation'],
              ['Data exchange',    'JSON files + REST polling (15s signal, 30s scheduler)'],
            ].map(([name, value]) => (
              <div key={name} className="rounded-lg p-3" style={{ background: 'var(--bg-sub)' }}>
                <div className="mb-0.5" style={{ color: 'var(--text-3)' }}>{name}</div>
                <div className="font-medium" style={{ color: 'var(--text-1)' }}>{value}</div>
              </div>
            ))}
          </div>
        </Section>
      </div>
    </div>
  );
}
