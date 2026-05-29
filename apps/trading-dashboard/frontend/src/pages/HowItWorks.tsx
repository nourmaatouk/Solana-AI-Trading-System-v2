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

import {
  ACTIVE_SYMBOL,
  ACTIVE_TIMEFRAME,
  CORRELATION_SYMBOL,
  DASHBOARD_POLL_INTERVAL_MS,
  DEFAULT_CONFIDENCE_GATE,
  FUSION_WEIGHT_CLASSIFICATION,
  FUSION_WEIGHT_GNN,
  FUSION_WEIGHT_RL,
  FUSION_WEIGHT_SENTIMENT,
  FUSION_WEIGHT_SPOOFING,
  GUARDIAN_SPOOFING_THRESHOLD,
  MODEL_FEATURE_COUNT,
  SIGNAL_POLL_INTERVAL_MS,
  SPOOFING_HIGH,
} from '../config';
import { formatMarketPair } from '../utils/symbol';

const GATE_PCT = Math.round(DEFAULT_CONFIDENCE_GATE * 100);
const MARKET_PAIR = formatMarketPair(ACTIVE_SYMBOL);
const CORRELATION_PAIR = formatMarketPair(CORRELATION_SYMBOL);
const pct = (value: number) => `${Math.round(value * 100)}%`;
const rlWeightLabel = FUSION_WEIGHT_RL > 0 ? `${pct(FUSION_WEIGHT_RL)} blend` : 'disabled';
const rlFusionDesc = FUSION_WEIGHT_RL > 0
  ? `The PPO RL score is blended in at ${pct(FUSION_WEIGHT_RL)} weight.`
  : 'PPO is visible as a side signal, but its fusion weight is currently disabled.';

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
            Solvanta is an AI-powered predictive trading assistant for {MARKET_PAIR}. It fuses independent
            models — a machine learning classifier, a reinforcement learning agent, an NLP sentiment model,
            and a graph neural network — to generate BUY / SELL / HOLD signals on the configured {ACTIVE_TIMEFRAME} cadence.
            Signals feed a paper-trading simulator by default; a Guardian-gated executor can place real
            devnet / mainnet trades once a wallet is connected.
          </p>
        </Section>

        <Section title={`Data Pipeline (${ACTIVE_TIMEFRAME} cadence)`}>
          <div className="space-y-3">
            <Step n={1} title="Market data collection" desc={`Fetches the latest OHLCV candles (${ACTIVE_TIMEFRAME}) for ${MARKET_PAIR} from Binance REST API. ${CORRELATION_PAIR} data is collected simultaneously for correlation features.`} />
            <Step n={2} title="Technical feature engineering" desc={`Computes ${MODEL_FEATURE_COUNT} features: RSI, MACD, EMA, ATR, Bollinger Bands, Hawkes intensity (market microstructure), and ${CORRELATION_PAIR} correlation. Designed to be leak-free across the training/test boundary.`} />
            <Step n={3} title="Order book collection" desc="A separate daemon streams the Binance depth20 WebSocket continuously, saving snapshots used as real-time order book features." />
            <Step n={4} title="News sentiment" desc="FinBERT processes the latest crypto news headlines. The sentiment score (POSITIVE / NEUTRAL / NEGATIVE) and relevance weight are computed and merged into the feature vector." />
            <Step n={5} title="XGBoost inference" desc={`The trained XGBoost model outputs P(UP) and P(DOWN). The ${GATE_PCT}% confidence gate requires one direction to exceed ${DEFAULT_CONFIDENCE_GATE} to produce a non-HOLD signal.`} />
            <Step n={6} title="PPO RL inference" desc={`A Proximal Policy Optimization agent trained on historical episodes observes the same 49 features plus portfolio state. Current fusion weight: ${rlWeightLabel}.`} />
            <Step n={7} title="Spoofing detection" desc="TEN-GNN (Temporal Event Network GNN) analyzes order book graph structure for abnormal bid/ask patterns, large walls, and imbalance. Output: spoofing_confidence ∈ [0, 1]." />
            <Step n={8} title="Fusion + Guardian" desc={`Base fusion: XGBoost ${pct(FUSION_WEIGHT_CLASSIFICATION)} + FinBERT ${pct(FUSION_WEIGHT_SENTIMENT)} + TEN-GNN ${pct(FUSION_WEIGHT_SPOOFING)} + GNN ${pct(FUSION_WEIGHT_GNN)}. ${rlFusionDesc} Guardian blocks entries when spoofing exceeds ${pct(GUARDIAN_SPOOFING_THRESHOLD)}.`} />
            <Step n={9} title="Paper trade execution" desc="The fusion decision is logged and a simulated trade is recorded. Equity is tracked from a normalized baseline of 1.0." />
          </div>
        </Section>

        <Section title="AI Models">
          <div className="space-y-4">
            <Model name="XGBoost Classifier" weight={pct(FUSION_WEIGHT_CLASSIFICATION)} color="#8B5CF6"
              desc={`Gradient-boosted tree trained on walk-forward validated historical data. Input: ${MODEL_FEATURE_COUNT} technical and microstructure features. Output: probability of price going UP or DOWN in the next ${ACTIVE_TIMEFRAME} candle. Confidence gate at ${GATE_PCT}% prevents low-conviction signals.`} />
            <Model name="PPO RL Agent" weight={rlWeightLabel} color="#F59E0B"
              desc="Proximal Policy Optimization agent trained with a custom TradingEnv on historical bars. Action space: HOLD / BUY / SELL. It remains a side signal while RL fusion weight is disabled." />
            <Model name="FinBERT Sentiment" weight={pct(FUSION_WEIGHT_SENTIMENT)} color="#14B8A6"
              desc="BERT-based NLP model fine-tuned on financial text. Processes crypto news from GDELT and RSS feeds. Outputs POSITIVE / NEUTRAL / NEGATIVE label with a relevance score. Reduces BUY confidence when sentiment is strongly negative." />
            <Model name="TEN-GNN Spoofing Detector" weight={pct(FUSION_WEIGHT_SPOOFING)} color="#f6465d"
              desc={`Temporal Event Network GNN that models the order book as a graph. Confidence above ${pct(SPOOFING_HIGH)} is highlighted as suspicious, and Guardian blocks entries above ${pct(GUARDIAN_SPOOFING_THRESHOLD)}.`} />
            <Model name="GNN Structural" weight={pct(FUSION_WEIGHT_GNN)} color="#0ecb81"
              desc="Graph neural network encoding the structural properties of the order book (depth, spread, level distribution). Provides a low-weight sanity check on the overall market microstructure." />
          </div>
        </Section>

        <Section title="Real-Time Data (WebSocket)">
          <div className="text-xs space-y-2" style={{ color: 'var(--text-2)' }}>
            {[
              ['Price ticker',     `${ACTIVE_SYMBOL.toLowerCase()}@ticker`,           'Live price, 24h change, volume'],
              ['Candlestick chart',`${ACTIVE_SYMBOL.toLowerCase()}@kline_{interval}`, 'Real-time OHLCV candles with REST backfill on connect'],
              ['Order book',       `${ACTIVE_SYMBOL.toLowerCase()}@depth`,            'Top bid/ask levels'],
              ['Trade stream',     `${ACTIVE_SYMBOL.toLowerCase()}@trade`,            'Every executed trade with buyer/seller side'],
              ['Watchlist',        '{symbol}@ticker',      'Configured watchlist symbols'],
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
              ['Backend',          'Node.js 24 + Express + WebSocket server'],
              ['AI API',           'FastAPI (Python) on port 8000'],
              ['Process manager',  'PM2 — 6 daemons always running'],
              ['ML training',      'XGBoost + PPO (stable-baselines3) + walk-forward validation'],
              ['Data exchange',    `JSON files + REST polling (${Math.round(SIGNAL_POLL_INTERVAL_MS / 1000)}s signal, ${Math.round(DASHBOARD_POLL_INTERVAL_MS / 1000)}s dashboard)`],
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
