function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <div className="font-semibold text-gray-800 text-sm mb-3">{title}</div>
      {children}
    </div>
  );
}

function Step({ n, title, desc }: { n: number; title: string; desc: string }) {
  return (
    <div className="flex gap-3">
      <div className="w-6 h-6 rounded-full bg-indigo-100 text-indigo-600 text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">{n}</div>
      <div>
        <div className="text-sm font-medium text-gray-700">{title}</div>
        <div className="text-xs text-gray-400 mt-0.5 leading-relaxed">{desc}</div>
      </div>
    </div>
  );
}

function Model({ name, weight, color, desc }: { name: string; weight: string; color: string; desc: string }) {
  return (
    <div className="flex gap-3 items-start">
      <div className="px-2 py-0.5 rounded text-[10px] font-bold flex-shrink-0 mt-0.5" style={{ backgroundColor: color + '20', color, border: `1px solid ${color}40` }}>
        {weight}
      </div>
      <div>
        <div className="text-sm font-medium text-gray-700">{name}</div>
        <div className="text-xs text-gray-400 mt-0.5 leading-relaxed">{desc}</div>
      </div>
    </div>
  );
}

export function HowItWorks() {
  return (
    <div className="flex-1 overflow-y-auto p-6" style={{ background: '#f8f9fd' }}>
      <div className="max-w-2xl mx-auto space-y-4">
        <div>
          <h1 className="text-xl font-bold text-gray-800 mb-1">How It Works</h1>
          <p className="text-gray-400 text-sm">Architecture and data flow of the Solvanta AI Trading System.</p>
        </div>

        <Section title="System Overview">
          <p className="text-sm text-gray-600 leading-relaxed">
            Solvanta is an AI-powered predictive trading assistant for SOL/USDT. It fuses three independent
            models — a machine learning classifier, an NLP sentiment model, and a graph neural network — to
            generate BUY / SELL / HOLD signals every 15 minutes. All execution is simulated (paper trading).
          </p>
        </Section>

        <Section title="Data Pipeline (every 15 minutes)">
          <div className="space-y-3">
            <Step n={1} title="Market data collection" desc="Fetches the latest OHLCV candles (15-min) for SOL/USDT from Binance REST API. BTC/USDT data is collected simultaneously for correlation features." />
            <Step n={2} title="Technical feature engineering" desc="Computes 43 features: RSI, MACD, EMA, ATR, Bollinger Bands, Hawkes intensity (market microstructure), and BTC correlation. Designed to be leak-free across the training/test boundary." />
            <Step n={3} title="Order book collection" desc="A separate daemon streams the Binance depth20 WebSocket continuously, saving snapshots used as real-time order book features." />
            <Step n={4} title="News sentiment" desc="FinBERT processes the latest crypto news headlines. The sentiment score (POSITIVE / NEUTRAL / NEGATIVE) and relevance weight are computed and merged into the feature vector." />
            <Step n={5} title="XGBoost inference" desc="The trained XGBoost model outputs P(UP) and P(DOWN). The 69% confidence gate requires one direction to exceed 0.69 to produce a non-HOLD signal." />
            <Step n={6} title="Spoofing detection" desc="TEN-GNN (Temporal Event Network GNN) analyzes order book graph structure for abnormal bid/ask patterns, large walls, and imbalance. Output: spoofing_confidence ∈ [0, 1]." />
            <Step n={7} title="Fusion" desc="Weighted fusion: XGBoost 50% + FinBERT 25% + TEN-GNN penalty 20% + GNN structural 5%. If spoofing > 0.7, BUY signals are vetoed regardless of XGBoost output." />
            <Step n={8} title="Paper trade execution" desc="The fusion decision is logged and a simulated trade is recorded. Equity is tracked from a normalized baseline of 1.0." />
          </div>
        </Section>

        <Section title="AI Models">
          <div className="space-y-4">
            <Model name="XGBoost Classifier" weight="50%" color="#6366f1"
              desc="Gradient-boosted tree trained on walk-forward validated historical data. Input: 43 technical and microstructure features. Output: probability of price going UP or DOWN in the next 15-minute candle. Confidence gate at 69% prevents low-conviction signals." />
            <Model name="FinBERT Sentiment" weight="25%" color="#8b5cf6"
              desc="BERT-based NLP model fine-tuned on financial text. Processes news headlines from crypto RSS feeds. Outputs POSITIVE / NEUTRAL / NEGATIVE label with a relevance score. Reduces BUY confidence when sentiment is strongly negative." />
            <Model name="TEN-GNN Spoofing Detector" weight="20%" color="#f59e0b"
              desc="Temporal Event Network GNN that models the order book as a graph. Detects abnormal patterns: large liquidity walls, rapid imbalance shifts, and coordinated bid/ask pressure. Confidence > 0.7 vetoes all BUY signals." />
            <Model name="GNN Structural" weight="5%" color="#06b6d4"
              desc="Graph neural network encoding the structural properties of the order book (depth, spread, level distribution). Provides a low-weight sanity check on the overall market microstructure." />
          </div>
        </Section>

        <Section title="Real-Time Data (WebSocket)">
          <div className="text-xs text-gray-600 space-y-2">
            {[
              ['Price ticker', 'solusdt@ticker', 'Live price, 24h change, volume — updated every 100ms'],
              ['Candlestick chart', 'solusdt@kline_{interval}', 'Real-time OHLCV candles — REST backfill of 500 historical candles on connect'],
              ['Order book', 'solusdt@depth20@100ms', 'Top 20 bid/ask levels — refreshed every 100ms'],
              ['Trade stream', 'solusdt@trade', 'Every executed trade with buyer/seller side'],
              ['Watchlist', '{symbol}@ticker × 6', 'Live prices for SOL, BTC, ETH, BNB, ADA, XRP'],
            ].map(([name, stream, desc]) => (
              <div key={name} className="border-b border-gray-50 pb-2 last:border-0 last:pb-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-gray-700">{name}</span>
                  <code className="text-[10px] bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">{stream}</code>
                </div>
                <div className="text-gray-400 mt-0.5">{desc}</div>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Infrastructure">
          <div className="grid grid-cols-2 gap-3 text-xs">
            {[
              ['Frontend', 'React 18 + TypeScript + Vite + Tailwind CSS'],
              ['Charts', 'lightweight-charts v4 (Binance-style candlesticks)'],
              ['State', 'Zustand (zero-selector re-render optimization)'],
              ['Backend', 'Node.js 20 + Express + WebSocket server'],
              ['AI API', 'FastAPI (Python) on port 8000'],
              ['Process manager', 'PM2 — 4 daemons always running'],
              ['ML training', 'XGBoost + walk-forward validation (no data leak)'],
              ['Data exchange', 'JSON files + REST polling (60s signal, 30s scheduler)'],
            ].map(([name, value]) => (
              <div key={name} className="bg-gray-50 rounded-lg p-3">
                <div className="text-gray-400 mb-0.5">{name}</div>
                <div className="text-gray-700 font-medium">{value}</div>
              </div>
            ))}
          </div>
        </Section>
      </div>
    </div>
  );
}
