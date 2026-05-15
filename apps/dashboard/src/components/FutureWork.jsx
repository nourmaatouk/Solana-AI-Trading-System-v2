const ITEMS = [
  { icon: '🔑', title: 'Connect Solana Wallet', desc: 'Phantom / Backpack wallet signing' },
  { icon: '⚡', title: 'Real Solana Execution', desc: 'Jupiter/Raydium swap transactions' },
  { icon: '📖', title: 'OpenBook / Helius', desc: 'Live Level-2 orderbook ingestion' },
  { icon: '🤖', title: 'RL / PPO Agent', desc: 'Reinforcement learning position sizing' },
  { icon: '🧠', title: 'ElizaOS Orchestration', desc: 'LLM reasoning + memory layer' },
  { icon: '🗄', title: 'ChromaDB Memory', desc: 'Semantic search over past decisions' },
]

export default function FutureWork() {
  return (
    <div className="card full-width">
      <div className="card-header">
        <span className="card-title">Future Work — Disabled</span>
        <span className="badge badge-gray">Not connected to real wallet</span>
      </div>
      <div className="card-body">
        <div className="alert alert-warn">
          ⚠ The features below are planned but not yet implemented. No on-chain execution or real wallet interaction occurs in this version.
        </div>
        <div className="future-grid">
          {ITEMS.map((item) => (
            <div key={item.title} className="future-card">
              <div className="future-icon">{item.icon}</div>
              <div className="future-title">{item.title}</div>
              <div className="future-desc">{item.desc}</div>
              <div style={{ marginTop: 8 }}>
                <span className="badge badge-gray" style={{ fontSize: 10 }}>Future</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
