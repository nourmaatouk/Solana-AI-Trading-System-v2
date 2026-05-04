# 🗺️ Implementation Roadmap & Action Plan

**Current Date**: May 4, 2026  
**Phase**: Integration (Layers 4-6 wiring)  
**Estimated Time to Production**: 2-3 weeks  

---

## 📍 Current State vs. Target State

### Current State (May 4, 2026) ✅

```
Layer 1: Data Ingest
├─ ✅ Designed (Binance, OpenBook, Helius, Reddit)
└─ 🟡 Ready for live testing

Layer 2: Processing
├─ ✅ Designed (Features, normalization, windowing)
└─ 🟡 Ready for feature stream

Layer 3: AI Models ✅ COMPLETE
├─ 🔴 Spoofing: TEN model (tested)
├─ 🟢 Classification: XGBoost (tested)
├─ 🔵 NLP Sentiment: FinBERT (tested)
├─ 🟣 Fusion: Weighted aggregation (22/23 tests ✅)
└─ ✅ Producing fused decisions (8 decisions/test)

Layer 4: Agent ⚠️ ISOLATED
├─ ✅ Designed (ElizaOS orchestrator)
└─ ❌ NOT consuming Layer 3 output

Layer 5: Execution ⚠️ ISOLATED
├─ ✅ Designed (Jupiter, Raydium, Jito)
└─ ❌ NOT consuming Layer 4 output

Layer 6: Dashboard 🟡 PARTIAL
├─ ✅ Designed (FastAPI, React, SHAP)
└─ ❌ NOT consuming Layer 5 output
```

### Target State (Production Ready)

```
Layer 1 → Layer 2 → Layer 3 → Layer 4 → Layer 5 → Layer 6
  ↓         ↓         ↓         ↓         ↓        ↓
Queues    Queues    Queues    Queues    Queues   Queues
  ↓         ↓         ↓         ↓         ↓        ↓
Kafka/Redis Streams Connection Points
  ↓         ↓         ↓         ↓         ↓        ↓
Monitoring, Logging, Metrics Collection at Each Layer
```

---

## 🎯 Phase 1: Integration (Weeks 1-2)

### Task 1.1: Layer 4 Integration (1-2 days)

**Objective**: Wire Layer 3 fused decisions → Layer 4 orchestrator

**File to Create**: `apps/agent/fusion_consumer.py`

```python
# apps/agent/fusion_consumer.py
import asyncio
from datetime import datetime
from fusion.src.fusion_contracts import FusedDecision
from orchestrator import ElizaOSOrchestrator

class FusionConsumer:
    def __init__(self):
        self.orchestrator = ElizaOSOrchestrator()
        self.decision_history = []
    
    async def consume_fusion_decisions(self):
        """Listen for decisions from Layer 3"""
        while True:
            # Get decision from queue
            decision = await self.fusion_queue.get()  # FusedDecision object
            
            # Log decision
            self.decision_history.append({
                'timestamp': decision.timestamp,
                'action': decision.action,
                'score': decision.fused_score,
                'risk': decision.risk_level,
                'alignment': decision.alignment_score,
            })
            
            # Route to orchestrator
            await self.orchestrator.process_decision(
                action=decision.action,
                confidence=abs(decision.fused_score),
                risk_level=decision.risk_level,
                alignment_score=decision.alignment_score,
                reasoning=decision.reasoning,  # Provide decision context
            )
    
    async def get_decision_history(self, limit=100):
        """Retrieve recent decisions"""
        return self.decision_history[-limit:]

# Startup
async def main():
    consumer = FusionConsumer()
    await consumer.consume_fusion_decisions()
```

**Checklist**:
- [ ] Create fusion_consumer.py
- [ ] Implement decision queue connection
- [ ] Add decision logging
- [ ] Test with mock decisions
- [ ] Wire to orchestrator

**Expected Outcome**:
```
✅ Fused decisions routed to orchestrator
✅ Decision history tracking
✅ Reasoning preserved end-to-end
```

---

### Task 1.2: Layer 5 Integration (1-2 days)

**Objective**: Wire Layer 4 approvals → Layer 5 execution

**File to Create**: `services/execution/fusion_executor.py`

```python
# services/execution/fusion_executor.py
from router import ExecutionRouter
from position_sizer import PositionSizer
from risk.circuit_breaker import CircuitBreaker

class FusionExecutor:
    def __init__(self):
        self.router = ExecutionRouter()
        self.sizer = PositionSizer()
        self.circuit_breaker = CircuitBreaker()
        self.positions = {}
    
    async def execute_decision(self, orchestrator_signal):
        """
        Execute approved decision from Layer 4
        
        orchestrator_signal = {
            'action': 'BUY' | 'SELL' | 'HOLD',
            'position_size_fraction': 0.1,  # % of portfolio
            'risk_level': 'LOW' | 'MEDIUM' | 'HIGH',
            'reasoning': dict,
        }
        """
        
        # Validate risk
        if not self.circuit_breaker.validate(orchestrator_signal):
            return {'status': 'rejected', 'reason': 'Circuit breaker triggered'}
        
        if orchestrator_signal['action'] == 'HOLD':
            return {'status': 'skipped', 'reason': 'HOLD signal'}
        
        # Calculate position size (Kelly Criterion)
        position_size = self.sizer.calculate_size(
            signal_strength=orchestrator_signal['position_size_fraction'],
            risk_level=orchestrator_signal['risk_level'],
            portfolio_value=self.get_portfolio_value(),
        )
        
        # Execute trade
        try:
            if orchestrator_signal['action'] == 'BUY':
                tx_hash = await self.router.execute_buy(position_size)
            else:  # SELL
                tx_hash = await self.router.execute_sell(position_size)
            
            # Track position
            self.positions[tx_hash] = {
                'entry_price': self.get_spot_price(),
                'size': position_size,
                'action': orchestrator_signal['action'],
                'timestamp': datetime.now(),
                'stop_loss': self.get_spot_price() * 0.95,
                'profit_target': self.get_spot_price() * 1.10,
            }
            
            return {
                'status': 'executed',
                'tx_hash': tx_hash,
                'size': position_size,
            }
        
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
            }
```

**Checklist**:
- [ ] Create fusion_executor.py
- [ ] Implement decision-to-execution pipeline
- [ ] Add position tracking
- [ ] Add Kelly Criterion sizing
- [ ] Test with testnet

**Expected Outcome**:
```
✅ Decisions converted to transactions
✅ Position tracking enabled
✅ Risk management active
```

---

### Task 1.3: Real-Time Streaming (2-3 days)

**Objective**: Convert batch pipeline → real-time streaming

**Files to Create**:
- `services/models/fusion/src/stream_adapter.py`
- `services/ingest/streaming_handler.py`

```python
# services/models/fusion/src/stream_adapter.py
import asyncio
from kafka import KafkaConsumer, KafkaProducer
from fusion_aggregator import FusionAggregator

class StreamingFusionAdapter:
    def __init__(self):
        self.aggregator = FusionAggregator()
        
        # Kafka topics
        self.classification_consumer = KafkaConsumer('classification-signals')
        self.sentiment_consumer = KafkaConsumer('sentiment-signals')
        self.spoofing_consumer = KafkaConsumer('spoofing-signals')
        self.decision_producer = KafkaProducer('fused-decisions')
        
        # Buffer for temporal alignment
        self.classification_buffer = {}
        self.sentiment_buffer = {}
        self.spoofing_buffer = {}
        self.buffer_timeout = 5  # seconds
    
    async def stream_fuse_decisions(self):
        """Continuous streaming fusion"""
        while True:
            # Collect signals with timeout
            tasks = [
                asyncio.wait_for(
                    self._get_next_signal(self.classification_consumer),
                    timeout=self.buffer_timeout
                ),
                asyncio.wait_for(
                    self._get_next_signal(self.sentiment_consumer),
                    timeout=self.buffer_timeout
                ),
                asyncio.wait_for(
                    self._get_next_signal(self.spoofing_consumer),
                    timeout=self.buffer_timeout
                ),
            ]
            
            try:
                # Get all available signals within timeout
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.TimeoutError:
                pass  # Use buffered signals
            
            # Fuse available signals
            decision = self.aggregator.fuse_batch(
                classification_df=self.classification_buffer,
                sentiment_df=self.sentiment_buffer,
                spoofing_df=self.spoofing_buffer,
            )
            
            # Publish decision
            self.decision_producer.send(
                'fused-decisions',
                decision.to_dict()
            )
            
            # Clear buffers
            self.classification_buffer.clear()
            self.sentiment_buffer.clear()
            self.spoofing_buffer.clear()
```

**Checklist**:
- [ ] Set up Kafka or Redis Streams
- [ ] Create stream adapter
- [ ] Implement buffer management
- [ ] Handle backpressure
- [ ] Test latency < 100ms

**Expected Outcome**:
```
✅ Real-time decision generation
✅ Sub-100ms latency
✅ Continuous operation
```

---

## 📅 Phase 2: Monitoring (Weeks 2-3)

### Task 2.1: Observability Layer (1 day)

**File to Create**: `services/monitoring/metrics_collector.py`

```python
# services/monitoring/metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge

class MetricsCollector:
    def __init__(self):
        # Counters
        self.decisions_total = Counter(
            'trading_decisions_total',
            'Total decisions',
            ['action', 'risk_level']
        )
        
        self.trades_executed = Counter(
            'trades_executed_total',
            'Executed trades',
            ['action', 'status']
        )
        
        # Histograms (latency)
        self.decision_latency = Histogram(
            'decision_latency_ms',
            'Decision generation latency'
        )
        
        self.execution_latency = Histogram(
            'execution_latency_ms',
            'Trade execution latency'
        )
        
        # Gauges
        self.active_positions = Gauge(
            'active_positions',
            'Number of open positions'
        )
        
        self.portfolio_pnl = Gauge(
            'portfolio_pnl',
            'Portfolio P&L'
        )

    def record_decision(self, decision):
        """Record fused decision metrics"""
        self.decisions_total.labels(
            action=decision.action,
            risk_level=decision.risk_level
        ).inc()
        
        self.decision_latency.observe(
            decision.generation_time_ms
        )

    def record_trade(self, tx_hash, status):
        """Record trade execution"""
        self.trades_executed.labels(
            action=self.trades[tx_hash]['action'],
            status=status
        ).inc()
        
        self.execution_latency.observe(
            self.trades[tx_hash]['latency_ms']
        )
```

**Checklist**:
- [ ] Set up Prometheus
- [ ] Create metrics collector
- [ ] Add metrics to each layer
- [ ] Create Grafana dashboards
- [ ] Alert on thresholds

---

### Task 2.2: Decision Audit Trail (1 day)

**File to Create**: `services/monitoring/audit_logger.py`

```python
# services/monitoring/audit_logger.py
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

class DecisionAuditEntry(declarative_base().__bases__[0]):
    __tablename__ = 'decision_audit'
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime)
    fused_score = Column(Float)
    action = Column(String)
    risk_level = Column(String)
    alignment_score = Column(Float)
    
    # Source signals
    classification_score = Column(Float)
    sentiment_score = Column(Float)
    spoofing_confidence = Column(Float)
    
    # Execution
    executed = Column(String)  # 'yes' | 'no' | 'rejected'
    tx_hash = Column(String)
    execution_reason = Column(String)
    
    # Outcome (after 24h)
    outcome_price = Column(Float)
    pnl = Column(Float)
    
    # Metadata
    reasoning = Column(JSON)

class AuditLogger:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
    
    def log_decision(self, decision, execution_result):
        """Log complete decision lifecycle"""
        session = self.Session()
        
        entry = DecisionAuditEntry(
            id=f"{decision.timestamp}_{decision.action}",
            timestamp=decision.timestamp,
            fused_score=decision.fused_score,
            action=decision.action,
            risk_level=decision.risk_level,
            alignment_score=decision.alignment_score,
            
            classification_score=decision.classification_confidence,
            sentiment_score=decision.sentiment_relevance,
            spoofing_confidence=decision.spoofing_confidence,
            
            executed=execution_result['status'],
            tx_hash=execution_result.get('tx_hash'),
            execution_reason=execution_result.get('reason'),
            
            reasoning=decision.reasoning,
        )
        
        session.add(entry)
        session.commit()
```

**Checklist**:
- [ ] Set up audit database
- [ ] Create audit logger
- [ ] Log all decisions end-to-end
- [ ] Add outcome tracking
- [ ] Create audit reports

---

### Task 2.3: Dashboard Integration (2-3 days)

**File to Create**: `apps/api/routes/decisions.py`

```python
# apps/api/routes/decisions.py
from fastapi import APIRouter, WebSocket
from datetime import datetime, timedelta

router = APIRouter(prefix="/api")

@router.get("/decisions/recent")
async def get_recent_decisions(limit: int = 100):
    """Last N fused decisions"""
    decisions = audit_logger.get_recent(limit)
    return decisions

@router.get("/decisions/stats")
async def get_decision_stats(window_hours: int = 24):
    """Decision statistics for time window"""
    decisions = audit_logger.get_recent_by_time(
        datetime.now() - timedelta(hours=window_hours)
    )
    
    stats = {
        'total_decisions': len(decisions),
        'buy_count': len([d for d in decisions if d.action == 'BUY']),
        'sell_count': len([d for d in decisions if d.action == 'SELL']),
        'hold_count': len([d for d in decisions if d.action == 'HOLD']),
        'avg_alignment': sum(d.alignment_score for d in decisions) / len(decisions),
        'win_rate': calculate_win_rate(decisions),
    }
    
    return stats

@router.get("/positions/current")
async def get_current_positions():
    """Currently open positions"""
    positions = execution_layer.get_open_positions()
    return positions

@router.get("/pnl/daily")
async def get_daily_pnl():
    """Daily P&L breakdown"""
    pnl = calculate_daily_pnl()
    return pnl

@router.websocket("/ws/decisions")
async def websocket_decisions(websocket: WebSocket):
    """Real-time decision stream"""
    await websocket.accept()
    
    while True:
        # Get latest decision
        decision = await fusion_queue.get()
        
        # Send to client
        await websocket.send_json({
            'timestamp': str(decision.timestamp),
            'action': decision.action,
            'score': decision.fused_score,
            'risk': decision.risk_level,
            'alignment': decision.alignment_score,
        })
```

**Checklist**:
- [ ] Create decision endpoints
- [ ] Implement WebSocket stream
- [ ] Create React components
- [ ] Build dashboard views
- [ ] Add real-time updates

---

## 📊 Phase 3: Optimization (Week 3+)

### Task 3.1: Weight Optimization

**Objective**: A/B test different fusion weights

```python
# Test configurations
configs = [
    # Current
    {'classification': 0.50, 'sentiment': 0.25, 'spoofing': 0.20, 'gnn': 0.05},
    # More aggressive
    {'classification': 0.60, 'sentiment': 0.20, 'spoofing': 0.15, 'gnn': 0.05},
    # More conservative
    {'classification': 0.40, 'sentiment': 0.30, 'spoofing': 0.25, 'gnn': 0.05},
    # Sentiment-heavy
    {'classification': 0.40, 'sentiment': 0.40, 'spoofing': 0.15, 'gnn': 0.05},
]

# A/B test each configuration
results = {}
for config in configs:
    aggregator = FusionAggregator(**config)
    pnl, sharpe, win_rate = backtest(aggregator, historical_data)
    results[str(config)] = {'pnl': pnl, 'sharpe': sharpe, 'win_rate': win_rate}

# Select best
best_config = max(results, key=lambda x: results[x]['sharpe'])
```

---

### Task 3.2: Performance Tuning

**Checklist**:
- [ ] Profile decision generation (target: < 10ms)
- [ ] Profile trade execution (target: < 50ms)
- [ ] Optimize Kaggle adapter (vectorize operations)
- [ ] Cache model predictions where possible
- [ ] Load testing with 1000+ TPS

---

### Task 3.3: Risk Management Hardening

**Checklist**:
- [ ] Implement drawdown limits (stop if -10% daily)
- [ ] Add trade frequency caps (max N trades/hour)
- [ ] Implement volatility-adjusted sizing
- [ ] Add emergency procedures (pause trading)
- [ ] Add position exit rules (take profit at +5%, stop loss at -5%)

---

## ✅ Success Criteria by Phase

### Phase 1: Integration ✅
- [ ] Layer 4 consuming Layer 3 decisions
- [ ] Layer 5 executing Layer 4 approvals
- [ ] Real-time streaming operational
- [ ] 0 errors in 24h operation
- [ ] Latency < 100ms end-to-end

### Phase 2: Monitoring ✅
- [ ] All metrics collecting
- [ ] Audit trail recording all decisions
- [ ] Dashboard displaying live data
- [ ] Alerts firing correctly
- [ ] No data loss

### Phase 3: Optimization ✅
- [ ] Best weight configuration selected
- [ ] Performance profiled (< 50ms execution)
- [ ] Risk management hardened
- [ ] Backtest results positive
- [ ] Ready for live trading

---

## 📈 Expected Timeline

```
Week 1-2 (May 5-18): Integration
├─ Day 1-2: Layer 4 integration ✓
├─ Day 3-4: Layer 5 integration ✓
└─ Day 5-10: Streaming pipeline ✓

Week 2-3 (May 19-25): Monitoring
├─ Day 1: Observability ✓
├─ Day 2: Audit trail ✓
└─ Day 3-5: Dashboard ✓

Week 3+ (May 26+): Optimization
├─ Weight tuning
├─ Performance optimization
├─ Risk hardening
└─ Live trading (if approved)
```

---

## 🎯 Go-Live Checklist

Before live trading:

- [ ] Phase 1 (Integration) **COMPLETE**
- [ ] Phase 2 (Monitoring) **COMPLETE**
- [ ] Phase 3 (Optimization) **COMPLETE**
- [ ] 7-day testnet validation **COMPLETE**
- [ ] Security audit **COMPLETE**
- [ ] Risk assessment **APPROVED**
- [ ] Runbook/Procedures **DOCUMENTED**
- [ ] On-call rotation **ESTABLISHED**

---

## 📞 Support & Escalation

**Layer Issues**: Contact responsible team  
**Critical Bug**: Pause trading, investigate, restart  
**Performance Issues**: Check metrics, profile, optimize  
**Risk Issues**: Activate circuit breaker immediately  

---

**Ready to start Phase 1?** 🚀
