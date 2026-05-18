import asyncio
import json
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import websockets

# =========================================================
# CONFIG
# =========================================================
SYMBOL = os.environ.get("ACTIVE_SYMBOL", "SOLUSDT").lower().replace("/", "")
WS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@depth@100ms"
TOP_N_LEVELS = 10  # Number of price levels to include in volume calculation
SAVE_INTERVAL_SEC = 60  # Save aggregated features every minute
AGGREGATION_WINDOW = "1min" # Change to 15min if preferred

BASE_DIR = Path(__file__).resolve().parents[3] / "services" / "models" / "classification"
OUTPUT_PATH = BASE_DIR / "data" / "realtime" / "orderbook_features.csv"

class OrderBookCollector:
    def __init__(self, top_n=10):
        self.top_n = top_n
        self.raw_updates = []  # Store 100ms snapshots
        self.aggregated_features = []
        
    def calculate_imbalance(self, bids, asks):
        """
        Calculates order book imbalance: (bid_v - ask_v) / (bid_v + ask_v)
        """
        # Binance depth format: [[price, qty], ...]
        bid_vol = sum(float(b[1]) for b in bids[:self.top_n])
        ask_vol = sum(float(a[1]) for a in asks[:self.top_n])
        
        if (bid_vol + ask_vol) == 0:
            return 0.0
            
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def process_snapshot(self, data):
        """
        Extracts features from a single depth snapshot.
        """
        bids = data.get("b", [])
        asks = data.get("a", [])
        
        if not bids or not asks:
            return None
            
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        imbalance = self.calculate_imbalance(bids, asks)
        
        return {
            "timestamp": datetime.now(timezone.utc),
            "mid_price": mid_price,
            "spread": spread,
            "imbalance": imbalance,
            "bid_volume": sum(float(b[1]) for b in bids[:self.top_n]),
            "ask_volume": sum(float(a[1]) for a in asks[:self.top_n])
        }

    async def run(self):
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        retry_delay = 1
        max_delay = 60
        
        while True:
            try:
                print(f"Connecting to Binance WebSocket: {WS_URL}")
                async with websockets.connect(WS_URL) as ws:
                    print("Connected successfully.")
                    retry_delay = 1  # Reset delay on successful connection
                    last_save = time.time()
                    
                    while True:
                        response = await ws.recv()
                        data = json.loads(response)
                        
                        snapshot = self.process_snapshot(data)
                        if snapshot:
                            self.raw_updates.append(snapshot)
                        
                        # Periodic Aggregation and Saving
                        if time.time() - last_save >= SAVE_INTERVAL_SEC:
                            if self.raw_updates:
                                self.aggregate_and_save()
                                self.raw_updates = [] # Clear raw buffer
                            last_save = time.time()
                            
            except Exception as e:
                print(f"WebSocket Error / Disconnect: {e}")
                
            print(f"Reconnecting in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_delay)

    def aggregate_and_save(self):
        """
        Aggregates raw 100ms updates into periodic features.
        """
        df_raw = pd.DataFrame(self.raw_updates)
        df_raw.set_index("timestamp", inplace=True)
        
        # Calculate stats for the window
        resampled = df_raw.resample(AGGREGATION_WINDOW).agg({
            "imbalance": ["mean", "std"],
            "spread": "mean",
            "mid_price": "last",
            "bid_volume": "mean",
            "ask_volume": "mean"
        })
        
        # Flatten columns
        resampled.columns = [
            "imbalance_mean", "imbalance_std", 
            "spread_mean", "mid_price", 
            "bid_volume_mean", "ask_volume_mean"
        ]
        
        # Add pressure indicator (binary flag for high imbalance)
        resampled["pressure_indicator"] = np.where(resampled["imbalance_mean"] > 0.5, 1, 
                                          np.where(resampled["imbalance_mean"] < -0.5, -1, 0))
        
        resampled = resampled.dropna().reset_index()
        
        # Append to CSV
        file_exists = OUTPUT_PATH.exists()
        resampled.to_csv(OUTPUT_PATH, mode='a', header=not file_exists, index=False)
        print(f"Saved {len(resampled)} aggregated orderbook rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    import time
    collector = OrderBookCollector(top_n=TOP_N_LEVELS)
    asyncio.run(collector.run())
