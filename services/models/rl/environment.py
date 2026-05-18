"""
environment.py — Custom Gymnasium trading environment for PPO training.

Observation (52-dim):
  - 49 stationary market features (normalized by StandardScaler)
  - 3 portfolio state features: [position, unrealized_pnl_pct, steps_since_trade/100]

Action space: Discrete(3)
  0 = HOLD    — maintain current position
  1 = BUY     — enter / stay long
  2 = SELL    — exit long / go short

Reward:
  step_log_return × position − cost_penalty − drawdown_penalty
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import (
    DRAWDOWN_PENALTY_COEF,
    FEE_RT,
    INITIAL_EQUITY,
    MIN_EQUITY_FRACTION,
    PORTFOLIO_FEATURE_DIM,
    SLIPPAGE_PCT,
    TRADE_PENALTY,
)

_ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}
_COST = FEE_RT + SLIPPAGE_PCT   # total round-trip cost fraction


class TradingEnv(gym.Env):
    """Vectorisable single-asset trading environment."""

    metadata = {"render_modes": []}

    def __init__(self, features: np.ndarray, prices: np.ndarray, *, seed: int | None = None):
        """
        Args:
            features: shape (T, n_features) — pre-scaled market observations
            prices:   shape (T,)            — close prices aligned with features
        """
        super().__init__()
        assert len(features) == len(prices), "features and prices must have same length"

        self._features = features.astype(np.float32)
        self._prices   = prices.astype(np.float32)
        self._T        = len(prices)

        n_market = features.shape[1]
        obs_dim  = n_market + PORTFOLIO_FEATURE_DIM

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # episode state (reset in reset())
        self._t              = 0
        self._position       = 0.0    # 0 = flat, 1 = long
        self._entry_price    = 0.0
        self._equity         = INITIAL_EQUITY
        self._peak_equity    = INITIAL_EQUITY
        self._steps_no_trade = 0

        if seed is not None:
            self.action_space.seed(seed)

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t              = 0
        self._position       = 0.0
        self._entry_price    = 0.0
        self._equity         = INITIAL_EQUITY
        self._peak_equity    = INITIAL_EQUITY
        self._steps_no_trade = 0
        return self._get_obs(), {}

    def step(self, action: int):
        price_now  = float(self._prices[self._t])
        price_next = float(self._prices[min(self._t + 1, self._T - 1)])

        cost = 0.0
        prev_position = self._position

        if action == 1:    # BUY
            if self._position == 0.0:
                self._position    = 1.0
                self._entry_price = price_now * (1 + _COST / 2)
                cost              = _COST / 2 + TRADE_PENALTY
            self._steps_no_trade = 0
        elif action == 2:  # SELL
            if self._position == 1.0:
                self._position = 0.0
                cost           = _COST / 2 + TRADE_PENALTY
            self._steps_no_trade = 0
        else:
            self._steps_no_trade += 1

        # Step return while holding
        if prev_position == 1.0 or (action == 1 and prev_position == 0.0):
            log_ret = math.log(price_next / price_now) if price_now > 0 else 0.0
        else:
            log_ret = 0.0

        self._equity *= math.exp(log_ret * self._position) * (1 - cost)
        self._peak_equity = max(self._peak_equity, self._equity)

        drawdown = (self._peak_equity - self._equity) / self._peak_equity
        reward   = float(log_ret * self._position - cost - DRAWDOWN_PENALTY_COEF * drawdown)

        self._t += 1
        terminated = (
            self._t >= self._T - 1
            or self._equity < INITIAL_EQUITY * MIN_EQUITY_FRACTION
        )

        info = {
            "equity":    self._equity,
            "position":  self._position,
            "price":     price_now,
            "drawdown":  drawdown,
        }
        return self._get_obs(), reward, terminated, False, info

    def render(self):
        pass

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        t = min(self._t, self._T - 1)
        market = self._features[t]

        unrealized_pnl = 0.0
        if self._position == 1.0 and self._entry_price > 0:
            unrealized_pnl = (self._prices[t] - self._entry_price) / self._entry_price

        portfolio = np.array([
            self._position,
            np.clip(unrealized_pnl, -1.0, 1.0),
            min(self._steps_no_trade / 100.0, 1.0),
        ], dtype=np.float32)

        return np.concatenate([market, portfolio])
