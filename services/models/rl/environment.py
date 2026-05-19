"""
environment.py — Custom Gymnasium trading environment for PPO training.

Observation (53-dim):
  - 49 stationary market features (normalized by StandardScaler)
  - 4 portfolio state features:
      [position, unrealized_pnl_pct, steps_since_trade/100, log(equity/bh_equity)]

Action space: Discrete(3)
  0 = HOLD    — maintain current position
  1 = BUY     — enter long
  2 = SELL    — exit long

Reward (benchmark-relative):
  excess_return = agent_step_return - buyhold_step_return
  reward = excess_return - cost - drawdown_penalty + sharpe_excess_bonus

  Being flat when the market rises earns a negative reward (missed move).
  Being flat when the market falls earns a positive reward (avoided loss).
  The agent can only accumulate positive total reward by outperforming buy-and-hold.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    DRAWDOWN_PENALTY_COEF,
    FEE_RT,
    INITIAL_EQUITY,
    MIN_EQUITY_FRACTION,
    PORTFOLIO_FEATURE_DIM,
    SHARPE_REWARD_COEF,
    SHARPE_WINDOW,
    SLIPPAGE_PCT,
    TRADE_PENALTY,
)

_COST = FEE_RT + SLIPPAGE_PCT


class TradingEnv(gym.Env):
    """Vectorisable single-asset trading environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        *,
        random_start: bool = False,
        min_episode_len: int = 500,
        seed: int | None = None,
    ):
        """
        Args:
            features:        shape (T, n_features) — pre-scaled market observations
            prices:          shape (T,)            — close prices aligned with features
            random_start:    if True, each episode begins at a random offset within the
                             slice so the agent encounters all market regimes, not just
                             the ones near the start of the training data.
            min_episode_len: minimum steps after the random start offset.
        """
        super().__init__()
        assert len(features) == len(prices), "features and prices must have same length"

        self._features        = features.astype(np.float32)
        self._prices          = prices.astype(np.float32)
        self._T               = len(prices)
        self._random_start    = random_start
        self._min_episode_len = min_episode_len
        self._start_t         = 0

        n_market = features.shape[1]
        obs_dim  = n_market + PORTFOLIO_FEATURE_DIM

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._t              = 0
        self._position       = 0.0
        self._entry_price    = 0.0
        self._equity         = INITIAL_EQUITY
        self._bh_equity      = INITIAL_EQUITY
        self._peak_equity    = INITIAL_EQUITY
        self._steps_no_trade = 0
        self._return_buf: deque[float] = deque(maxlen=SHARPE_WINDOW)
        self._rng            = np.random.default_rng(seed)

        if seed is not None:
            self.action_space.seed(seed)

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self._random_start:
            max_start = max(0, self._T - self._min_episode_len - 1)
            self._start_t = int(self._rng.integers(0, max_start + 1))
        else:
            self._start_t = 0
        self._t              = self._start_t
        self._position       = 0.0
        self._entry_price    = 0.0
        self._equity         = INITIAL_EQUITY
        self._bh_equity      = INITIAL_EQUITY
        self._peak_equity    = INITIAL_EQUITY
        self._steps_no_trade = 0
        self._return_buf.clear()
        return self._get_obs(), {}

    def step(self, action: int):
        price_now  = float(self._prices[self._t])
        price_next = float(self._prices[min(self._t + 1, self._T - 1)])

        # Buy-and-hold reference return for this step (theoretical, no costs)
        bh_log_ret = math.log(price_next / price_now) if price_now > 0 else 0.0

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

        # Agent's step return: only realised while in position
        in_market = prev_position == 1.0 or (action == 1 and prev_position == 0.0)
        agent_log_ret = bh_log_ret if in_market else 0.0

        # Excess return: alpha vs buy-and-hold
        # Flat + market up  → negative (missed move)
        # Flat + market down → positive (avoided loss)
        # Long in either   → 0 excess (matches benchmark, costs aside)
        excess = agent_log_ret - bh_log_ret

        # Update equities
        self._equity    *= math.exp(agent_log_ret) * (1 - cost)
        self._bh_equity *= math.exp(bh_log_ret)
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = (self._peak_equity - self._equity) / self._peak_equity

        # Sharpe bonus on excess returns — rewards alpha consistency, not raw returns
        self._return_buf.append(excess)
        sharpe_bonus = 0.0
        if len(self._return_buf) >= SHARPE_WINDOW:
            buf = np.asarray(self._return_buf)
            std = buf.std() + 1e-8
            sharpe_bonus = SHARPE_REWARD_COEF * (buf.mean() / std)

        # Drawdown penalty only while holding
        dd_penalty = DRAWDOWN_PENALTY_COEF * drawdown * self._position

        reward = float(excess - cost + sharpe_bonus - dd_penalty)

        self._t += 1
        terminated = (
            self._t >= self._T - 1
            or self._equity < INITIAL_EQUITY * MIN_EQUITY_FRACTION
        )
        # truncated = False (never: SB3 handles time limits separately)

        info = {
            "equity":    self._equity,
            "bh_equity": self._bh_equity,
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

        # Relative performance vs buy-and-hold: positive = outperforming
        rel_perf = math.log(self._equity / max(self._bh_equity, 1e-10))
        rel_perf = float(np.clip(rel_perf, -0.5, 0.5))

        portfolio = np.array([
            self._position,
            np.clip(unrealized_pnl, -1.0, 1.0),
            min(self._steps_no_trade / 100.0, 1.0),
            rel_perf,
        ], dtype=np.float32)

        return np.concatenate([market, portfolio])
