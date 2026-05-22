"""
environment.py — Custom Gymnasium trading environment for PPO training.

Observation (54-dim when pipeline_signals are provided, 53-dim legacy otherwise):
  - N market features (normalized by StandardScaler passed in from train_ppo.py)
  - P pipeline signal features (optional, inserted after market features):
      [xgb_prob_up]  — XGBoost P(UP) at the current bar
  - 4 portfolio state features:
      [position, unrealized_pnl_pct, steps_since_trade/100, log(equity/bh_equity)]

Action space: Discrete(3)
  0 = HOLD    — maintain current position
  1 = BUY     — enter long (no-op if already long)
  2 = SELL    — exit long  (no-op if already flat)

Reward (absolute portfolio return):
  reward = agent_log_ret − cost − drawdown_penalty + sharpe_bonus − invalid_penalty

  agent_log_ret = market log-return when in position, 0 when flat.
  Flat when market rises  → reward = 0   (no reward, but no benchmark penalty).
  Flat when market falls  → reward = 0   (agent implicitly avoided the loss).
  Long + market rises     → reward ≈ +log_ret − cost  (positive).
  Long + market falls     → reward ≈ −log_ret − cost  (negative — incentives exit).

  This formulation is learnable with the xgb_prob_up pipeline signal:
  the policy can learn "BUY when prob_up is high, SELL when it falls".

  Invalid actions (BUY when long, SELL when flat) incur INVALID_ACTION_PENALTY
  in addition to being no-ops, teaching the policy to avoid wasted actions.
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
    INVALID_ACTION_PENALTY,
    MIN_EQUITY_FRACTION,
    MIN_HOLD_STEPS,
    OVERTRADING_PENALTY,
    PIPELINE_IDX_FUSED,
    PIPELINE_IDX_RISK,
    PIPELINE_SIGNAL_DIM,
    PORTFOLIO_FEATURE_DIM,
    RISKY_BUY_PENALTY,
    SHARPE_REWARD_COEF,
    SHARPE_WINDOW,
    SIGNAL_ALIGN_COEF,
    SLIPPAGE_PCT,
    TRADE_PENALTY,
)

_COST = FEE_RT + SLIPPAGE_PCT


class TradingEnv(gym.Env):
    """Vectorisable single-asset trading environment.

    Backward-compatible: ``pipeline_signals`` is optional.  When omitted the
    environment behaves exactly as the original 53-dim version, allowing the
    legacy ``ppo_trading_agent.zip`` model to be evaluated without changes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        pipeline_signals: np.ndarray | None = None,
        *,
        random_start: bool = False,
        min_episode_len: int = 500,
        seed: int | None = None,
    ):
        """
        Args:
            features:          shape (T, n_market) — pre-scaled market observations
            prices:            shape (T,)           — close prices aligned with features
            pipeline_signals:  shape (T, P) or None — Layer-3 pipeline outputs
                               (xgb_prob_up, …).  When None the env falls back to
                               the original 53-dim observation for legacy model eval.
            random_start:      if True, episodes start at a random offset so the
                               agent sees all market regimes during training.
            min_episode_len:   minimum steps after the random start offset.
        """
        super().__init__()
        if len(features) != len(prices):
            raise ValueError(
                f"features and prices must have the same length "
                f"(got {len(features)} vs {len(prices)})"
            )

        self._features = features.astype(np.float32)
        self._prices   = prices.astype(np.float32)
        self._T        = len(prices)

        if pipeline_signals is not None:
            if len(pipeline_signals) != self._T:
                raise ValueError(
                    f"pipeline_signals length must match features length "
                    f"(got {len(pipeline_signals)} vs {self._T})"
                )
            # Accept any pipeline dim ≥ 1; callers are responsible for alignment.
            # PIPELINE_SIGNAL_DIM=1 (aligned) and FUSION_SIGNAL_DIM=3 (fusion v2) both valid.
            self._pipeline_signals = pipeline_signals.astype(np.float32)
            self._n_pipeline       = pipeline_signals.shape[1]
            self._use_pipeline     = True
        else:
            self._pipeline_signals = np.zeros((self._T, 0), dtype=np.float32)
            self._n_pipeline       = 0
            self._use_pipeline     = False

        self._random_start    = random_start
        self._min_episode_len = min_episode_len
        self._start_t         = 0

        n_market   = features.shape[1]
        n_pipeline = self._n_pipeline
        obs_dim    = n_market + n_pipeline + PORTFOLIO_FEATURE_DIM

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._t                = 0
        self._position         = 0.0
        self._entry_price      = 0.0
        self._equity           = INITIAL_EQUITY
        self._bh_equity        = INITIAL_EQUITY
        self._peak_equity      = INITIAL_EQUITY
        self._steps_no_trade   = 0
        self._steps_since_entry = 0   # bars since last valid BUY; used for overtrading detection
        self._return_buf: deque[float] = deque(maxlen=SHARPE_WINDOW)
        self._rng              = np.random.default_rng(seed)

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
        self._t                 = self._start_t
        self._position          = 0.0
        self._entry_price       = 0.0
        self._equity            = INITIAL_EQUITY
        self._bh_equity         = INITIAL_EQUITY
        self._peak_equity       = INITIAL_EQUITY
        self._steps_no_trade    = 0
        self._steps_since_entry = 0
        self._return_buf.clear()
        return self._get_obs(), {}

    def step(self, action: int):
        price_now  = float(self._prices[self._t])
        price_next = float(self._prices[min(self._t + 1, self._T - 1)])

        # Buy-and-hold log-return for this bar (bh_equity tracking and reward)
        bh_log_ret = math.log(price_next / price_now) if price_now > 0 else 0.0

        cost          = 0.0
        prev_position = self._position
        invalid       = False

        if action == 1:    # BUY
            if self._position == 0.0:
                self._position      = 1.0
                self._entry_price   = price_now * (1 + _COST / 2)
                cost                = _COST / 2 + TRADE_PENALTY
                self._steps_since_entry = 0
            else:
                invalid = True   # BUY when already long
            self._steps_no_trade = 0
        elif action == 2:  # SELL
            if self._position == 1.0:
                self._position = 0.0
                cost           = _COST / 2 + TRADE_PENALTY
            else:
                invalid = True   # SELL when already flat
            self._steps_no_trade = 0
        else:
            self._steps_no_trade    += 1
            if self._position == 1.0:
                self._steps_since_entry += 1

        # Agent's absolute log-return: realised only when in position
        in_market     = prev_position == 1.0 or (action == 1 and prev_position == 0.0)
        agent_log_ret = bh_log_ret if in_market else 0.0

        # Update equities
        self._equity    *= math.exp(agent_log_ret) * (1 - cost)
        self._bh_equity *= math.exp(bh_log_ret)
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = (self._peak_equity - self._equity) / self._peak_equity

        # ── Reward: absolute portfolio return with shaped penalties/bonuses ────
        self._return_buf.append(agent_log_ret)
        sharpe_bonus = 0.0
        if len(self._return_buf) >= SHARPE_WINDOW:
            buf  = np.asarray(self._return_buf)
            std  = buf.std() + 1e-8
            sharpe_bonus = SHARPE_REWARD_COEF * (buf.mean() / std)

        dd_penalty      = DRAWDOWN_PENALTY_COEF * drawdown * self._position
        invalid_penalty = INVALID_ACTION_PENALTY if invalid else 0.0

        # Overtrading penalty: SELL within MIN_HOLD_STEPS bars of entry
        flip_penalty = 0.0
        if action == 2 and prev_position == 1.0 and not invalid:
            if self._steps_since_entry < MIN_HOLD_STEPS:
                flip_penalty = OVERTRADING_PENALTY

        # Risky-BUY penalty: BUY when risk_encoded (pipeline index 2) is HIGH
        risky_penalty = 0.0
        if action == 1 and prev_position == 0.0 and self._use_pipeline and self._n_pipeline > PIPELINE_IDX_RISK:
            risk_encoded = float(self._pipeline_signals[min(self._t, self._T - 1), PIPELINE_IDX_RISK])
            if risk_encoded >= 1.0:
                risky_penalty = RISKY_BUY_PENALTY

        # Signal-alignment bonus: log_ret direction matches fused_score sign
        align_bonus = 0.0
        if self._use_pipeline and self._n_pipeline > PIPELINE_IDX_FUSED and agent_log_ret != 0.0:
            fused_score = float(self._pipeline_signals[min(self._t, self._T - 1), PIPELINE_IDX_FUSED])
            if fused_score != 0.0 and math.copysign(1, agent_log_ret) == math.copysign(1, fused_score):
                align_bonus = SIGNAL_ALIGN_COEF * abs(fused_score) * abs(agent_log_ret)

        reward = float(
            agent_log_ret - cost + sharpe_bonus
            - dd_penalty - invalid_penalty - flip_penalty - risky_penalty
            + align_bonus
        )

        self._t += 1
        terminated = (
            self._t >= self._T - 1
            or self._equity < INITIAL_EQUITY * MIN_EQUITY_FRACTION
        )

        info = {
            "equity":    self._equity,
            "bh_equity": self._bh_equity,
            "position":  self._position,
            "price":     price_now,
            "drawdown":  drawdown,
            "invalid":   invalid,
        }
        return self._get_obs(), reward, terminated, False, info

    def render(self):
        pass

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        t      = min(self._t, self._T - 1)
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

        # Pipeline signals are empty (0-width) when use_pipeline=False — graceful fallback.
        pipeline = self._pipeline_signals[t]   # shape () when n_pipeline==0
        return np.concatenate([market, pipeline, portfolio])
