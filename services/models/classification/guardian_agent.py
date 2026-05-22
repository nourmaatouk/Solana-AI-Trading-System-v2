"""
guardian_agent.py

Standalone safety validation for fused realtime decisions before paper trading.
It does not execute trades and does not mutate model outputs; it only returns a
structured allow/block/force-hold decision for the caller to enforce.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from config import (
    CONFIDENCE_THRESHOLD,
    GUARDIAN_DAILY_LOSS_LIMIT,
    GUARDIAN_SLIPPAGE_THRESHOLD,
    GUARDIAN_SPOOFING_THRESHOLD,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_action(action: str | None) -> str | None:
    if action is None:
        return None
    value = str(action).upper()
    if value in ("UP", "BUY"):
        return "BUY"
    if value in ("DOWN", "SELL"):
        return "SELL"
    if value in ("NEUTRAL", "HOLD"):
        return "HOLD"
    return value


def _normalise_position(position: str | None) -> str:
    if position is None:
        return "FLAT"
    value = str(position).upper()
    if value in ("UP", "BUY", "LONG"):
        return "LONG"
    if value in ("DOWN", "SELL", "SHORT"):
        return "SHORT"
    return "FLAT"


def _is_present(value: float | None) -> bool:
    return value is not None


@dataclass(frozen=True)
class GuardianInput:
    fused_score: float
    risk_level: str
    confidence: float | None
    classification_probability_up: float | None
    sentiment_score: float | None
    spoofing_confidence: float | None
    gnn_risk: float | None
    ppo_action: str | None
    fusion_action: str
    current_position: str
    unrealized_pnl: float | None
    daily_loss: float | None
    daily_loss_limit: float | None = GUARDIAN_DAILY_LOSS_LIMIT
    estimated_slippage: float | None = None
    slippage_threshold: float | None = GUARDIAN_SLIPPAGE_THRESHOLD

    def __post_init__(self) -> None:
        object.__setattr__(self, "risk_level", str(self.risk_level).upper())
        object.__setattr__(self, "fusion_action", _normalise_action(self.fusion_action) or "HOLD")
        object.__setattr__(self, "ppo_action", _normalise_action(self.ppo_action))
        object.__setattr__(self, "current_position", _normalise_position(self.current_position))


@dataclass(frozen=True)
class GuardianResult:
    decision: str
    reason: str
    fusion_action: str
    ppo_action: str | None
    confidence: float | None
    risk_level: str
    fused_score: float
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


class GuardianAgent:
    def __init__(
        self,
        *,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        spoofing_threshold: float = GUARDIAN_SPOOFING_THRESHOLD,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.spoofing_threshold = float(spoofing_threshold)

    def validate(self, payload: GuardianInput) -> GuardianResult:
        if payload.risk_level == "HIGH":
            return self._result("BLOCK", "high risk level", payload)

        if (
            _is_present(payload.confidence)
            and float(payload.confidence) < self.confidence_threshold
        ):
            return self._result("BLOCK", "confidence below threshold", payload)

        if (
            _is_present(payload.spoofing_confidence)
            and float(payload.spoofing_confidence) > self.spoofing_threshold
        ):
            return self._result("BLOCK", "spoofing detected", payload)

        if (
            payload.ppo_action is not None
            and payload.ppo_action != payload.fusion_action
            and payload.ppo_action != "HOLD"
        ):
            return self._result("BLOCK", "PPO conflicts with Fusion", payload)

        if (
            _is_present(payload.daily_loss)
            and _is_present(payload.daily_loss_limit)
            and float(payload.daily_loss) >= float(payload.daily_loss_limit)
        ):
            return self._result("BLOCK", "daily loss limit reached", payload)

        if (
            _is_present(payload.estimated_slippage)
            and _is_present(payload.slippage_threshold)
            and float(payload.estimated_slippage) > float(payload.slippage_threshold)
        ):
            return self._result("BLOCK", "slippage too high", payload)

        if payload.current_position == "LONG" and payload.fusion_action == "BUY":
            return self._result("FORCE_HOLD", "already long", payload)

        if payload.current_position == "SHORT" and payload.fusion_action == "SELL":
            return self._result("FORCE_HOLD", "already short", payload)

        return self._result("ALLOW", "all checks passed", payload)

    @staticmethod
    def _result(decision: str, reason: str, payload: GuardianInput) -> GuardianResult:
        return GuardianResult(
            decision=decision,
            reason=reason,
            fusion_action=payload.fusion_action,
            ppo_action=payload.ppo_action,
            confidence=payload.confidence,
            risk_level=payload.risk_level,
            fused_score=float(payload.fused_score),
            timestamp=_utc_now_iso(),
        )


def format_guardian_log(result: GuardianResult, payload: GuardianInput) -> str:
    parts = [
        f"[GUARDIAN] decision={result.decision}",
        f"reason={result.reason}",
    ]
    if result.decision == "FORCE_HOLD":
        parts.append(f"position={payload.current_position}")
    parts.extend(
        [
            f"fused_score={result.fused_score:.4f}",
            f"confidence={0.0 if result.confidence is None else float(result.confidence):.4f}",
            f"risk={result.risk_level}",
            f"action={result.fusion_action}",
        ]
    )
    return " | ".join(parts)
