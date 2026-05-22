import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardian_agent import GuardianAgent, GuardianInput, format_guardian_log  # noqa: E402


def _base_input(**overrides) -> GuardianInput:
    data = {
        "fused_score": 0.72,
        "risk_level": "LOW",
        "confidence": 0.81,
        "classification_probability_up": 0.8,
        "sentiment_score": None,
        "spoofing_confidence": 0.1,
        "gnn_risk": None,
        "ppo_action": None,
        "fusion_action": "BUY",
        "current_position": "FLAT",
        "unrealized_pnl": 0.0,
        "daily_loss": 0.0,
        "daily_loss_limit": 0.03,
        "estimated_slippage": 0.0005,
        "slippage_threshold": 0.005,
    }
    data.update(overrides)
    return GuardianInput(**data)


def test_guardian_allows_when_all_checks_pass():
    payload = _base_input()
    result = GuardianAgent().validate(payload)

    assert result.decision == "ALLOW"
    assert result.reason == "all checks passed"
    assert result.fusion_action == "BUY"
    assert result.ppo_action is None


def test_guardian_blocks_in_rule_order():
    payload = _base_input(risk_level="HIGH", confidence=0.1, spoofing_confidence=0.9)
    result = GuardianAgent().validate(payload)

    assert result.decision == "BLOCK"
    assert result.reason == "high risk level"


def test_guardian_skips_missing_optional_fields():
    payload = _base_input(
        sentiment_score=None,
        spoofing_confidence=None,
        gnn_risk=None,
        daily_loss=None,
        estimated_slippage=None,
    )
    result = GuardianAgent().validate(payload)

    assert result.decision == "ALLOW"


def test_guardian_blocks_ppo_conflict():
    payload = _base_input(ppo_action="SELL", fusion_action="BUY")
    result = GuardianAgent().validate(payload)

    assert result.decision == "BLOCK"
    assert result.reason == "PPO conflicts with Fusion"


def test_guardian_forces_hold_when_already_long():
    payload = _base_input(fusion_action="BUY", current_position="LONG")
    result = GuardianAgent().validate(payload)

    assert result.decision == "FORCE_HOLD"
    assert result.reason == "already long"
    assert "position=LONG" in format_guardian_log(result, payload)


# ── Individual BLOCK branches ─────────────────────────────────────────────────

def test_guardian_blocks_low_confidence():
    payload = _base_input(confidence=0.50)
    result = GuardianAgent(confidence_threshold=0.69).validate(payload)

    assert result.decision == "BLOCK"
    assert result.reason == "confidence below threshold"


def test_guardian_blocks_spoofing():
    payload = _base_input(spoofing_confidence=0.85)
    result = GuardianAgent(spoofing_threshold=0.60).validate(payload)

    assert result.decision == "BLOCK"
    assert result.reason == "spoofing detected"


def test_guardian_blocks_daily_loss_limit():
    payload = _base_input(daily_loss=0.03, daily_loss_limit=0.03)
    result = GuardianAgent().validate(payload)

    assert result.decision == "BLOCK"
    assert result.reason == "daily loss limit reached"


def test_guardian_blocks_slippage_too_high():
    payload = _base_input(estimated_slippage=0.006, slippage_threshold=0.005)
    result = GuardianAgent().validate(payload)

    assert result.decision == "BLOCK"
    assert result.reason == "slippage too high"


def test_guardian_forces_hold_when_already_short():
    payload = _base_input(fusion_action="SELL", current_position="SHORT")
    result = GuardianAgent().validate(payload)

    assert result.decision == "FORCE_HOLD"
    assert result.reason == "already short"


# ── Boundary / near-miss values ───────────────────────────────────────────────

def test_guardian_allows_confidence_exactly_at_threshold():
    # confidence < threshold → BLOCK; at threshold is NOT blocked
    agent = GuardianAgent(confidence_threshold=0.69)
    payload = _base_input(confidence=0.69)
    result = agent.validate(payload)

    assert result.decision == "ALLOW"


def test_guardian_allows_slippage_exactly_at_threshold():
    # slippage > threshold → BLOCK; exactly equal is NOT blocked
    payload = _base_input(estimated_slippage=0.005, slippage_threshold=0.005)
    result = GuardianAgent().validate(payload)

    assert result.decision == "ALLOW"


# ── PPO edge cases ────────────────────────────────────────────────────────────

def test_guardian_allows_when_ppo_agrees_with_fusion():
    payload = _base_input(ppo_action="BUY", fusion_action="BUY")
    result = GuardianAgent().validate(payload)

    assert result.decision == "ALLOW"


def test_guardian_allows_when_ppo_is_hold():
    # PPO=HOLD should never block, even when fusion differs
    payload = _base_input(ppo_action="HOLD", fusion_action="BUY")
    result = GuardianAgent().validate(payload)

    assert result.decision == "ALLOW"


# ── Input normalisation ───────────────────────────────────────────────────────

def test_guardian_normalises_action_aliases():
    # "UP" should normalise to "BUY" — same as a direct BUY signal
    payload_up  = _base_input(fusion_action="UP",  ppo_action="UP")
    payload_buy = _base_input(fusion_action="BUY", ppo_action="BUY")

    result_up  = GuardianAgent().validate(payload_up)
    result_buy = GuardianAgent().validate(payload_buy)

    assert result_up.decision  == result_buy.decision
    assert result_up.fusion_action  == "BUY"


def test_guardian_normalises_position_aliases():
    # "BUY" and "LONG" both mean we are in a long position
    payload_buy  = _base_input(fusion_action="BUY", current_position="BUY")
    payload_long = _base_input(fusion_action="BUY", current_position="LONG")

    result_buy  = GuardianAgent().validate(payload_buy)
    result_long = GuardianAgent().validate(payload_long)

    assert result_buy.decision  == "FORCE_HOLD"
    assert result_long.decision == "FORCE_HOLD"


# ── format_guardian_log surface tests ────────────────────────────────────────

def test_format_log_allow_contains_key_fields():
    payload = _base_input()
    result  = GuardianAgent().validate(payload)
    log     = format_guardian_log(result, payload)

    assert "decision=ALLOW"  in log
    assert "fused_score="    in log
    assert "confidence="     in log
    assert "action=BUY"      in log


def test_format_log_block_contains_key_fields():
    payload = _base_input(risk_level="HIGH")
    result  = GuardianAgent().validate(payload)
    log     = format_guardian_log(result, payload)

    assert "decision=BLOCK"         in log
    assert "reason=high risk level" in log


def test_format_log_force_hold_includes_position():
    payload = _base_input(fusion_action="BUY", current_position="LONG")
    result  = GuardianAgent().validate(payload)
    log     = format_guardian_log(result, payload)

    assert "decision=FORCE_HOLD" in log
    assert "position=LONG"       in log
