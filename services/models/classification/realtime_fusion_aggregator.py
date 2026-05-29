"""
realtime_fusion_aggregator.py — Lightweight multi-signal fusion for the
15-minute realtime pipeline.

Reads three JSON/CSV artifacts written by earlier pipeline steps:

  data/realtime/last_signal.json          — XGBoost classification signal
  data/realtime/spoofing_score.json       — TEN spoofing detector output
  data/realtime/latest_features_with_sentiment.csv — merged feature frame
                                            (last row contains sentiment cols)

Converts each to its fusion contract type, calls FusionAggregator, and writes:

  data/realtime/last_fused_decision.json

The script is intentionally self-contained and works even when some upstream
files are missing: absent signals are passed as None to the aggregator, which
gracefully fuses whatever is available.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIDENCE_THRESHOLD, SYMBOL, SIGNAL_STALE_MINUTES

# ── path setup ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent
FUSION_SRC = MODELS_DIR / "fusion" / "src"
if str(FUSION_SRC) not in sys.path:
    sys.path.insert(0, str(FUSION_SRC))

from fusion_aggregator import (                          # noqa: E402
    FusionAggregator,
    SPOOFING_SEVERITY_HIGH,
    SPOOFING_SEVERITY_MEDIUM,
    SPOOFING_HOLD_THRESHOLD,
    SENTIMENT_POSITIVE_THRESHOLD,
    SENTIMENT_NEGATIVE_THRESHOLD,
)
from fusion_contracts import (                           # noqa: E402
    ClassificationSignal,
    GNNSignal,
    SentimentSignal,
    SpoofingSignal,
)

# ── I/O paths ────────────────────────────────────────────────────────────────
REALTIME_DIR        = BASE_DIR / "data" / "realtime"
SIGNAL_PATH         = REALTIME_DIR / "last_signal.json"
SPOOFING_PATH       = REALTIME_DIR / "spoofing_score.json"
SENTIMENT_FEAT_PATH = REALTIME_DIR / "latest_features_with_sentiment.csv"
RL_SIGNAL_PATH      = REALTIME_DIR / "last_rl_signal.json"
GNN_SIGNAL_PATH     = REALTIME_DIR / "last_gnn_signal.json"
OUTPUT_PATH         = REALTIME_DIR / "last_fused_decision.json"

# ── RL weight ─────────────────────────────────────────────────────────────────
# Loaded from fusion config so it stays in sync with WEIGHT_RL constant.
import importlib.util as _ilu
_fc_path = FUSION_SRC.parent / "config.py"
_fc_spec = _ilu.spec_from_file_location("fusion.config", _fc_path)
_fc_mod  = _ilu.module_from_spec(_fc_spec)
_fc_spec.loader.exec_module(_fc_mod)
WEIGHT_RL = float(getattr(_fc_mod, "WEIGHT_RL", 0.00))
# When True, PPO is the decision-maker: the final action is taken straight from
# the agent rather than from the weighted fused score (see fusion/config.py).
RL_OWNS_ACTION = bool(getattr(_fc_mod, "RL_OWNS_ACTION", False))

# PPO emits BUY/SELL/NEUTRAL; map to the fusion action vocabulary (NEUTRAL→HOLD).
_PPO_SIGNAL_TO_ACTION = {"BUY": "BUY", "SELL": "SELL", "NEUTRAL": "HOLD"}

_AGGREGATOR = FusionAggregator()


def _rl_signal_is_fresh(rl_sig: dict | None) -> bool:
    """True when the RL signal carries a timestamp within SIGNAL_STALE_MINUTES.

    A missing/unparseable timestamp is treated as fresh (back-compat); only an
    explicitly old timestamp marks it stale, so PPO never drives the decision on
    outdated portfolio/market state.
    """
    if not rl_sig:
        return False
    ts_raw = rl_sig.get("generated_at_utc")
    if not ts_raw:
        return True
    try:
        ts = datetime.fromisoformat(str(ts_raw))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except Exception:
        return True
    age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60.0
    return age_min <= SIGNAL_STALE_MINUTES


# ── adapters ─────────────────────────────────────────────────────────────────

def _load_classification_context() -> dict:
    """Read non-model context needed by downstream consumers."""
    ctx = {"asset": SYMBOL}
    if not SIGNAL_PATH.exists():
        raw = {}
    else:
        raw = {}
        try:
            with open(SIGNAL_PATH) as f:
                raw = json.load(f)
            ctx.update({
                "asset": raw.get("asset", SYMBOL),
                "latest_timestamp": raw.get("latest_timestamp") or raw.get("latest_utc"),
                "latest_close": raw.get("latest_close"),
                "classification_probability_up": raw.get("probability_up"),
                "classification_probability_down": raw.get("probability_down"),
                "classification_reason": raw.get("reason"),
                "classification_signal": raw.get("signal"),
                "classification_confidence_threshold": raw.get("confidence_threshold"),
            })
        except Exception as exc:
            print(f"  [fusion] WARNING: failed to parse classification context: {exc}")

    if (not ctx.get("latest_timestamp") or ctx.get("latest_close") is None) and SENTIMENT_FEAT_PATH.exists():
        try:
            df = pd.read_csv(SENTIMENT_FEAT_PATH, usecols=lambda c: c in {"timestamp", "close"})
            if not df.empty:
                row = df.iloc[-1]
                ctx["latest_timestamp"] = ctx.get("latest_timestamp") or row.get("timestamp")
                if ctx.get("latest_close") is None:
                    close = row.get("close")
                    if not pd.isna(close):
                        ctx["latest_close"] = float(close)
        except Exception as exc:
            print(f"  [fusion] WARNING: failed to load feature price context: {exc}")
    try:
        if ctx.get("classification_probability_up") is None and raw.get("probability_down") is not None:
            ctx["classification_probability_up"] = 1.0 - float(raw["probability_down"])
    except Exception as exc:
        print(f"  [fusion] WARNING: prob_up fallback calculation failed: {exc}")
    return ctx

def _load_classification_signal() -> ClassificationSignal | None:
    """Read last_signal.json and adapt to ClassificationSignal."""
    if not SIGNAL_PATH.exists():
        print(f"  [fusion] last_signal.json not found — skipping classification signal")
        return None
    try:
        with open(SIGNAL_PATH) as f:
            raw = json.load(f)
        if raw.get("status") != "ok":
            print(f"  [fusion] last_signal status={raw.get('status')} — skipping")
            return None

        prob_up   = float(raw.get("probability_up", 0.5))
        prob_down = float(raw.get("probability_down", 0.5))
        raw_sig   = str(raw.get("signal", "NEUTRAL")).upper()

        if raw_sig in ("BUY", "UP"):
            signal_label = "BUY"
        elif raw_sig in ("SELL", "DOWN"):
            signal_label = "SELL"
        else:
            signal_label = "HOLD"

        confidence = max(prob_up, prob_down)
        # Use directional probability so SELL signals carry full weight in the
        # fused score formula (which does −prediction for SELL). Using prob_up
        # for SELL would give −0.187 instead of −0.813, systematically blocking
        # strong sell signals from breaching the SELL_THRESHOLD.
        if signal_label == "SELL":
            directional_pred = prob_down
        elif signal_label == "BUY":
            directional_pred = prob_up
        else:
            directional_pred = confidence

        ts_raw = raw.get("latest_timestamp") or raw.get("generated_at_utc")
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            ts = datetime.now(timezone.utc)

        return ClassificationSignal(
            timestamp=ts,
            prediction=directional_pred,
            signal=signal_label,
            confidence=confidence,
        )
    except Exception as exc:
        print(f"  [fusion] WARNING: failed to parse last_signal.json: {exc}")
        return None


def _load_spoofing_signal() -> SpoofingSignal | None:
    """Read spoofing_score.json and adapt to SpoofingSignal."""
    if not SPOOFING_PATH.exists():
        print(f"  [fusion] spoofing_score.json not found — skipping spoofing signal")
        return None
    try:
        with open(SPOOFING_PATH) as f:
            raw = json.load(f)
        if raw.get("status") not in ("ok", "suspicious"):
            print(f"  [fusion] spoofing status={raw.get('status')} — skipping")
            return None

        confidence  = float(raw.get("spoofing_probability", 0.0))
        branching   = float(raw.get("branching_ratio", 0.0))
        is_susp     = bool(raw.get("is_suspicious", False))
        n_events    = int(raw.get("n_events", 0))

        if confidence > SPOOFING_SEVERITY_HIGH or is_susp:
            severity = "high"
        elif confidence > SPOOFING_SEVERITY_MEDIUM:
            severity = "medium"
        else:
            severity = "low"

        summary = (
            f"branching_ratio={branching:.3f} n_events={n_events} "
            f"suspicious={is_susp}"
        )

        ts_raw = raw.get("generated_at_utc")
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            ts = datetime.now(timezone.utc)

        return SpoofingSignal(
            timestamp=ts,
            confidence=confidence,
            event_window_summary=summary,
            severity=severity,
        )
    except Exception as exc:
        print(f"  [fusion] WARNING: failed to parse spoofing_score.json: {exc}")
        return None


def _load_sentiment_signal() -> SentimentSignal | None:
    """Read the last row of latest_features_with_sentiment.csv and adapt to SentimentSignal."""
    if not SENTIMENT_FEAT_PATH.exists():
        print(f"  [fusion] latest_features_with_sentiment.csv not found — skipping sentiment")
        return None
    try:
        df = pd.read_csv(SENTIMENT_FEAT_PATH)
        if df.empty:
            return None
        row = df.iloc[-1]

        score = float(row.get("relevance_weighted_sentiment", row.get("sentiment_mean", 0.0)))
        if np.isnan(score):
            score = 0.0

        if score > SENTIMENT_POSITIVE_THRESHOLD:
            label = "POSITIVE"
        elif score < SENTIMENT_NEGATIVE_THRESHOLD:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        high_impact = bool(row.get("high_impact_event", 0))
        relevance   = float(row.get("max_confidence", row.get("positive_ratio", 0.5)))
        if np.isnan(relevance):
            relevance = 0.5

        n_articles = int(row.get("news_volume", row.get("gdelt_volume", 0)))

        ts_raw = row.get("timestamp")
        try:
            ts = pd.to_datetime(ts_raw, utc=True).to_pydatetime()
        except Exception:
            ts = datetime.now(timezone.utc)

        return SentimentSignal(
            timestamp=ts,
            sentiment_score=score,
            sentiment_label=label,
            event_type="news",
            high_impact_event=high_impact,
            relevance_score=relevance,
            n_articles=n_articles,
        )
    except Exception as exc:
        print(f"  [fusion] WARNING: failed to parse sentiment features: {exc}")
        return None


def _load_rl_signal() -> dict | None:
    """Read last_rl_signal.json; returns None if unavailable or NEUTRAL."""
    if not RL_SIGNAL_PATH.exists():
        return None
    try:
        raw = json.loads(RL_SIGNAL_PATH.read_text())
        sig = str(raw.get("signal", "NEUTRAL")).upper()
        conf = float(raw.get("confidence", 0.0))
        probs = raw.get("probabilities", {})
        buy_prob  = float(probs.get("BUY",  0.0))
        sell_prob = float(probs.get("SELL", 0.0))
        # Directional score in [-1, 1]: BUY pushes positive, SELL pushes negative
        if sig == "BUY":
            rl_score = buy_prob - 0.5
        elif sig == "SELL":
            rl_score = -(sell_prob - 0.5)
        else:
            rl_score = 0.0
        return {
            "signal": sig,
            "confidence": conf,
            "rl_score": rl_score,
            "probabilities": probs,
            "generated_at_utc": raw.get("generated_at_utc"),
        }
    except Exception as exc:
        print(f"  [fusion] WARNING: failed to parse last_rl_signal.json: {exc}")
        return None


def _load_gnn_signal() -> GNNSignal | None:
    """Read last_gnn_signal.json; returns None if unavailable."""
    if not GNN_SIGNAL_PATH.exists():
        return None
    try:
        raw = json.loads(GNN_SIGNAL_PATH.read_text())
        ts_str = raw.get("generated_at_utc", "")
        from datetime import datetime, timezone
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            ts = datetime.now(timezone.utc)
        return GNNSignal(
            timestamp=ts,
            anomaly_score=float(raw.get("anomaly_score", 0.0)),
            community_sentiment=float(raw.get("community_sentiment", 0.0)),
            liquidity_stress=float(raw.get("liquidity_stress", 0.0)),
        )
    except Exception as exc:
        print(f"  [fusion] WARNING: failed to parse last_gnn_signal.json: {exc}")
        return None


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Realtime Fusion Aggregator")
    print("=" * 60)

    cls_sig   = _load_classification_signal()
    spoof_sig = _load_spoofing_signal()
    sent_sig  = _load_sentiment_signal()
    rl_sig    = _load_rl_signal()
    gnn_sig   = _load_gnn_signal()
    cls_ctx   = _load_classification_context()

    if cls_sig is None and spoof_sig is None and sent_sig is None:
        print("  ERROR: no signals available for fusion — aborting.")
        payload = {
            "status": "no_signals",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "action": "HOLD",
            "fused_score": 0.0,
            "risk_level": "HIGH",
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        return 1

    print(f"  classification : {'ok' if cls_sig   else 'missing'}")
    print(f"  spoofing       : {'ok' if spoof_sig else 'missing'}")
    print(f"  sentiment      : {'ok' if sent_sig  else 'missing'}")
    print(f"  gnn_signal     : {'ok (anomaly=' + str(round(gnn_sig.anomaly_score, 3)) + ')' if gnn_sig else 'missing'}")
    print(f"  rl_signal      : {'ok (' + rl_sig['signal'] + ')' if rl_sig else 'missing'}")

    decision = _AGGREGATOR.fuse(
        classification_signal=cls_sig,
        spoofing_signal=spoof_sig,
        sentiment_signal=sent_sig,
        gnn_signal=gnn_sig,
    )

    # ── Decide the final action ──────────────────────────────────────────────
    # Two modes (toggled by RL_OWNS_ACTION in fusion/config.py):
    #
    #   RL_OWNS_ACTION=True  → PPO IS the decision-maker. The final BUY/SELL/HOLD
    #     is taken straight from the agent; XGBoost/FinBERT/TEN-GNN/GNN are
    #     advisory context only. The spoofing HOLD veto still overrides PPO, and
    #     a stale RL signal falls back to the fusion action so we never trade on
    #     outdated state. The downstream Guardian validates regardless.
    #
    #   RL_OWNS_ACTION=False → weighted-score fusion, PPO blended at WEIGHT_RL.
    base_fused_score = float(decision.fused_score)
    base_action      = decision.action
    decision_source  = "fusion"
    final_fused_score = base_fused_score
    final_action      = base_action
    final_confidence  = abs(base_fused_score)

    rl_fresh = _rl_signal_is_fresh(rl_sig)

    if RL_OWNS_ACTION and rl_sig is not None and rl_fresh:
        # PPO owns the action.
        rl_score          = float(rl_sig["rl_score"])
        final_fused_score = float(np.clip(rl_score, -1.0, 1.0))
        ppo_action        = _PPO_SIGNAL_TO_ACTION.get(rl_sig["signal"], "HOLD")
        # Safety: high spoofing confidence forces HOLD even when PPO says trade.
        if decision.spoofing_confidence > SPOOFING_HOLD_THRESHOLD:
            ppo_action = "HOLD"
        final_action     = ppo_action
        final_confidence = float(rl_sig["confidence"])
        decision_source  = "ppo"
    elif rl_sig is not None and not RL_OWNS_ACTION:
        # Legacy weighted blend: PPO nudges the fused score at WEIGHT_RL.
        rl_score          = float(rl_sig["rl_score"])
        final_fused_score = float(np.clip(
            (1.0 - WEIGHT_RL) * base_fused_score + WEIGHT_RL * rl_score, -1.0, 1.0,
        ))
        final_action      = _AGGREGATOR._score_to_action(final_fused_score, decision.spoofing_confidence)
        final_confidence  = abs(final_fused_score)
    elif RL_OWNS_ACTION and rl_sig is not None and not rl_fresh:
        # PPO is meant to decide but its signal is stale → fall back to fusion.
        decision_source = "fusion_rl_stale"

    ts_str = decision.timestamp.isoformat() if hasattr(decision.timestamp, "isoformat") else str(decision.timestamp)

    reasoning = dict(decision.reasoning)
    if rl_sig:
        tag = " [DECIDER]" if decision_source == "ppo" else (" [STALE→fusion]" if decision_source == "fusion_rl_stale" else "")
        reasoning["rl"] = (
            f"PPO {rl_sig['signal']}{tag} (confidence={rl_sig['confidence']:.3f}, "
            f"rl_score={rl_sig['rl_score']:+.3f})"
        )

    payload: dict = {
        "status":                    "ok",
        "generated_at_utc":          datetime.now(timezone.utc).isoformat(),
        "asset":                     cls_ctx.get("asset", SYMBOL),
        "signal_timestamp":          ts_str,
        "latest_timestamp":          cls_ctx.get("latest_timestamp", ts_str),
        "latest_close":              cls_ctx.get("latest_close"),
        "action":                    final_action,
        "signal":                    final_action,
        "decision_source":           decision_source,
        "fused_score":               float(final_fused_score),
        "decision_score":            float(final_fused_score),
        "confidence":                float(final_confidence),
        "risk_level":                decision.risk_level,
        "alignment_score":           float(decision.alignment_score),
        "classification_confidence": float(decision.classification_confidence),
        "classification_probability_up": cls_ctx.get("classification_probability_up"),
        "classification_probability_down": cls_ctx.get("classification_probability_down"),
        "spoofing_confidence":       float(decision.spoofing_confidence),
        "sentiment_relevance":       float(decision.sentiment_relevance),
        # Structured sentiment-event fields (consumed directly by the API/UI so
        # they no longer regex-parse the reasoning prose). None when no sentiment.
        "event_type":                sent_sig.event_type if sent_sig else None,
        "high_impact_event":         bool(sent_sig.high_impact_event) if sent_sig else False,
        "gnn_anomaly_score":         round(gnn_sig.anomaly_score, 4) if gnn_sig else None,
        "gnn_liquidity_stress":      round(gnn_sig.liquidity_stress, 4) if gnn_sig else None,
        "gnn_community_sentiment":   round(gnn_sig.community_sentiment, 4) if gnn_sig else None,
        "rl_signal":                 rl_sig["signal"] if rl_sig else None,
        "rl_confidence":             rl_sig["confidence"] if rl_sig else None,
        "confidence_gate":           CONFIDENCE_THRESHOLD,
        "reason":                    " | ".join(str(v) for v in reasoning.values()),
        "reasoning":                 reasoning,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Action        : {final_action}")
    print(f"  Fused score   : {final_fused_score:+.4f}  (base={base_fused_score:+.4f})")
    print(f"  Risk level    : {decision.risk_level}")
    print(f"  Alignment     : {decision.alignment_score:.3f}")
    print(f"  Reasoning     : {reasoning}")
    print(f"\n  Saved to      : {OUTPUT_PATH}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
