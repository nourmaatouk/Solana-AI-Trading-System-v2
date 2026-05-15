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

from config import SYMBOL

# ── path setup ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent
FUSION_SRC = MODELS_DIR / "fusion" / "src"
if str(FUSION_SRC) not in sys.path:
    sys.path.insert(0, str(FUSION_SRC))

from fusion_aggregator import FusionAggregator          # noqa: E402
from fusion_contracts import (                           # noqa: E402
    ClassificationSignal,
    SentimentSignal,
    SpoofingSignal,
)

# ── I/O paths ────────────────────────────────────────────────────────────────
REALTIME_DIR        = BASE_DIR / "data" / "realtime"
SIGNAL_PATH         = REALTIME_DIR / "last_signal.json"
SPOOFING_PATH       = REALTIME_DIR / "spoofing_score.json"
SENTIMENT_FEAT_PATH = REALTIME_DIR / "latest_features_with_sentiment.csv"
OUTPUT_PATH         = REALTIME_DIR / "last_fused_decision.json"

_AGGREGATOR = FusionAggregator()


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
    except Exception:
        pass
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

        ts_raw = raw.get("latest_timestamp") or raw.get("generated_at_utc")
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            ts = datetime.now(timezone.utc)

        return ClassificationSignal(
            timestamp=ts,
            prediction=prob_up,
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

        if confidence > 0.6 or is_susp:
            severity = "high"
        elif confidence > 0.3:
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

        if score > 0.1:
            label = "POSITIVE"
        elif score < -0.1:
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


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("Realtime Fusion Aggregator")
    print("=" * 60)

    cls_sig   = _load_classification_signal()
    spoof_sig = _load_spoofing_signal()
    sent_sig  = _load_sentiment_signal()
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

    decision = _AGGREGATOR.fuse(
        classification_signal=cls_sig,
        spoofing_signal=spoof_sig,
        sentiment_signal=sent_sig,
    )

    ts_str = decision.timestamp.isoformat() if hasattr(decision.timestamp, "isoformat") else str(decision.timestamp)

    payload: dict = {
        "status":                    "ok",
        "generated_at_utc":          datetime.now(timezone.utc).isoformat(),
        "asset":                     cls_ctx.get("asset", SYMBOL),
        "signal_timestamp":          ts_str,
        "latest_timestamp":          cls_ctx.get("latest_timestamp", ts_str),
        "latest_close":              cls_ctx.get("latest_close"),
        "action":                    decision.action,
        "signal":                    decision.action,
        "fused_score":               float(decision.fused_score),
        "decision_score":            float(decision.fused_score),
        "confidence":                abs(float(decision.fused_score)),
        "risk_level":                decision.risk_level,
        "alignment_score":           float(decision.alignment_score),
        "classification_confidence": float(decision.classification_confidence),
        "classification_probability_up": cls_ctx.get("classification_probability_up"),
        "classification_probability_down": cls_ctx.get("classification_probability_down"),
        "spoofing_confidence":       float(decision.spoofing_confidence),
        "sentiment_relevance":       float(decision.sentiment_relevance),
        "reason":                    " | ".join(str(v) for v in decision.reasoning.values()),
        "reasoning":                 decision.reasoning,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Action        : {decision.action}")
    print(f"  Fused score   : {decision.fused_score:+.4f}")
    print(f"  Risk level    : {decision.risk_level}")
    print(f"  Alignment     : {decision.alignment_score:.3f}")
    print(f"  Reasoning     : {decision.reasoning}")
    print(f"\n  Saved to      : {OUTPUT_PATH}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
