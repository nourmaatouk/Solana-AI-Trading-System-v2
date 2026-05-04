"""Backward-compatible NLP event contract wrappers over the shared Layer 2 schema."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict

import pandas as pd

from .layer2_contract import (
    ARTICLE_DEFAULTS,
    normalize_layer2_article_frame,
    normalize_layer2_article_record,
)


DEFAULT_NLP_EVENT_VALUES: Dict[str, Any] = dict(ARTICLE_DEFAULTS)


@dataclass(frozen=True)
class NLPEventContract:
    timestamp: Any = pd.NaT
    source: str = "unknown"
    feed_source: str = ""
    title: str = ""
    url: str = ""
    cleaned_text: str = ""
    relevance_score: float = 0.0
    event_type: str = "general"
    event_weight: float = 0.5
    event_signal: float = 0.0
    high_impact_event: int = 0
    negative_prob: float = 0.0
    neutral_prob: float = 0.0
    positive_prob: float = 0.0
    sentiment_score: float = 0.0
    confidence: float = 0.0
    sentiment_label: str = "neutral"

    @classmethod
    def normalize(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """Return a schema-complete NLP event dictionary."""
        return normalize_layer2_article_record(record)

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "NLPEventContract":
        return cls(**cls.normalize(record))

    def to_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def normalize_nlp_event(record: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_layer2_article_record(record)


def normalize_nlp_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an NLP dataframe to the shared contract columns."""
    return normalize_layer2_article_frame(df)
