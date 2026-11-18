"""Shared types for B-roll detection and sourcing.

This module breaks the circular import between broll_detection and broll_sourcing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ReasonType = Literal["visual_language", "concept_intro", "topic_shift"]
MatchType = Literal["local", "generated"]


@dataclass(frozen=True)
class BrollOpportunity:
    """One proposed B-roll insertion window on the source timeline."""

    start: float
    end: float
    concept_query: str
    keywords: tuple[str, ...]
    priority: float  # 0.0 - 1.0
    reason: ReasonType

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, object]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "concept_query": self.concept_query,
            "keywords": list(self.keywords),
            "priority": round(self.priority, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BrollSource:
    """Resolved B-roll asset source."""

    match_type: MatchType
    path: str
    confidence: float  # 0.0 - 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "match_type": self.match_type,
            "path": self.path,
            "confidence": round(self.confidence, 4),
        }


__all__ = [
    "BrollOpportunity",
    "BrollSource",
    "MatchType",
    "ReasonType",
]
