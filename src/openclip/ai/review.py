"""Review items: turn an AI Director plan into human-reviewable records.

Every proposed clip, cut, caption, B-roll insert and zoom becomes an
:class:`AIReviewItem` with status ``PENDING`` — nothing is auto-accepted.
Output is deterministic: items are sorted by (start, kind, id).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openclip.ai.edl import AIDirectorPlan


class ReviewStatus(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class AIReviewItem:
    id: str
    kind: str
    title: str
    reason: str
    start_seconds: float
    end_seconds: float
    status: ReviewStatus = ReviewStatus.PENDING
    payload: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "reason": self.reason,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "status": self.status.value,
            "payload": dict(self.payload),
        }


def plan_to_review_items(plan: AIDirectorPlan) -> list[AIReviewItem]:
    """Convert every span-based plan item into a PENDING review item."""
    items: list[AIReviewItem] = []
    for clip in plan.clips:
        items.append(
            AIReviewItem(
                id=clip.id,
                kind="clip",
                title=clip.title or clip.id,
                reason=clip.reason,
                start_seconds=clip.span.start_seconds,
                end_seconds=clip.span.end_seconds,
                payload=clip.to_dict(),
            )
        )
    for cut in plan.cuts:
        items.append(
            AIReviewItem(
                id=cut.id,
                kind="cut",
                title=f"Cut {cut.id}",
                reason=cut.reason,
                start_seconds=cut.span.start_seconds,
                end_seconds=cut.span.end_seconds,
                payload=cut.to_dict(),
            )
        )
    for caption in plan.captions:
        items.append(
            AIReviewItem(
                id=caption.id,
                kind="caption",
                title=caption.primary_text or caption.id,
                reason=caption.primary_text,
                start_seconds=caption.span.start_seconds,
                end_seconds=caption.span.end_seconds,
                payload=caption.to_dict(),
            )
        )
    for broll in plan.broll:
        items.append(
            AIReviewItem(
                id=broll.id,
                kind="broll",
                title=broll.query or broll.id,
                reason=broll.query,
                start_seconds=broll.span.start_seconds,
                end_seconds=broll.span.end_seconds,
                payload=broll.to_dict(),
            )
        )
    for zoom in plan.zooms:
        items.append(
            AIReviewItem(
                id=zoom.id,
                kind="zoom",
                title=f"Zoom {zoom.id}",
                reason="ai-proposed zoom",
                start_seconds=zoom.span.start_seconds,
                end_seconds=zoom.span.end_seconds,
                payload=zoom.to_dict(),
            )
        )
    items.sort(key=lambda item: (item.start_seconds, item.kind, item.id))
    return items


def items_to_dict(items: list[AIReviewItem]) -> list[dict[str, Any]]:
    """JSON-serializable form of review items."""
    return [item.to_dict() for item in items]
