"""Bridge from AI Director plans to the existing Production Director EDL.

Maps an :class:`~openclip.ai.edl.AIDirectorPlan` onto
:class:`openclip.director.models.Decision` records so AI proposals enter the
same review-gated pipeline as rule-planned decisions. Mapping rules:

- ``cut`` items become ``kind="cut"`` decisions; ``caption`` items become
  ``kind="caption"`` decisions.
- Every decision is created with ``status="pending"`` — **nothing is
  auto-accepted** — and carries the AI reason in ``rationale``.
- ``clip``, ``broll``, ``zoom``, ``audio`` and ``export`` have no v1
  ``DECISION_KINDS`` equivalent, so they are intentionally not mapped yet
  (documented stub); they remain available as review items via
  :func:`openclip.ai.review.plan_to_review_items`. Rendering behavior is
  unchanged.
"""

from __future__ import annotations

from openclip.ai.edl import AIDirectorPlan
from openclip.director.models import Decision


def ai_plan_to_director_decisions(plan: AIDirectorPlan) -> list[Decision]:
    """Convert a (typically normalized) AI plan into pending director decisions.

    Deterministic: decisions are sorted by timeline position. The plan itself
    is not modified.
    """
    decisions: list[Decision] = []
    for index, cut in enumerate(plan.cuts):
        decisions.append(
            Decision(
                id=cut.id or f"ai-cut-{index:03d}",
                kind="cut",
                source_span=(cut.span.start_seconds, cut.span.end_seconds),
                params={
                    "origin": "ai",
                    "provider": plan.provider,
                    "confidence": cut.confidence,
                    **cut.extra,
                },
                rationale=cut.reason or f"AI-proposed cut {cut.id}",
                status="pending",
            )
        )
    for index, caption in enumerate(plan.captions):
        decisions.append(
            Decision(
                id=caption.id or f"ai-caption-{index:03d}",
                kind="caption",
                source_span=(caption.span.start_seconds, caption.span.end_seconds),
                params={
                    "origin": "ai",
                    "provider": plan.provider,
                    "primary": caption.primary_text,
                    "support": caption.support_text,
                    "style": caption.style,
                    **caption.extra,
                },
                rationale=caption.primary_text or f"AI-proposed caption {caption.id}",
                status="pending",
            )
        )
    decisions.sort(key=lambda d: (d.source_span[0] if d.source_span else 0.0, d.id))
    return decisions
