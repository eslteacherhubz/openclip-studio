"""Bridge from AI Director plans to the existing Production Director EDL.

Maps an :class:`~openclip.ai.edl.AIDirectorPlan` onto
:class:`openclip.director.models.Decision` records so AI proposals enter the
same review-gated pipeline as rule-planned decisions. Mapping rules:

- ``clip`` → ``kind="clip"``, ``cut`` → ``"cut"``, ``caption`` → ``"caption"``,
  ``broll`` → ``"broll"``, ``zoom`` → ``"zoom"``, ``audio`` →
  ``"audio_enhance"``, ``export`` → ``"export_config"``.
- Every decision is created with ``status="pending"`` — **nothing is
  auto-accepted** — and carries the AI reason in ``rationale`` plus
  ``params["origin"] = "ai"`` and the provider name.
- The new AI kinds (clip/broll/zoom/audio_enhance/export_config) have no
  render primitive yet: the executor safely skips accepted ones with a
  warning instead of crashing. ``cut``/``caption`` rendering is unchanged.
"""

from __future__ import annotations

from openclip.ai.edl import AIDirectorPlan
from openclip.director.models import Decision


def ai_plan_to_director_decisions(plan: AIDirectorPlan) -> list[Decision]:
    """Convert a (typically normalized) AI plan into pending director decisions.

    Deterministic: decisions are sorted by timeline position (span-less
    audio/export decisions last). The plan itself is not modified.
    """
    decisions: list[Decision] = []

    def _params(extra: dict[str, object], **fields: object) -> dict[str, object]:
        return {"origin": "ai", "provider": plan.provider, **fields, **extra}

    for index, clip in enumerate(plan.clips):
        decisions.append(
            Decision(
                id=clip.id or f"ai-clip-{index:03d}",
                kind="clip",
                source_span=(clip.span.start_seconds, clip.span.end_seconds),
                params=_params(clip.extra, title=clip.title, score=clip.score),
                rationale=clip.reason or f"AI-proposed clip {clip.id or index}",
                status="pending",
            )
        )
    for index, cut in enumerate(plan.cuts):
        decisions.append(
            Decision(
                id=cut.id or f"ai-cut-{index:03d}",
                kind="cut",
                source_span=(cut.span.start_seconds, cut.span.end_seconds),
                params=_params(cut.extra, confidence=cut.confidence),
                rationale=cut.reason or f"AI-proposed cut {cut.id or index}",
                status="pending",
            )
        )
    for index, caption in enumerate(plan.captions):
        decisions.append(
            Decision(
                id=caption.id or f"ai-caption-{index:03d}",
                kind="caption",
                source_span=(caption.span.start_seconds, caption.span.end_seconds),
                params=_params(
                    caption.extra,
                    primary=caption.primary_text,
                    support=caption.support_text,
                    style=caption.style,
                ),
                rationale=caption.primary_text or f"AI-proposed caption {caption.id or index}",
                status="pending",
            )
        )
    for index, broll in enumerate(plan.broll):
        decisions.append(
            Decision(
                id=broll.id or f"ai-broll-{index:03d}",
                kind="broll",
                source_span=(broll.span.start_seconds, broll.span.end_seconds),
                params=_params(
                    broll.extra, query=broll.query, source=broll.source, mode=broll.mode
                ),
                rationale=broll.query or f"AI-proposed B-roll {broll.id or index}",
                status="pending",
            )
        )
    for index, zoom in enumerate(plan.zooms):
        decisions.append(
            Decision(
                id=zoom.id or f"ai-zoom-{index:03d}",
                kind="zoom",
                source_span=(zoom.span.start_seconds, zoom.span.end_seconds),
                params=_params(zoom.extra, mode=zoom.mode, strength=zoom.strength),
                rationale=f"AI-proposed zoom {zoom.id or index}",
                status="pending",
            )
        )
    if plan.audio is not None:
        audio = plan.audio
        decisions.append(
            Decision(
                id="ai-audio-000",
                kind="audio_enhance",
                params=_params(
                    audio.extra,
                    enabled=audio.enabled,
                    denoise=audio.denoise,
                    loudness_normalize=audio.loudness_normalize,
                ),
                rationale=(
                    "AI-proposed audio enhancement "
                    f"(enabled={audio.enabled}, denoise={audio.denoise}, "
                    f"loudness_normalize={audio.loudness_normalize})"
                ),
                status="pending",
            )
        )
    if plan.export is not None:
        export = plan.export
        decisions.append(
            Decision(
                id="ai-export-000",
                kind="export_config",
                params=_params(
                    export.extra,
                    format=export.format,
                    width=export.width,
                    height=export.height,
                    fps=export.fps,
                    vertical=export.vertical,
                ),
                rationale=(
                    "AI-proposed export settings "
                    f"(format={export.format}, {export.width}x{export.height}"
                    f"@{export.fps}, vertical={export.vertical})"
                ),
                status="pending",
            )
        )
    decisions.sort(
        key=lambda d: (
            d.source_span[0] if d.source_span else float("inf"),
            d.kind,
            d.id,
        )
    )
    return decisions
