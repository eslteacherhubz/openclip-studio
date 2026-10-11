"""EDL executor: deterministic rendering of accepted decisions.

v1 executes the kinds the engine already has primitives for (cut, caption)
plus the two new ASS-overlay compositions (callout, section_title). Kinds
whose engine primitives do not exist yet (diagram_slot, color_preset,
reframe, background, emphasis) fail with an explicit NotImplementedError —
never a silent fallback.
"""

from __future__ import annotations

from pathlib import Path

from openclip.director.models import Decision, EditDecisionList, check_invariants
from openclip.engine.captions import format_ass_time
from openclip.engine.enhance import build_audio_filter
from openclip.engine.ffmpeg import MediaInfo
from openclip.engine.project import Project
from openclip.engine.render import (
    ProgressFn,
    RenderOptions,
    RenderResult,
    build_export_ass,
    render_export,
)

_UNSUPPORTED = ("diagram_slot", "color_preset", "reframe", "background", "emphasis")

_SAFE_MARGIN_V = 140  # callout/title band: below the caption block


def callout_event(decision: Decision, res_y: int) -> str:
    """Deterministic ASS event for a callout/section-title decision."""
    assert decision.source_span is not None
    start, end = (
        format_ass_time(decision.source_span[0]),
        format_ass_time(decision.source_span[1] + 0.05),
    )
    if decision.kind == "section_title":
        text = str(decision.params.get("title", ""))
        tag = r"{\fad(250,250)\fscx90\fscy90\t(0,250,\fscx100\fscy100)}"
        margin = _SAFE_MARGIN_V + 60
    else:
        text = str(decision.params.get("term", ""))
        tag = r"{\fad(150,150)\fscx80\fscy80\t(0,150,\fscx100\fscy100)}"
        margin = _SAFE_MARGIN_V
    safe_text = text.replace("\n", " ")
    return (
        f"Dialogue: 2,{start},{end},Callout,,0,0,{margin},,{tag}{safe_text}"
    )


def compose_ass(
    project: Project,
    media: MediaInfo,
    cut_spans: list[tuple[float, float]],
    edl: EditDecisionList,
) -> str | None:
    """Captions (existing system) + accepted overlay decisions in one ASS."""
    base = build_export_ass(project, media, cut_spans)
    overlays = [
        d
        for d in edl.accepted()
        if d.kind in ("callout", "section_title") and d.source_span is not None
    ]
    if base is None and not overlays:
        return None
    lines = (base or "").rstrip("\n").splitlines()
    if not any(line.startswith("Style: Callout") for line in lines):
        insert_at = next(
            (i for i, line in enumerate(lines) if line.startswith("[Events]")),
            len(lines),
        )
        callout_style = (
            "Style: Callout,Arial,22,&H0010F0FF,&H000000FF,&H00101010,&H80000000,"
            "0,0,0,0,100,100,0,0,1,2,1,2,40,40,0,1"
        )
        lines.insert(insert_at, callout_style)
    res_y = media.height or 720
    tail = [callout_event(d, res_y) for d in overlays]
    return "\n".join(lines + tail) + "\n"


def render_edl(
    project: Project,
    media: MediaInfo,
    edl: EditDecisionList,
    out_path: str | Path,
    options: RenderOptions | None = None,
    progress: ProgressFn | None = None,
    require_review: bool = True,
) -> RenderResult:
    """Execute accepted decisions deterministically.

    Raises when invariants are violated, when `require_review` is set and
    pending decisions remain, or when an unsupported kind was accepted.
    """
    problems = check_invariants(edl)
    if problems:
        raise ValueError("EDL invariants violated: " + "; ".join(problems))
    pending = [d.id for d in edl.decisions if d.status == "pending"]
    if require_review and pending:
        raise ValueError(
            f"{len(pending)} decisions pending review "
            f"(e.g. {pending[:3]}); review or pass require_review=False"
        )
    accepted = edl.accepted()
    unsupported = [d.id for d in accepted if d.kind in _UNSUPPORTED]
    if unsupported:
        raise NotImplementedError(
            "Decision kinds without deterministic engine primitives yet: "
            f"{unsupported} (missing components: overlay/colour/reframe "
            "renderers — see docs/PRODUCTION_DIRECTOR.md §8)"
        )

    opts = options or RenderOptions()
    cut_spans = sorted(
        (d.source_span[0], d.source_span[1])
        for d in accepted
        if d.kind == "cut" and d.source_span is not None
    )
    kept = kept_intervals_from_spans(cut_spans, media.duration_s)

    audio_preset = opts.audio_preset or project.audio_preset
    audio_chain = build_audio_filter(audio_preset) if media.has_audio else None

    ass_text = compose_ass(project, media, cut_spans, edl)
    return render_export(
        media=media,
        kept=kept,
        cut_spans=cut_spans,
        ass_text=ass_text,
        audio_chain=audio_chain,
        out_path=out_path,
        opts=opts,
        progress=progress,
    )


def kept_intervals_from_spans(
    cut_spans: list[tuple[float, float]], duration_s: float
) -> list[tuple[float, float]]:
    """Complement of the cut spans over [0, duration] (same math as the engine)."""
    from openclip.engine.cutplan import CutDecision
    from openclip.engine.cutplan import kept_intervals as engine_kept

    fake = [
        CutDecision(kind="silence", start_s=s, end_s=e, reason="", confidence=1.0)
        for s, e in cut_spans
    ]
    return engine_kept(fake, duration_s)
