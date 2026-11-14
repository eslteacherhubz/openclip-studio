"""EDL executor: deterministic rendering of accepted decisions.

v1 executes the kinds the engine already has primitives for (cut, caption)
plus the two new ASS-overlay compositions (callout, section_title). Kinds
whose engine primitives do not exist yet (diagram_slot, color_preset,
reframe, background, emphasis) fail with an explicit NotImplementedError —
never a silent fallback.

Timeline contract: every visual decision carries a *source-timeline* span.
Before rendering, spans are mapped through ``map_span`` against the
normalized accepted cuts, so overlays land at export-timeline timestamps
(and overlays fully inside cuts are dropped). Captions render only for
``caption`` decisions the reviewer accepted, and only when
``RenderOptions.burn_captions`` is set (callouts/titles are governed by
their own decision status, not by burn_captions).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from openclip.ai.render_primitives import (
    audio_chain_for_enhancement,
    build_broll_filter,
    build_export_config_args,
    build_zoom_vf,
)
from openclip.director.models import Decision, EditDecisionList, check_invariants
from openclip.engine.align import Alignment, alignment_from_dict
from openclip.engine.captions import (
    build_ass_header,
    caption_events,
    format_ass_time,
)
from openclip.engine.enhance import build_audio_filter
from openclip.engine.ffmpeg import MediaInfo
from openclip.engine.project import Project
from openclip.engine.render import (
    BrollOverlay,
    ProgressFn,
    RenderOptions,
    RenderResult,
    map_span,
    normalize_cut_spans,
    render_export,
)

logger = logging.getLogger(__name__)

_UNSUPPORTED = ("diagram_slot", "color_preset", "reframe", "background", "emphasis")

#: AI Director kinds with no render primitive yet (currently none — all AI
#: kinds render or are safely skipped individually, e.g. broll with a missing
#: asset). Kept for forward compatibility; "cut"/"caption"/"callout"/
#: "section_title" rendering is untouched.
_AI_SKIPPED_KINDS: tuple[str, ...] = ()

_SAFE_MARGIN_V = 140  # callout/title band: below the caption block

# ASS PrimaryColour &HAABBGGRR: alpha 00, blue 0x10, green 0xF0, red 0xFF.
CALLOUT_RGB = (255, 240, 16)
_CALLOUT_STYLE = (
    "Style: Callout,Arial,22,&H0010F0FF,&H000000FF,&H00101010,&H80000000,"
    "0,0,0,0,100,100,0,0,1,2,1,2,40,40,0,1"
)


def callout_event(decision: Decision, export_span: tuple[float, float]) -> str:
    """Deterministic ASS event for a callout/section-title decision.

    `export_span` must already be in export-timeline time (mapped through
    ``map_span`` against the accepted cuts).
    """
    start, end = (
        format_ass_time(export_span[0]),
        format_ass_time(export_span[1] + 0.05),
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


def _accepted_caption_lines(
    project: Project,
    caption_decisions: list[Decision],
) -> tuple[list[dict[str, object]], Alignment]:
    """Script entries + alignment restricted to accepted caption decisions."""
    if project.alignment is None:
        return [], Alignment(lines=[], transcriber="none")
    alignment = alignment_from_dict(project.alignment)
    by_line = {ln.index: ln for ln in alignment.lines}
    script_by_index = {ln.index: ln for ln in project.script.lines}
    selected: list[int] = []
    for d in caption_decisions:
        if d.source_line is None:
            continue
        if d.source_line in by_line and d.source_line in script_by_index:
            selected.append(d.source_line)
    script_entries: list[dict[str, object]] = [
        {
            "index": i,
            "primary": script_by_index[i].primary,
            "support": script_by_index[i].support,
        }
        for i in selected
    ]
    selected_alignment = Alignment(
        lines=[by_line[i] for i in selected], transcriber=alignment.transcriber
    )
    return script_entries, selected_alignment


def compose_ass(
    project: Project,
    media: MediaInfo,
    cut_spans: list[tuple[float, float]],
    edl: EditDecisionList,
    burn_captions: bool = True,
) -> str | None:
    """Compose the export-timeline ASS document for accepted decisions.

    - caption decisions render only when accepted (review) AND
      ``burn_captions`` is set;
    - callout/section_title decisions render only when accepted, at
      export-timeline times mapped through the accepted cuts; overlays fully
      inside a cut are dropped;
    - returns None when nothing renders (fast path: no video re-encode).
    """
    cuts = normalize_cut_spans(cut_spans)
    accepted = edl.accepted()
    caption_decisions = [d for d in accepted if d.kind == "caption"]
    overlay_decisions = [
        d
        for d in accepted
        if d.kind in ("callout", "section_title") and d.source_span is not None
    ]
    if not caption_decisions and not overlay_decisions:
        return None

    res_x = media.width or 1280
    res_y = media.height or 720
    events: list[str] = []

    if burn_captions and caption_decisions:
        script_entries, selected = _accepted_caption_lines(project, caption_decisions)
        if selected.lines:
            from openclip.engine.render import map_alignment

            mapped = map_alignment(selected, cuts)
            events.extend(
                caption_events(script_entries, mapped, res_x, res_y, project.captions)
            )

    for d in overlay_decisions:
        assert d.source_span is not None
        export_span = map_span(d.source_span[0], d.source_span[1], cuts)
        if export_span is None:
            continue  # overlay fully removed by cuts
        events.append(callout_event(d, export_span))

    if not events:
        return None
    header = build_ass_header(res_x, res_y, project.captions, (_CALLOUT_STYLE,))
    return "\n".join([header, *events]) + "\n"


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
    skipped_ai = [d.id for d in accepted if d.kind in _AI_SKIPPED_KINDS]
    if skipped_ai:
        logger.warning(
            "Skipping %d accepted AI decision(s) with no render primitive yet "
            "(kinds %s): %s — export continues with cut/caption/overlay rendering only",
            len(skipped_ai),
            sorted({d.kind for d in accepted if d.kind in _AI_SKIPPED_KINDS}),
            skipped_ai,
        )

    opts = options or RenderOptions()
    base_res = (media.width or 1280, media.height or 720)
    cut_spans = normalize_cut_spans(
        [
            (d.source_span[0], d.source_span[1])
            for d in accepted
            if d.kind == "cut" and d.source_span is not None
        ]
    )
    clip_spans = normalize_cut_spans(
        [
            (d.source_span[0], d.source_span[1])
            for d in accepted
            if d.kind == "clip" and d.source_span is not None
        ]
    )
    if clip_spans:
        # "clip" = keep ONLY these spans: everything outside them is removed,
        # so the export timeline condenses to the clip spans (minus any cuts
        # inside them). Overlays are mapped through the combined removed set.
        outside = _complement_spans(clip_spans, media.duration_s)
        cut_spans = normalize_cut_spans(cut_spans + outside)
    kept = kept_intervals_from_spans(cut_spans, media.duration_s)

    # AI zoom decisions: map source-timeline spans to the export timeline and
    # build deterministic punch-in filters.
    video_filters: list[str] = []
    for d in accepted:
        if d.kind != "zoom" or d.source_span is None:
            continue
        export_span = map_span(d.source_span[0], d.source_span[1], cut_spans)
        if export_span is None:
            continue  # zoom outside clips / fully removed by cuts
        strength_raw = d.params.get("strength")
        strength = (
            float(strength_raw)
            if isinstance(strength_raw, (int, float)) and not isinstance(strength_raw, bool)
            else None
        )
        video_filters.append(
            build_zoom_vf(export_span, base_res[0], base_res[1], strength, media.fps or 25.0)
        )

    # AI export_config: apply target resolution (with 9:16 center-crop when
    # vertical) as the last video filter, and the target fps as output -r.
    # Computed BEFORE broll so overlays pad to the *target* resolution (the
    # export scale runs on the main chain first, then broll layers on top).
    out_fps: float | None = None
    target_res = base_res
    export_decisions = [d for d in accepted if d.kind == "export_config"]
    if export_decisions:
        cfg = build_export_config_args(export_decisions[-1])  # ordered; last wins
        if cfg.vf is not None:
            video_filters.append(cfg.vf)
        out_fps = cfg.fps
        if cfg.width is not None and cfg.height is not None:
            target_res = (cfg.width, cfg.height)

    # AI broll decisions: overlay the referenced asset during the mapped span,
    # scaled/padded to the target (post-export-config) resolution.
    # Missing/invalid assets are dropped with a warning, never crash the render.
    broll_overlays: list[BrollOverlay] = []
    prev_label = "base0"
    for index, d in enumerate(
        d for d in accepted if d.kind == "broll" and d.source_span is not None
    ):
        asset_raw = d.params.get("asset")
        asset_path = (
            Path(str(asset_raw)) if isinstance(asset_raw, str) and asset_raw.strip() else None
        )
        if asset_path is None or not asset_path.exists():
            logger.warning(
                "Skipping broll decision %s: asset %r is missing", d.id, asset_raw
            )
            continue
        span = d.source_span
        if span is None:
            continue
        export_span = map_span(span[0], span[1], cut_spans)
        if export_span is None:
            continue  # broll span outside clips / fully removed by cuts
        export_decision = replace(d, source_span=export_span)
        segment, out_label = build_broll_filter(export_decision, target_res, index + 1, prev_label)
        broll_overlays.append(
            BrollOverlay(
                asset=asset_path,
                start_s=export_span[0],
                end_s=export_span[1],
                segment=segment,
                out_label=out_label,
            )
        )
        prev_label = out_label

    audio_preset = opts.audio_preset or project.audio_preset
    audio_chain = build_audio_filter(audio_preset) if media.has_audio else None
    enhance_decisions = [d for d in accepted if d.kind == "audio_enhance"]
    if enhance_decisions and media.has_audio:
        last = enhance_decisions[-1]  # deterministic: ordered by span/kind/id
        chain = audio_chain_for_enhancement(
            bool(last.params.get("enabled", True)),
            last.params.get("denoise") if isinstance(last.params.get("denoise"), bool) else None,
            (
                last.params.get("loudness_normalize")
                if isinstance(last.params.get("loudness_normalize"), bool)
                else None
            ),
        )
        if chain is not None:
            audio_chain = chain

    ass_text = compose_ass(
        project, media, cut_spans, edl, burn_captions=opts.burn_captions
    )
    return render_export(
        media=media,
        kept=kept,
        cut_spans=cut_spans,
        ass_text=ass_text,
        audio_chain=audio_chain,
        out_path=out_path,
        opts=opts,
        progress=progress,
        video_filters=video_filters or None,
        broll=broll_overlays or None,
        out_fps=out_fps,
    )


def kept_intervals_from_spans(
    cut_spans: list[tuple[float, float]], duration_s: float
) -> list[tuple[float, float]]:
    """Complement of the (normalized) cut spans over [0, duration]."""
    from openclip.engine.cutplan import CutDecision
    from openclip.engine.cutplan import kept_intervals as engine_kept

    normalized = normalize_cut_spans(cut_spans)
    fake = [
        CutDecision(kind="silence", start_s=s, end_s=e, reason="", confidence=1.0)
        for s, e in normalized
    ]
    return engine_kept(fake, duration_s)


def _complement_spans(
    spans: list[tuple[float, float]], duration_s: float
) -> list[tuple[float, float]]:
    """Gaps between normalized `spans` over [0, duration] (i.e. what is NOT kept).

    ``kept_intervals_from_spans`` already returns the complement of its input,
    so this is a thin, explicit wrapper.
    """
    return kept_intervals_from_spans(spans, duration_s)
