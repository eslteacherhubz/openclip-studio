"""MP4 export renderer.

Strategy (proven primitives, same ones Shotcut/Kdenlive use):

1. Render each *kept* interval as an accurately re-encoded segment
   (fast input seek + re-encode = frame-accurate cut points).
2. Join segments with the concat demuxer (`-c copy`) into a master file.
3. One final pass burns animated captions (libass) and/or applies the audio
   enhancement chain, re-encoding only what actually changed.

Caption event times are mapped from the source timeline into the exported
(post-cut) timeline, so burned text stays in sync after cuts.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openclip.engine.align import Alignment, LineAlignment
from openclip.engine.captions import build_ass
from openclip.engine.cutplan import kept_intervals
from openclip.engine.enhance import build_audio_filter
from openclip.engine.ffmpeg import (
    FFmpegError,
    MediaInfo,
    escape_filter_path,
    run_ffmpeg,
)
from openclip.engine.project import Project

ProgressFn = Callable[[str, int, int], None]


@dataclass
class RenderOptions:
    crf: int = 20
    x264_preset: str = "veryfast"
    burn_captions: bool = True
    audio_preset: str | None = None  # None → use project.audio_preset
    keep_workdir: bool = False


@dataclass
class RenderResult:
    out: Path
    kept_duration_s: float
    segments: int
    cuts_applied: int
    captions_burned: bool
    audio_enhanced: bool


def map_span(
    start_s: float,
    end_s: float,
    cut_spans: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """Map a [start, end] span from source to exported timeline.

    Each endpoint is shifted by the total duration of accepted cuts that lie
    before it, so a cut *inside* the span compresses it (caption 0–10s with a
    removed 3–6s interval maps to 0–7s). Spans fully inside a cut (or clipped
    to nothing) return None. Overlapping/unsorted cut lists are normalized.
    """
    if end_s <= start_s:
        return None
    cuts = sorted(cut_spans)
    s, e = start_s, end_s
    for cs, ce in cuts:
        if ce <= s or cs >= e:
            continue  # cut entirely outside the span
        if cs <= s and ce >= e:
            return None  # span fully inside this cut
        if cs <= s < ce:
            s = ce  # span starts inside the cut: resume at the cut's end
        elif cs < e <= ce:
            e = cs  # span ends inside the cut: stop at the cut's start
        if e - s <= 1e-9:
            return None

    def removed_before(t: float) -> float:
        return sum(ce - cs for cs, ce in cuts if ce <= t)

    return s - removed_before(s), e - removed_before(e)


def map_alignment(alignment: Alignment, cut_spans: list[tuple[float, float]]) -> Alignment:
    """Map every aligned line into the exported timeline, dropping cut lines."""
    mapped: list[LineAlignment] = []
    for ln in alignment.lines:
        if ln.confidence <= 0.0:
            continue
        span = map_span(ln.start_s, ln.end_s, cut_spans)
        if span is None:
            continue
        s, e = span
        mapped.append(
            LineAlignment(
                index=ln.index,
                start_s=s,
                end_s=e,
                confidence=ln.confidence,
                matched_words=ln.matched_words,
            )
        )
    return Alignment(lines=mapped, transcriber=alignment.transcriber)


def build_export_ass(
    project: Project,
    media: MediaInfo,
    cut_spans: list[tuple[float, float]],
) -> str | None:
    """Build the caption ASS document (export-timeline times) for a project.

    Public so the Production Director can compose captions together with its
    own overlay events into one ASS document. Returns None when there is
    nothing to burn.
    """
    if project.alignment is None or len(project.script) == 0:
        return None
    from openclip.engine.align import alignment_from_dict

    alignment = alignment_from_dict(project.alignment)
    mapped = map_alignment(alignment, cut_spans)
    return build_ass(
        project.script.to_dict(),
        mapped,
        media.width or 1280,
        media.height or 720,
        project.captions,
    )


def render_export(
    media: MediaInfo,
    kept: list[tuple[float, float]],
    cut_spans: list[tuple[float, float]],
    ass_text: str | None,
    audio_chain: str | None,
    out_path: str | Path,
    opts: RenderOptions,
    progress: ProgressFn | None = None,
) -> RenderResult:
    """Execute kept intervals + one final pass (captions/audio) deterministically.

    Shared by `render_project` and the director executor. `ass_text` must
    already be in export-timeline time.
    """
    out = Path(out_path)
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    kept_duration = sum(e - s for s, e in kept)

    workdir = Path(tempfile.mkdtemp(prefix="openclip-render-"))
    try:
        segments = _render_segments(media, kept, workdir, opts, progress)
        master = workdir / "master.mp4"
        _concat(segments, master, media.has_audio)

        need_video = ass_text is not None
        need_audio = audio_chain is not None
        if not need_video and not need_audio:
            shutil.move(str(master), str(out))
            captions_burned = False
        else:
            vf: list[str] = []
            if ass_text is not None:
                ass_path = workdir / "captions.ass"
                ass_path.write_text(ass_text, encoding="utf-8")
                vf.append(f"ass=filename='{escape_filter_path(ass_path)}'")
            _final_pass(
                master,
                out,
                vf_graph=",".join(vf) if vf else None,
                af_chain=audio_chain,
                crf=opts.crf,
                preset=opts.x264_preset,
                has_audio=media.has_audio,
            )
            captions_burned = True
        return RenderResult(
            out=out,
            kept_duration_s=kept_duration,
            segments=len(segments),
            cuts_applied=len(cut_spans),
            captions_burned=captions_burned,
            audio_enhanced=audio_chain is not None,
        )
    finally:
        if not opts.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def render_project(
    project: Project,
    media: MediaInfo,
    out_path: str | Path,
    options: RenderOptions | None = None,
    progress: ProgressFn | None = None,
) -> RenderResult:
    """Render the project to an MP4 at `out_path`."""
    opts = options or RenderOptions()

    cuts = project.cut_plan.accepted_cuts() if project.cut_plan else []
    cut_spans = sorted((c.start_s, c.end_s) for c in cuts)
    kept = kept_intervals(cuts, media.duration_s)
    if not kept:
        kept = [(0.0, media.duration_s)]
        cut_spans = []

    audio_preset = opts.audio_preset or project.audio_preset
    audio_chain = build_audio_filter(audio_preset) if media.has_audio else None

    ass_text: str | None = None
    if opts.burn_captions:
        ass_text = build_export_ass(project, media, cut_spans)

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


def _render_segments(
    media: MediaInfo,
    kept: list[tuple[float, float]],
    workdir: Path,
    opts: RenderOptions,
    progress: ProgressFn | None,
) -> list[Path]:
    segs: list[Path] = []
    n = len(kept)
    for i, (s, e) in enumerate(kept):
        dur = e - s
        if dur <= 0.01:
            continue
        seg = workdir / f"seg_{i:03d}.mp4"
        args = ["-y", "-ss", f"{s:.3f}", "-i", str(media.path), "-t", f"{dur:.3f}"]
        if media.has_video:
            args += ["-map", "0:v:0"]
        if media.has_audio:
            args += ["-map", "0:a:0"]
        else:
            args += ["-an"]
        args += [
            "-c:v",
            "libx264",
            "-preset",
            opts.x264_preset,
            "-crf",
            str(opts.crf),
            "-pix_fmt",
            "yuv420p",
        ]
        if media.has_audio:
            args += ["-c:a", "aac", "-b:a", "128k"]
        args.append(str(seg))
        run_ffmpeg(args, timeout=3600)
        segs.append(seg)
        if progress:
            progress("segments", i + 1, n)
    if not segs:
        raise FFmpegError("Nothing to render: the cut plan removed all content")
    return segs


def _concat(segments: list[Path], master: Path, has_audio: bool) -> None:
    listing = master.parent / "concat.txt"
    listing.write_text(
        "\n".join(f"file '{s!s}'" for s in segments), encoding="utf-8"
    )
    run_ffmpeg(
        ["-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(master)],
        timeout=600,
    )


def _final_pass(
    master: Path,
    out: Path,
    vf_graph: str | None,
    af_chain: str | None,
    crf: int,
    preset: str,
    has_audio: bool,
) -> None:
    args = ["-y", "-i", str(master)]
    if vf_graph:
        args += ["-vf", vf_graph]
        args += [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        args += ["-c:v", "copy"]
    if has_audio:
        if af_chain:
            args += ["-af", af_chain, "-c:a", "aac", "-b:a", "160k"]
        else:
            args += ["-c:a", "copy"]
    else:
        args += ["-an"]
    args += ["-movflags", "+faststart", str(out)]
    run_ffmpeg(args, timeout=3600)
