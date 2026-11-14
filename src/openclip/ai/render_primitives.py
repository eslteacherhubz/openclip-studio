"""Deterministic render primitives for AI Director decisions.

Pure builders (no IO, no network): they turn accepted AI decision params into
FFmpeg filter strings that the director executor appends to the final pass.

- ``zoom`` → animated center punch-in via FFmpeg's ``zoompan`` filter: the
  zoom factor is active only for the output frames inside the span and the
  output size stays the original resolution. ``strength`` s means a zoom
  factor of ``1 / (1 - s)`` (strength 0.5 → 2x punch-in; default 0.2 → 1.25x).
- ``audio_enhance`` → maps ``denoise`` / ``loudness_normalize`` flags onto the
  existing presets in :mod:`openclip.engine.enhance`.
- ``broll`` → `build_broll_filter` builds the `overlay` filter_complex segment
  (asset scaled/padded to the base resolution, `enable='between(t,…)'`) that
  layers the B-roll over the base video but under burned-in captions.
- ``export_config`` → `build_export_config_args` maps width/height/fps/vertical
  to a final-pass scale/crop filter (9:16 center-crop when ``vertical``) plus
  an output ``-r`` fps.
- transcript captions → `build_caption_filter` builds self-contained `drawtext`
  filters (no temp .srt/.ass files) for the kept transcript blocks, bilingual
  stacked (original above translation), white text with black outline,
  bottom-center with safe margins.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from openclip.director.models import Decision
from openclip.engine.enhance import PRESETS

logger = logging.getLogger(__name__)

DEFAULT_ZOOM_STRENGTH = 0.2
DEFAULT_FPS = 25.0
_MAX_ZOOM_STRENGTH = 0.9
_VERTICAL_ASPECT = 9 / 16


def build_zoom_vf(
    export_span: tuple[float, float],
    out_width: int,
    out_height: int,
    strength: float | None = None,
    fps: float = DEFAULT_FPS,
) -> str:
    """FFmpeg ``zoompan`` punch-in filter for one zoom span.

    ``export_span`` must already be in export-timeline time (mapped through
    accepted cuts). The span is converted to an output-frame range via ``fps``;
    outside the range the zoom factor is 1 (frame untouched), so the rest of
    the video is pixel-identical to a render without the zoom.
    """
    s = DEFAULT_ZOOM_STRENGTH if strength is None else strength
    s = min(max(s, 0.0), _MAX_ZOOM_STRENGTH)
    zoom = 1.0 / (1.0 - s)
    start, end = export_span
    frame_a = max(0, int(start * fps))
    frame_b = max(frame_a + 1, int(end * fps))
    return (
        f"zoompan=z='if(between(on,{frame_a},{frame_b}),{zoom:.4f},1)'"
        f":x='iw/2-(iw/zoom)/2':y='ih/2-(ih/zoom)/2'"
        f":d=1:s={out_width}x{out_height}:fps={fps:.6f}"
    )


def audio_chain_for_enhancement(
    enabled: bool,
    denoise: bool | None,
    loudness_normalize: bool | None,
) -> str | None:
    """Map an ``audio_enhance`` decision onto the existing audio presets.

    Returns the ``-af`` chain, or None when disabled or no flag selected.
    """
    if not enabled:
        return None
    filters: list[str] = []
    if denoise:
        filters.extend(PRESETS["denoise"])
    if loudness_normalize:
        filters.extend(PRESETS["normalize"])
    return ",".join(filters) if filters else None


# ---------------------------------------------------------------------------
# B-roll overlay
# ---------------------------------------------------------------------------


def build_broll_filter(
    decision: Decision,
    base_resolution: tuple[int, int],
    input_index: int = 1,
    prev_label: str = "base0",
) -> tuple[str, str]:
    """Build the filter_complex segment for one accepted ``broll`` decision.

    Returns ``(segment, out_label)``: the B-roll asset (input ``input_index``)
    is scaled to fit inside the base resolution (aspect ratio preserved) and
    padded to full frame, then overlaid at the top-left corner (z-order: above
    the base video, below any caption burn which is applied earlier in the
    main chain) only during the decision span. Times must already be in
    export-timeline coordinates. Raises ``ValueError`` when the decision has
    no usable asset path or span.
    """
    asset = decision.params.get("asset")
    if not isinstance(asset, str) or not asset.strip():
        raise ValueError(f"broll decision {decision.id!r} has no 'asset' path in params")
    if decision.source_span is None:
        raise ValueError(f"broll decision {decision.id!r} has no source_span")
    base_w, base_h = base_resolution
    start, end = decision.source_span
    out_label = f"bm{input_index}"
    segment = (
        f"[{input_index}:v]scale={base_w}:{base_h}:force_original_aspect_ratio=decrease,"
        f"pad={base_w}:{base_h}:(ow-iw)/2:(oh-ih)/2[bo{input_index}];"
        f"[{prev_label}][bo{input_index}]overlay=0:0"
        f":enable='between(t,{start:.3f},{end:.3f})'[{out_label}]"
    )
    return segment, out_label


# ---------------------------------------------------------------------------
# Export configuration
# ---------------------------------------------------------------------------


@dataclass
class ExportConfigArgs:
    """Resolved export settings for the final render pass."""

    vf: str | None  # final-pass video filter (scale / vertical crop+scale)
    fps: float | None  # output frame rate (-r)
    width: int | None
    height: int | None
    vertical: bool


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number > 0 else None


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0 else None


def build_export_vf(width: int | None, height: int | None, vertical: bool) -> str | None:
    """Final-pass video filter for an export config (None = keep resolution)."""
    if width is None or height is None:
        return None
    if vertical:
        # 9:16 shorts: center-crop the source to 9:16 (even width for yuv420p),
        # then scale to the target resolution.
        return (
            f"crop=w=trunc(ih*{_VERTICAL_ASPECT:.6f}/2)*2:h=ih:x=(iw-ow)/2:y=0,"
            f"scale={width}:{height}"
        )
    return f"scale={width}:{height}"


def build_export_config_args(decision: Decision) -> ExportConfigArgs:
    """Map an accepted ``export_config`` decision to final-pass arguments."""
    params = decision.params
    width = _positive_int(params.get("width"))
    height = _positive_int(params.get("height"))
    fps = _positive_float(params.get("fps"))
    vertical = params.get("vertical") is True
    return ExportConfigArgs(
        vf=build_export_vf(width, height, vertical),
        fps=fps,
        width=width,
        height=height,
        vertical=vertical,
    )


# ---------------------------------------------------------------------------
# Transcript captions (burned-in, self-contained drawtext chain)
# ---------------------------------------------------------------------------


@dataclass
class CaptionBlock:
    """One kept transcript block to burn in (times in export timeline)."""

    text: str
    start_s: float
    end_s: float
    translation: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object], context: str = "caption") -> CaptionBlock:
        text = str(data.get("text") or data.get("primary") or "")
        translation_raw = data.get("translation") or data.get("support")
        start = data.get("start", data.get("start_s"))
        end = data.get("end", data.get("end_s"))
        if not text.strip():
            raise ValueError(f"{context}: empty text")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not (0.0 <= float(start) < float(end))
        ):
            raise ValueError(f"{context}: invalid timestamps {start!r}..{end!r}")
        return cls(
            text=text,
            start_s=float(start),
            end_s=float(end),
            translation=str(translation_raw) if isinstance(translation_raw, str) else None,
        )


def _escape_drawtext(text: str) -> str:
    """Escape text for a drawtext `text=` option and flatten newlines."""
    flattened = " ".join(text.split())
    for char in ("\\", "'", ":", ",", ";", "[", "]", "%"):
        flattened = flattened.replace(char, "\\" + char)
    return flattened


def build_caption_filter(
    kept_transcript_blocks: Sequence[CaptionBlock | dict[str, object]],
    target_resolution: tuple[int, int],
) -> str | None:
    """Build a deterministic `drawtext` chain burning the kept caption blocks.

    Each block becomes one `drawtext` (original language) plus, when a
    translation is present, a second stacked below it (target language).
    Styling: white text, black outline, bottom-center, safe margin. Times are
    export-timeline seconds. Blocks with empty text or invalid timestamps are
    skipped with a warning. Returns None when nothing is burnable. No temp
    files — the whole chain is inline.
    """
    _width, height = target_resolution
    fontsize = max(12, int(height * 0.06))
    margin = max(8, int(height * 0.05))
    filters: list[str] = []
    for index, raw in enumerate(kept_transcript_blocks):
        try:
            block = raw if isinstance(raw, CaptionBlock) else CaptionBlock.from_dict(raw)
        except ValueError as exc:
            logger.warning("caption block %d skipped: %s", index, exc)
            continue
        enable = f"enable='between(t,{block.start_s:.3f},{block.end_s:.3f})'"
        base = (
            f"drawtext=font='Sans':text='{{}}'"
            f":fontsize={fontsize}:fontcolor=white:borderw=2:bordercolor=black"
            f":x=(w-text_w)/2"
        )
        if block.translation and block.translation.strip():
            # stacked: original on top, translation below it
            y_top = height - 2 * fontsize - margin
            y_bottom = height - fontsize - margin
            filters.append(
                base.replace("{}", _escape_drawtext(block.text), 1)
                + f":y={y_top}:{enable}"
            )
            filters.append(
                base.replace("{}", _escape_drawtext(block.translation), 1)
                + f":y={y_bottom}:{enable}"
            )
        else:
            filters.append(
                base.replace("{}", _escape_drawtext(block.text), 1)
                + f":y={height - fontsize - margin}:{enable}"
            )
    return ",".join(filters) if filters else None
