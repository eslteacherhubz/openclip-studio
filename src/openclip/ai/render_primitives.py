"""Deterministic render primitives for AI Director decisions.

Pure builders (no IO, no network): they turn accepted AI decision params into
FFmpeg filter strings that the director executor appends to the final pass.

- ``zoom`` → animated center punch-in via FFmpeg's ``zoompan`` filter: the
  zoom factor is active only for the output frames inside the span and the
  output size stays the original resolution. ``strength`` s means a zoom
  factor of ``1 / (1 - s)`` (strength 0.5 → 2x punch-in; default 0.2 → 1.25x).
- ``audio_enhance`` → maps ``denoise`` / ``loudness_normalize`` flags onto the
  existing presets in :mod:`openclip.engine.enhance`.
"""

from __future__ import annotations

from openclip.engine.enhance import PRESETS

DEFAULT_ZOOM_STRENGTH = 0.2
DEFAULT_FPS = 25.0
_MAX_ZOOM_STRENGTH = 0.9


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
