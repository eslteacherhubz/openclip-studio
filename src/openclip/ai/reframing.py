"""Smart reframing: smooth subject-tracking crop for vertical (9:16) exports.

Generates a single inline FFmpeg ``crop`` filter whose ``x``/``y`` are
mathematical expressions in ``t`` — piecewise-linear interpolation between
tracking points, wrapped in ``clip(...)`` so the crop window can never leave
the source frame (no black bars). No keyframes, no temp files.

Tracking data is normalized (0..1 across width / down height). Today the data
comes from :func:`mock_tracking_data`; later a real MediaPipe/YOLO tracker can
feed the same :class:`TrackingPoint` list.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Default vertical (portrait) target aspect ratio, width / height.
DEFAULT_TARGET_ASPECT: float = 9 / 16


@dataclass(frozen=True)
class TrackingPoint:
    """One tracked subject position on the source timeline.

    ``x_center_norm`` / ``y_center_norm`` are the subject center in normalized
    source coordinates (0 = left/top edge, 1 = right/bottom edge).
    """

    t: float
    x_center_norm: float
    y_center_norm: float


def mock_tracking_data(duration_s: float = 10.0, steps: int = 4) -> list[TrackingPoint]:
    """Simulated subject drifting from the left side to the right side.

    Placeholder for a real tracker (MediaPipe pose / YOLO detection): a smooth
    left→right pan over ``duration_s`` at mid-height.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return [
        TrackingPoint(
            t=duration_s * i / steps,
            x_center_norm=0.2 + 0.6 * (i / steps),  # 0.2 → 0.8
            y_center_norm=0.5,
        )
        for i in range(steps + 1)
    ]


def _even(v: int) -> int:
    """Round down to an even value (yuv420p requires even crop dimensions)."""
    v = max(int(v), 2)
    return v if v % 2 == 0 else v - 1


def crop_size_for_aspect(
    src_w: int, src_h: int, target_aspect_ratio: float = DEFAULT_TARGET_ASPECT
) -> tuple[int, int]:
    """Largest centered crop window with the target aspect that fits the source.

    Both dimensions are even so the crop is safe for yuv420p encodes.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"invalid source resolution {(src_w, src_h)}")
    if target_aspect_ratio <= 0:
        raise ValueError(f"invalid aspect ratio {target_aspect_ratio}")
    crop_h = _even(src_h)
    crop_w = _even(round(src_h * target_aspect_ratio))
    if crop_w > src_w:  # source is narrower than the target aspect → crop width
        crop_w = _even(src_w)
        crop_h = _even(round(src_w / target_aspect_ratio))
    return crop_w, crop_h


def _normalize_points(tracking_data: Sequence[TrackingPoint]) -> list[TrackingPoint]:
    if not tracking_data:
        raise ValueError("tracking_data must contain at least one point")
    dedup: dict[float, TrackingPoint] = {}
    for p in tracking_data:
        if p.t < 0:
            raise ValueError(f"tracking point time must be >= 0, got {p.t}")
        dedup[p.t] = TrackingPoint(
            t=p.t,
            x_center_norm=min(max(p.x_center_norm, 0.0), 1.0),
            y_center_norm=min(max(p.y_center_norm, 0.0), 1.0),
        )
    return [dedup[t] for t in sorted(dedup)]


def anchor_at(
    points: Sequence[TrackingPoint],
    t: float,
    src_w: int,
    src_h: int,
    crop_w: int,
    crop_h: int,
) -> tuple[float, float]:
    """Clamped crop top-left (x, y) for the subject position at time ``t``.

    Linear interpolation between the bracketing tracking points; the result is
    clamped to ``[0, src-crop]`` on both axes so the window stays in bounds.
    """
    pts = _normalize_points(points)
    if t <= pts[0].t:
        cx, cy = pts[0].x_center_norm, pts[0].y_center_norm
    elif t >= pts[-1].t:
        cx, cy = pts[-1].x_center_norm, pts[-1].y_center_norm
    else:
        i = 0
        while pts[i + 1].t < t:
            i += 1
        p0, p1 = pts[i], pts[i + 1]
        f = (t - p0.t) / (p1.t - p0.t)
        cx = p0.x_center_norm + (p1.x_center_norm - p0.x_center_norm) * f
        cy = p0.y_center_norm + (p1.y_center_norm - p0.y_center_norm) * f
    x = min(max(cx * src_w - crop_w / 2, 0.0), float(src_w - crop_w))
    y = min(max(cy * src_h - crop_h / 2, 0.0), float(src_h - crop_h))
    return x, y


def _fmt(v: float) -> str:
    return f"{v:.6f}"


def _track_expr(values: list[float], times: list[float], max_v: float) -> str:
    """Piecewise-linear expression in ``t``, every branch clamped to [0, max_v]."""

    def clipped(inner: str) -> str:
        return f"clip({inner},0,{_fmt(max_v)})"

    if len(values) == 1:
        return clipped(_fmt(values[0]))
    expr = clipped(_fmt(values[-1]))  # t beyond the last point → hold last anchor
    for i in range(len(values) - 2, -1, -1):
        dt = times[i + 1] - times[i]
        seg = (
            f"{_fmt(values[i])} + ({_fmt(values[i + 1])}-{_fmt(values[i])})"
            f"*(t-{_fmt(times[i])})/{_fmt(dt)}"
        )
        expr = f"if(lt(t,{_fmt(times[i + 1])}),{clipped(seg)},{expr})"
    return expr


def build_smart_reframe_filter(
    tracking_data: Sequence[TrackingPoint],
    source_resolution: tuple[int, int],
    target_aspect_ratio: float = DEFAULT_TARGET_ASPECT,
) -> str:
    """Build one inline ``crop`` filter that smoothly follows the tracked subject.

    The crop window has the target aspect (default 9:16) and its top-left corner
    is a piecewise-linear function of ``t`` centered on the subject, clamped so
    the window never exceeds the source boundaries. Apply BEFORE the final
    export scale, AFTER the base cuts.
    """
    src_w, src_h = source_resolution
    crop_w, crop_h = crop_size_for_aspect(src_w, src_h, target_aspect_ratio)
    pts = _normalize_points(tracking_data)
    xs: list[float] = []
    ys: list[float] = []
    for p in pts:
        x, y = anchor_at(pts, p.t, src_w, src_h, crop_w, crop_h)
        xs.append(x)
        ys.append(y)
    times = [p.t for p in pts]
    x_expr = _track_expr(xs, times, float(src_w - crop_w))
    y_expr = _track_expr(ys, times, float(src_h - crop_h))
    return f"crop=w={crop_w}:h={crop_h}:x='{x_expr}':y='{y_expr}'"
