"""Warp-based iris redirection.

Moves the iris toward the aperture center with a smooth displacement field
(the classic warp approach to gaze correction — no generative model, fully
local CPU math). Displacement is full-strength at the iris center and
cosine-fades to zero at ``reach`` px, so skin/lids barely move.

Inverse-mapped with cv2.remap for bicubic sub-pixel quality, then the
patch is feather-blended back into the frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from openclip.gaze.detect import EyeLocation


def compute_displacement(
    loc: EyeLocation,
    strength: float = 1.0,
    reach_factor: float = 1.35,
) -> tuple[float, float]:
    """Target displacement for the iris, in px, scaled by strength.

    The natural resting position keeps a small eccentricity so a fully
    centered stare does not look artificial (see research doc E5).
    """
    dx = (loc.eye_cx - loc.iris_cx) * strength
    dy = (loc.eye_cy - loc.iris_cy) * strength
    return dx, dy


def _falloff(dist: np.ndarray, iris_r: float, reach: float) -> np.ndarray:
    """1.0 inside the iris, cosine fade to 0.0 at reach."""
    inner = iris_r / max(reach, 1.0)
    t = np.clip(dist / max(reach, 1.0), 0.0, 1.0)
    inside = t <= inner
    denom = max(1.0 - inner, 1e-6)
    fade = 0.5 * (1.0 + np.cos(np.pi * (t - inner) / denom))
    return np.where(inside, 1.0, fade)


def warp_patch(
    patch_bgr: np.ndarray,
    loc: EyeLocation,
    strength: float = 1.0,
    reach_factor: float = 1.35,
) -> np.ndarray:
    """Warp one eye patch so the iris moves toward the aperture center."""
    dx, dy = compute_displacement(loc, strength, reach_factor)
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return patch_bgr
    h, w = patch_bgr.shape[:2]
    reach = max(loc.iris_r, 4.0) * reach_factor
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - loc.iris_cx) ** 2 + (yy - loc.iris_cy) ** 2)
    fall = _falloff(dist, loc.iris_r, reach)
    map_x = (xx - dx * fall).astype(np.float32)
    map_y = (yy - dy * fall).astype(np.float32)
    warped = cv2.remap(
        patch_bgr, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    # Feathered blend inside the aperture neighborhood so the patch edge
    # shows no seam.
    blend = np.clip(1.0 - (dist - reach) / max(reach, 1.0), 0.0, 1.0)[..., None]
    out = np.clip(
        warped.astype(np.float32) * blend + patch_bgr.astype(np.float32) * (1 - blend),
        0,
        255,
    )
    return np.asarray(out, dtype=np.uint8)


def warp_frame(
    frame_bgr: np.ndarray,
    locs: list[EyeLocation],
    strength: float = 1.0,
) -> np.ndarray:
    """Apply warp corrections for every detected eye in a full frame."""
    out = frame_bgr
    for loc in locs:
        pad = int(loc.iris_r * 3.5) + 4
        x0 = max(0, int(loc.iris_cx) - pad)
        y0 = max(0, int(loc.iris_cy) - pad)
        x1 = min(frame_bgr.shape[1], int(loc.iris_cx) + pad)
        y1 = min(frame_bgr.shape[0], int(loc.iris_cy) + pad)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        patch = out[y0:y1, x0:x1].copy()
        sub = EyeLocation(
            found=True,
            eye_cx=loc.eye_cx - x0,
            eye_cy=loc.eye_cy - y0,
            iris_cx=loc.iris_cx - x0,
            iris_cy=loc.iris_cy - y0,
            eye_w=loc.eye_w,
            iris_r=loc.iris_r,
        )
        out = out.copy()
        out[y0:y1, x0:x1] = warp_patch(patch, sub, strength)
    return out


class TemporalSmoother:
    """EMA smoothing of iris positions across frames (kills jitter)."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self._prev: list[tuple[float, float]] | None = None

    def smooth(self, locs: list[EyeLocation]) -> list[EyeLocation]:
        if self._prev is None or len(locs) != len(self._prev):
            self._prev = [(loc.iris_cx, loc.iris_cy) for loc in locs]
            return locs
        out: list[EyeLocation] = []
        for loc, (px, py) in zip(locs, self._prev, strict=True):
            sx = px + self.alpha * (loc.iris_cx - px)
            sy = py + self.alpha * (loc.iris_cy - py)
            out.append(
                EyeLocation(
                    found=loc.found,
                    eye_cx=loc.eye_cx,
                    eye_cy=loc.eye_cy,
                    iris_cx=sx,
                    iris_cy=sy,
                    eye_w=loc.eye_w,
                    iris_r=loc.iris_r,
                )
            )
        self._prev = [(loc.iris_cx, loc.iris_cy) for loc in out]
        return out
