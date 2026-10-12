"""Deterministic synthetic eye rig with exact ground truth.

Generates photometrically plausible eye patches (skin, lids, sclera, iris,
pupil, specular highlight, lash shadow, mild noise) with a known iris
offset, so correction quality can be measured without any private footage.
This is a development rig, NOT evidence about real-camera performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RigConfig:
    canvas_w: int = 200
    canvas_h: int = 100
    eye_cx: int = 100
    eye_cy: int = 52
    eye_w: int = 120  # aperture width (the metric normalizer)
    eye_h: int = 56
    iris_r: int = 17
    pupil_r: int = 7
    seed: int = 7


@dataclass
class RigEye:
    image: np.ndarray  # uint8 BGR
    ground_truth_dx: float  # iris offset from eye center, px
    ground_truth_dy: float
    eye_cx: float
    eye_cy: float
    eye_w: float
    iris_r: float


def _aperture_mask(cfg: RigConfig, w: int, h: int) -> np.ndarray:
    """Almond aperture: intersection of two half-ellipse parabolas."""
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float(cfg.eye_cx)
    cy = float(cfg.eye_cy)
    ex = cfg.eye_w / 2.0
    ey = cfg.eye_h / 2.0
    # upper lid: parabola opening down; lower lid: opening up
    inside = ((yy - cy) <= ey * (1 - ((xx - cx) / ex) ** 2)) & (
        (yy - cy) >= -ey * (1 - ((xx - cx) / ex) ** 2)
    )
    return np.asarray(inside, dtype=np.float32)


def render_eye(cfg: RigConfig, dx: float, dy: float) -> RigEye:
    """Render one eye with iris offset (dx, dy) from the aperture center."""
    rng = np.random.default_rng(cfg.seed)
    w, h = cfg.canvas_w, cfg.canvas_h
    img = np.zeros((h, w, 3), dtype=np.float32)

    # Skin: warm gradient + subtle horizontal texture
    base = np.linspace(0.86, 0.92, h, dtype=np.float32)[:, None]
    skin = np.empty((h, w, 3), dtype=np.float32)
    for c, k in enumerate((1.00, 0.86, 0.78)):  # B,G,R-ish tint (stored as BGR)
        skin[..., c] = (base * k) @ np.ones((1, w), dtype=np.float32)
    skin += rng.normal(0, 0.010, skin.shape).astype(np.float32)
    img[:] = np.clip(skin, 0, 1)

    mask = _aperture_mask(cfg, w, h)

    # Sclera: desaturated white with medial shadow
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sclera = np.full((h, w, 3), 0.93, dtype=np.float32)
    shade = 1.0 - 0.25 * np.clip((xx - cfg.eye_cx) / cfg.eye_w + 0.5, 0, 1) ** 2
    for c in range(3):
        sclera[..., c] *= shade
    img = img * (1 - mask[..., None]) + sclera * mask[..., None]

    # Iris + pupil, clipped to the aperture
    iris_cx = cfg.eye_cx + dx
    iris_cy = cfg.eye_cy + dy
    rr = (xx - iris_cx) ** 2 + (yy - iris_cy) ** 2
    iris_disk = rr <= cfg.iris_r**2
    pupil_disk = rr <= cfg.pupil_r**2
    # Iris: brown-blue mix with radial fibers
    rad = np.sqrt(rr)
    fiber = 0.5 + 0.5 * np.sin(rad * 1.7)
    iris_rgb = np.stack(
        (
            0.28 + 0.06 * fiber,  # R
            0.21 + 0.05 * fiber,  # G
            0.13 + 0.03 * fiber,  # B
        ),
        axis=-1,
    ).astype(np.float32)
    iris_rgb = iris_rgb[..., ::-1]  # store as BGR
    for c in range(3):
        layer = iris_rgb[..., c]
        img[..., c] = np.where(iris_disk & (mask > 0), layer, img[..., c])
    # Limbal ring (darker iris rim)
    rim = (rad >= cfg.iris_r - 2.5) & (rad <= cfg.iris_r)
    img = np.where(rim[..., None] & (mask[..., None] > 0), img * 0.55, img)
    # Pupil
    img = np.where(pupil_disk[..., None] & (mask[..., None] > 0), 0.02, img)
    # Specular highlight (top-left of pupil)
    spec = ((xx - (iris_cx - 4)) ** 2 + (yy - (iris_cy - 5)) ** 2) <= 2.5**2
    img = np.where(spec[..., None] & (mask[..., None] > 0), 0.95, img)

    # Upper-lash shadow just above the aperture
    lid_shadow = (
        (yy > cfg.eye_cy - cfg.eye_h / 2 - 5)
        & (yy < cfg.eye_cy - cfg.eye_h / 2 + 2)
        & (np.abs(xx - cfg.eye_cx) < cfg.eye_w / 2)
    )
    img = np.where(lid_shadow[..., None], img * 0.72, img)

    out = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
    return RigEye(
        image=out,
        ground_truth_dx=float(dx),
        ground_truth_dy=float(dy),
        eye_cx=float(cfg.eye_cx),
        eye_cy=float(cfg.eye_cy),
        eye_w=float(cfg.eye_w),
        iris_r=float(cfg.iris_r),
    )


def offset_grid(percent_steps: list[float]) -> list[tuple[float, float]]:
    """(dx, dy) offsets as fractions of eye width, e.g. 0.25 → 30 px on a 120 px eye."""
    cfg = RigConfig()
    out: list[tuple[float, float]] = []
    for px in percent_steps:
        out.append((px * cfg.eye_w, 0.0))
        out.append((-px * cfg.eye_w, 0.0))
        out.append((0.0, px * cfg.eye_w * 0.4))
        out.append((0.0, -px * cfg.eye_w * 0.4))
    return out
