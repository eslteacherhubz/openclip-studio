"""Evaluation harness: measurement + experiment runner.

Produces the numbers recorded in docs/EYE_CONTACT_RESEARCH.md. Run via
``python -m openclip.gaze.evaluate`` (needs the gaze extra) or through
pytest (tests/test_gaze.py). Synthetic rig only — see the research doc for
what that does and does not prove.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from openclip.gaze.detect import EyeLocation, locate_in_patch
from openclip.gaze.rig import RigConfig, RigEye, offset_grid, render_eye
from openclip.gaze.warp import warp_patch


@dataclass
class TrialResult:
    dx_gt: float
    dy_gt: float
    detected_before: tuple[float, float]
    detected_after: tuple[float, float]
    error_before_pct: float  # % of eye width
    error_after_pct: float
    psnr_outside_eye_db: float


def _iris_error(loc: EyeLocation, eye_w: float) -> float:
    return float(np.hypot(loc.iris_cx - loc.eye_cx, loc.iris_cy - loc.eye_cy) / eye_w * 100.0)


def _psnr_outside_aperture(orig: np.ndarray, corrected: np.ndarray, eye: RigEye) -> float:
    h, w = orig.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Outside-eye = anything clearly beyond a generous ellipse around the eye.
    ex = eye.eye_w / 2 + eye.iris_r * 2.6
    ey = eye.eye_w / 4 + eye.iris_r * 2.2
    outside = ((xx - eye.eye_cx) / ex) ** 2 + ((yy - eye.eye_cy) / ey) ** 2 > 1.0
    if outside.sum() < 50:
        return float("nan")
    a = orig[outside].astype(np.float64)
    b = corrected[outside].astype(np.float64)
    mse = float(((a - b) ** 2).mean())
    if mse <= 1e-9:
        return 99.0
    return float(10.0 * np.log10(255.0**2 / mse))


def run_trial(eye: RigEye, strength: float = 1.0) -> TrialResult:
    before = locate_in_patch(eye.image)
    # Build a synthetic EyeLocation from ground truth for the *warp input*
    # when detection failed, so warp quality is measured independently of
    # detector misses (detection accuracy is reported separately).
    warp_input = before if before.found else _gt_location(eye)
    corrected = warp_patch(eye.image, warp_input, strength)
    after = locate_in_patch(corrected)
    return TrialResult(
        dx_gt=eye.ground_truth_dx,
        dy_gt=eye.ground_truth_dy,
        detected_before=(before.iris_cx, before.iris_cy) if before.found else (-1, -1),
        detected_after=(after.iris_cx, after.iris_cy) if after.found else (-1, -1),
        error_before_pct=_iris_error(before, eye.eye_w) if before.found else float("nan"),
        error_after_pct=_iris_error(after, eye.eye_w) if after.found else float("nan"),
        psnr_outside_eye_db=_psnr_outside_aperture(eye.image, corrected, eye),
    )


def _gt_location(eye: RigEye) -> EyeLocation:
    return EyeLocation(
        found=True,
        eye_cx=eye.eye_cx,
        eye_cy=eye.eye_cy,
        iris_cx=eye.eye_cx + eye.ground_truth_dx,
        iris_cy=eye.eye_cy + eye.ground_truth_dy,
        eye_w=eye.eye_w,
        iris_r=eye.iris_r,
    )


def detection_accuracy(percent_steps: list[float]) -> dict[str, float]:
    """E1: how well does locate_in_patch recover ground-truth offsets?"""
    cfg = RigConfig()
    errs: list[float] = []
    misses = 0
    trials = 0
    for dx, dy in offset_grid(percent_steps):
        eye = render_eye(cfg, dx, dy)
        loc = locate_in_patch(eye.image)
        trials += 1
        if not loc.found:
            misses += 1
            continue
        err = float(
            np.hypot(
                loc.iris_cx - (eye.eye_cx + dx), loc.iris_cy - (eye.eye_cy + dy)
            )
        )
        errs.append(err)
    return {
        "trials": trials,
        "miss_rate": misses / max(1, trials),
        "mean_err_px": float(np.mean(errs)) if errs else float("nan"),
        "max_err_px": float(np.max(errs)) if errs else float("nan"),
    }


def correction_effect(strength: float = 1.0) -> dict[str, float]:
    """E2/E3: error reduction + artifact PSNR across the offset grid."""
    cfg = RigConfig()
    results: list[TrialResult] = []
    for dx, dy in offset_grid([0.05, 0.15, 0.25]):
        eye = render_eye(cfg, dx, dy)
        results.append(run_trial(eye, strength))
    valid = [r for r in results if not np.isnan(r.error_after_pct)]
    reduction = [
        100.0 * (r.error_before_pct - r.error_after_pct) / max(r.error_before_pct, 1e-6)
        for r in valid
        if not np.isnan(r.error_before_pct)
    ]
    return {
        "trials": len(results),
        "evaluated": len(valid),
        "mean_error_before_pct": float(np.nanmean([r.error_before_pct for r in results])),
        "mean_error_after_pct": float(np.nanmean([r.error_after_pct for r in results])),
        "mean_reduction_pct": float(np.mean(reduction)) if reduction else float("nan"),
        "min_psnr_outside_db": float(np.nanmin([r.psnr_outside_eye_db for r in results])),
    }


def speed_benchmark(n_frames: int = 120) -> dict[str, float]:
    """E4: detect+warp throughput on THIS hardware (documented in research doc)."""
    cfg = RigConfig()
    eye = render_eye(cfg, 0.22 * cfg.eye_w, 0.02 * cfg.eye_w)
    t0 = time.perf_counter()
    for _ in range(n_frames):
        loc = locate_in_patch(eye.image)
        if loc.found:
            warp_patch(eye.image, loc)
    dt = time.perf_counter() - t0
    return {"frames": n_frames, "seconds": dt, "fps": n_frames / dt}


def run_all(strength: float = 1.0) -> dict[str, dict[str, float]]:
    e1 = detection_accuracy([0.05, 0.15, 0.25])
    e2 = correction_effect(strength)
    e4 = speed_benchmark()
    return {"E1_detection": e1, "E2E3_correction": e2, "E4_speed": e4}


def render_report(report: dict[str, dict[str, float]]) -> str:
    e1 = report["E1_detection"]
    e2 = report["E2E3_correction"]
    e4 = report["E4_speed"]
    lines = [
        "Gaze evaluation report (synthetic rig)",
        "=====================================",
        f"E1 detection: miss_rate={e1['miss_rate']:.2f} mean_err={e1['mean_err_px']:.2f}px "
        f"max_err={e1['max_err_px']:.2f}px over {e1['trials']} trials",
        f"E2 correction: error {e2['mean_error_before_pct']:.1f}% -> "
        f"{e2['mean_error_after_pct']:.1f}% of eye width "
        f"(mean reduction {e2['mean_reduction_pct']:.1f}%, {e2['evaluated']}/{e2['trials']})",
        f"E3 artifacts: min PSNR outside eye = {e2['min_psnr_outside_db']:.1f} dB",
        f"E4 speed: {e4['fps']:.1f} frames/s (detect+warp, {e4['frames']} frames)",
        "",
        "Synthetic rig only. Real-footage validation: see",
        "docs/EYE_CONTACT_RESEARCH.md § Owner validation checklist.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import json

    print(json.dumps(run_all(), indent=2))
    print(render_report(run_all()))
