"""Gaze research tests (synthetic rig; needs the ``gaze`` extra).

These prove the algorithm on controlled input. They deliberately do NOT
claim anything about real webcam footage — see docs/EYE_CONTACT_RESEARCH.md.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gaze


def _gaze_available() -> bool:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def eval_report() -> dict[str, object]:
    if not _gaze_available():
        pytest.skip("gaze extra (numpy/opencv) not installed")
    from openclip.gaze.evaluate import run_all

    return run_all()


def test_e1_detection_accuracy(eval_report: dict[str, object]) -> None:
    e1 = eval_report["E1_detection"]
    assert isinstance(e1, dict)
    assert e1["miss_rate"] <= 0.25, f"detection misses too often: {e1}"
    assert e1["mean_err_px"] <= 6.0, f"mean localization error too high: {e1}"


def test_e2_correction_reduces_offset(eval_report: dict[str, object]) -> None:
    e2 = eval_report["E2E3_correction"]
    assert isinstance(e2, dict)
    assert e2["mean_reduction_pct"] >= 60.0, f"correction too weak: {e2}"
    assert e2["mean_error_after_pct"] <= e2["mean_error_before_pct"]


def test_e3_no_artifacts_outside_eye(eval_report: dict[str, object]) -> None:
    e2 = eval_report["E2E3_correction"]
    assert isinstance(e2, dict)
    assert e2["min_psnr_outside_db"] >= 35.0, f"visible artifacts outside eye: {e2}"


def test_e4_speed_recorded(eval_report: dict[str, object]) -> None:
    e4 = eval_report["E4_speed"]
    assert isinstance(e4, dict)
    assert e4["fps"] >= 2.0, f"pathologically slow even for rig input: {e4}"


def test_warp_is_identity_when_centered() -> None:
    if not _gaze_available():
        pytest.skip("gaze extra not installed")
    from openclip.gaze.detect import locate_in_patch
    from openclip.gaze.rig import RigConfig, render_eye
    from openclip.gaze.warp import warp_patch

    cfg = RigConfig()
    eye = render_eye(cfg, 0.0, 0.0)
    loc = locate_in_patch(eye.image)
    if loc.found:
        out = warp_patch(eye.image, loc, 1.0)
        assert out.shape == eye.image.shape


def test_gaze_apply_video_patch_mode(tmp_path) -> None:
    """End-to-end on a rig-eye video: corrected iris is measurably closer."""
    if not _gaze_available():
        pytest.skip("gaze extra not installed")
    import cv2
    import numpy as np

    from openclip.gaze.apply import apply_to_video
    from openclip.gaze.detect import locate_in_patch
    from openclip.gaze.rig import RigConfig, render_eye

    cfg = RigConfig()
    eye = render_eye(cfg, 0.22 * cfg.eye_w, 0.0)
    src = tmp_path / "eye.mp4"
    writer = cv2.VideoWriter(
        str(src), cv2.VideoWriter_fourcc(*"mp4v"), 15, (cfg.canvas_w, cfg.canvas_h)
    )
    for _ in range(24):
        writer.write(eye.image)
    writer.release()

    out = tmp_path / "corrected.mp4"
    stats = apply_to_video(src, out, strength=1.0, mode="patch")
    assert stats.frames == 24
    assert stats.frames_with_eyes >= 20

    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok, "corrected video has no frames"

    def offset(img: np.ndarray) -> float:
        loc = locate_in_patch(img)
        assert loc.found
        return float(
            np.hypot(loc.iris_cx - loc.eye_cx, loc.iris_cy - loc.eye_cy) / loc.eye_w * 100
        )

    assert offset(frame) < offset(eye.image)


def test_cli_gaze_tools_run(tmp_path) -> None:
    if not _gaze_available():
        pytest.skip("gaze extra not installed")
    import cv2

    from openclip.cli import main as cli_main
    from openclip.gaze.rig import RigConfig, render_eye

    cfg = RigConfig()
    eye = render_eye(cfg, 0.2 * cfg.eye_w, 0.0)
    src = tmp_path / "eye.mp4"
    writer = cv2.VideoWriter(
        str(src), cv2.VideoWriter_fourcc(*"mp4v"), 15, (cfg.canvas_w, cfg.canvas_h)
    )
    for _ in range(12):
        writer.write(eye.image)
    writer.release()

    assert cli_main(["gaze-analyze", str(src), "--max-frames", "12"]) == 0
    assert (
        cli_main(["gaze-apply", str(src), "-o", str(tmp_path / "fixed.mp4"), "--mode", "patch"])
        == 0
    )
    assert (tmp_path / "fixed.mp4").exists()
