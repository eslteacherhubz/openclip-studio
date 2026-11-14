"""Tests for Phase 8 Objective 3: smart reframing (subject-tracking 9:16 crop)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from openclip.ai.reframing import (
    DEFAULT_TARGET_ASPECT,
    TrackingPoint,
    anchor_at,
    build_smart_reframe_filter,
    crop_size_for_aspect,
    mock_tracking_data,
)
from openclip.cli import main
from openclip.engine.ffmpeg import probe, run_ffmpeg
from openclip.engine.synth import generate_lesson


class TestCropSize:
    def test_landscape_source(self) -> None:
        # 320x180, 9:16 → full height, width = 180 * 9/16 = 101.25 → even → 100
        assert crop_size_for_aspect(320, 180) == (100, 180)

    def test_1080p_source(self) -> None:
        w, h = crop_size_for_aspect(1920, 1080)
        assert h == 1080 and abs(w / h - 9 / 16) < 0.01

    def test_tall_source_falls_back_to_full_width(self) -> None:
        # 100x1000 is narrower than 9:16 → crop width = 100, height = 100/(9/16)
        w, h = crop_size_for_aspect(100, 1000)
        assert w == 100 and h == 178  # round(100/(9/16))=178, already even

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            crop_size_for_aspect(0, 100)
        with pytest.raises(ValueError):
            crop_size_for_aspect(100, 100, 0)


class TestAnchorAt:
    PTS: ClassVar[list[TrackingPoint]] = [
        TrackingPoint(0.0, 0.2, 0.5),
        TrackingPoint(10.0, 0.8, 0.5),
    ]
    # 320x180 source, 9:16 → crop 100x180

    def test_interpolation_midpoint(self) -> None:
        x, y = anchor_at(self.PTS, 5.0, 320, 180, 100, 180)
        assert x == pytest.approx(0.5 * 320 - 100 / 2)  # 109.5
        assert y == pytest.approx(0.0)  # centered vertically: 0.5*180-90=0

    def test_endpoints(self) -> None:
        x0, _ = anchor_at(self.PTS, 0.0, 320, 180, 100, 180)
        x1, _ = anchor_at(self.PTS, 10.0, 320, 180, 100, 180)
        assert x0 == pytest.approx(0.2 * 320 - 50.0)  # 13.5
        assert x1 == pytest.approx(0.8 * 320 - 50.0)  # 205.5

    def test_clamped_at_left_edge(self) -> None:
        pts = [TrackingPoint(0.0, 0.0, 0.5)]
        x, _ = anchor_at(pts, 0.0, 320, 180, 100, 180)
        assert x == 0.0  # would be -50.5 without clamping

    def test_clamped_at_right_edge(self) -> None:
        pts = [TrackingPoint(0.0, 1.0, 0.5)]
        x, _ = anchor_at(pts, 0.0, 320, 180, 100, 180)
        assert x == 320 - 100  # 220, not 270

    def test_clamped_vertically(self) -> None:
        pts = [TrackingPoint(0.0, 0.5, 1.0), TrackingPoint(4.0, 0.5, 0.0)]
        _, y_bottom = anchor_at(pts, 0.0, 320, 180, 100, 180)
        _, y_top = anchor_at(pts, 4.0, 320, 180, 100, 180)
        assert y_bottom == 0.0 and y_top == 0.0  # crop is full height → always 0
        # with a shorter crop window the clamp engages:
        _, y_lo = anchor_at(pts, 0.0, 320, 180, 100, 100)
        _, y_hi = anchor_at(pts, 4.0, 320, 180, 100, 100)
        assert y_lo == 80.0  # 1.0*180 - 50 clamped to 180-100
        assert y_hi == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            anchor_at([], 0.0, 320, 180, 100, 180)


class TestBuildSmartReframeFilter:
    def test_structure(self) -> None:
        vf = build_smart_reframe_filter(mock_tracking_data(10.0), (320, 180))
        assert vf.startswith("crop=w=100:h=180:x='")
        assert ":y='" in vf
        assert "clip(" in vf  # clamped
        assert "if(lt(t," in vf  # piecewise segments
        assert vf.count("clip(") >= 5  # one per tracking point per axis at least

    def test_single_point_is_constant_clamped(self) -> None:
        vf = build_smart_reframe_filter([TrackingPoint(0.0, 0.0, 0.0)], (320, 180))
        assert "if(lt(t," not in vf
        assert "x='clip(0.000000,0,220.000000)'" in vf
        assert "y='clip(0.000000,0,0.000000)'" in vf  # full-height crop

    def test_deterministic(self) -> None:
        a = build_smart_reframe_filter(mock_tracking_data(10.0), (320, 180))
        b = build_smart_reframe_filter(mock_tracking_data(10.0), (320, 180))
        assert a == b

    def test_custom_aspect(self) -> None:
        vf = build_smart_reframe_filter(mock_tracking_data(4.0), (200, 200), 1.0)
        assert vf.startswith("crop=w=200:h=200:")

    def test_out_of_range_norms_clamped(self) -> None:
        vf = build_smart_reframe_filter([TrackingPoint(0.0, 5.0, -3.0)], (320, 180))
        assert "x='clip(220.000000,0,220.000000)'" in vf  # 1.0 → right edge
        assert "y='clip(0.000000,0,0.000000)'" in vf

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            build_smart_reframe_filter([], (320, 180))

    def test_ffmpeg_accepts_expression(self, tmp_path: Path) -> None:
        # the generated filter must parse and run in real ffmpeg
        vf = build_smart_reframe_filter(mock_tracking_data(2.0), (320, 180))
        out = tmp_path / "crop.mp4"
        run_ffmpeg(
            [
                "-y",
                "-f", "lavfi", "-i", "color=c=blue:size=320x180:duration=2:rate=10",
                "-vf", vf,
                "-frames:v", "5",
                str(out),
            ],
            timeout=60,
        )
        info = probe(out)
        assert info.width == 100 and info.height == 180

    def test_ffmpeg_edge_tracking_no_black_bars(self, tmp_path: Path) -> None:
        # subject slammed to the extreme corners: window must stay in bounds,
        # i.e. output frame is fully sourced (no padding) — verified by running
        # the filter at the edge timestamps without error and correct size.
        pts = [TrackingPoint(0.0, 0.0, 0.0), TrackingPoint(2.0, 1.0, 1.0)]
        vf = build_smart_reframe_filter(pts, (320, 180))
        out = tmp_path / "edge.mp4"
        run_ffmpeg(
            [
                "-y",
                "-f", "lavfi", "-i", "color=c=green:size=320x180:duration=2:rate=10",
                "-vf", vf,
                str(out),
            ],
            timeout=60,
        )
        info = probe(out)
        assert (info.width, info.height) == (100, 180)


class TestMockTrackingData:
    def test_left_to_right(self) -> None:
        pts = mock_tracking_data(10.0)
        assert pts[0].t == 0.0 and pts[-1].t == 10.0
        assert pts[0].x_center_norm == pytest.approx(0.2)
        assert pts[-1].x_center_norm == pytest.approx(0.8)
        assert all(p.y_center_norm == 0.5 for p in pts)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            mock_tracking_data(0)
        with pytest.raises(ValueError):
            mock_tracking_data(10, steps=0)


@pytest.fixture(scope="module")
def lesson(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_lesson(tmp_path_factory.mktemp("reframe") / "lesson.mp4").path


class TestCliAutoReframe:
    def test_renders_9x16_with_auto_reframe(self, lesson: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.mp4"
        rc = main(
            [
                "ai-edit",
                "--media", str(lesson),
                "--prompt", "Make a short vertical clip about the main topic",
                "-o", str(out),
                "--provider", "local_mock",
                "--auto-accept",
                "--auto-reframe",
            ]
        )
        assert rc == 0
        info = probe(out)
        # mock plan condensation keeps 9s of the 14s lesson
        assert info.duration_s == pytest.approx(9.0, abs=1.5)
        assert info.width is not None and info.height is not None
        assert info.width / info.height == pytest.approx(DEFAULT_TARGET_ASPECT, abs=0.02)
