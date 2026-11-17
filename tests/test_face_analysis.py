"""Unit tests for :mod:`openclip.ai.face_analysis` and its reframing wiring.

Fully offline: the mock provider and synthetic spy providers only.
"""

from __future__ import annotations

import logging
import random

import pytest

from openclip.ai import face_analysis
from openclip.ai.face_analysis import (
    CachedFaceAnalysis,
    FaceAnalysisProvider,
    FaceFrame,
    MediaPipeFaceAnalysisProvider,
    MockFaceAnalysisProvider,
    Point,
    _build_face_frame,
    default_provider,
    real_tracking_available,
    select_provider,
)
from openclip.ai.reframing import (
    TrackingPoint,
    tracking_from_default_face_analysis,
    tracking_from_face_analysis,
)


def _frame(timestamp: float, center_x: float = 0.5, center_y: float = 0.5) -> FaceFrame:
    return FaceFrame(
        timestamp=timestamp,
        center_x=center_x,
        center_y=center_y,
        jaw_neck_landmarks=((center_x - 0.05, center_y + 0.1), (center_x + 0.05, center_y + 0.1)),
        skin_mask_cells=((center_x - 0.05, center_y - 0.05, 0.1, 0.1),),
        exclusion_zones=((center_x - 0.03, center_y - 0.03, 0.06, 0.04),),
    )


class _CountingProvider:
    """Spy provider: counts analyze_frame calls and returns a fixed frame."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def analyze_frame(
        self, frame_index: int, timestamp: float, frame: object = None
    ) -> FaceFrame | None:
        self.calls.append(frame_index)
        return _frame(timestamp)


class _NoisyProvider:
    """Deterministic noisy-center provider (fixed seed)."""

    def __init__(self, n: int = 100, seed: int = 7) -> None:
        self._values = [random.Random(seed).uniform(0.2, 0.8) for _ in range(n)]

    def analyze_frame(
        self, frame_index: int, timestamp: float, frame: object = None
    ) -> FaceFrame | None:
        cx = self._values[frame_index % len(self._values)]
        return _frame(timestamp, center_x=cx, center_y=0.5)


def _variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


class TestCachedFaceAnalysis:
    def test_same_frame_index_analyzed_once(self) -> None:
        spy = _CountingProvider()
        cached = CachedFaceAnalysis(spy)
        first = cached.analyze_frame(3, 0.1)
        second = cached.analyze_frame(3, 0.1)
        third = cached.analyze_frame(3, 0.1)
        assert spy.calls == [3]
        assert first is not None and first == second == third

    def test_distinct_frames_each_analyzed_once(self) -> None:
        spy = _CountingProvider()
        cached = CachedFaceAnalysis(spy)
        for i in range(5):
            cached.analyze_frame(i, i / 30.0)
            cached.analyze_frame(i, i / 30.0)
        assert spy.calls == [0, 1, 2, 3, 4]

    def test_none_results_are_cached_too(self) -> None:
        class _NoneProvider:
            def __init__(self) -> None:
                self.calls = 0

            def analyze_frame(
                self, frame_index: int, timestamp: float, frame: object = None
            ) -> FaceFrame | None:
                self.calls += 1
                return None

        spy = _NoneProvider()
        cached = CachedFaceAnalysis(spy)
        assert cached.analyze_frame(9, 0.3) is None
        assert cached.analyze_frame(9, 0.3) is None
        assert spy.calls == 1

    def test_smoothing_applied_to_output_streams(self) -> None:
        raw = _NoisyProvider()
        cached = CachedFaceAnalysis(raw)
        raw_xs = [raw.analyze_frame(i, i / 30.0).center_x for i in range(100)]  # type: ignore[union-attr]
        out_xs = [cached.analyze_frame(i, i / 30.0).center_x for i in range(100)]  # type: ignore[union-attr]
        assert _variance(out_xs) < _variance(raw_xs)

    def test_implements_provider_protocol(self) -> None:
        cached = CachedFaceAnalysis(_CountingProvider())
        assert isinstance(cached, FaceAnalysisProvider)


class TestMockFaceAnalysisProvider:
    def test_deterministic_across_runs(self) -> None:
        run_a = [MockFaceAnalysisProvider().analyze_frame(i, i / 30.0) for i in range(50)]
        run_b = [MockFaceAnalysisProvider().analyze_frame(i, i / 30.0) for i in range(50)]
        assert run_a == run_b

    def test_coordinates_within_unit_square(self) -> None:
        provider = MockFaceAnalysisProvider()
        for i in range(120):
            frame = provider.analyze_frame(i, i / 30.0)
            if frame is None:
                continue
            assert 0.0 <= frame.center_x <= 1.0
            assert 0.0 <= frame.center_y <= 1.0
            for p in frame.jaw_neck_landmarks:
                if p is None:
                    continue
                assert 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0
            for cell in frame.skin_mask_cells:
                assert all(0.0 <= v <= 1.0 for v in cell)
            for zone in frame.exclusion_zones:
                assert all(0.0 <= v <= 1.0 for v in zone)

    def test_landmarks_mask_and_zones_populated(self) -> None:
        frame = MockFaceAnalysisProvider().analyze_frame(0, 0.0)
        assert frame is not None
        assert len(frame.jaw_neck_landmarks) == 8
        assert len(frame.skin_mask_cells) == 9
        assert len(frame.exclusion_zones) == 3

    def test_simulated_whole_frame_dropout(self) -> None:
        provider = MockFaceAnalysisProvider()
        assert provider.analyze_frame(0, 0.0) is not None
        assert provider.analyze_frame(97, 97 / 30.0) is None


class TestDefaultProvider:
    def test_returns_mock_offline_safe(self) -> None:
        provider = default_provider()
        assert isinstance(provider, MockFaceAnalysisProvider)
        assert isinstance(provider, FaceAnalysisProvider)


class TestRealTrackingAvailable:
    def test_returns_bool_without_raising(self) -> None:
        assert isinstance(real_tracking_available(), bool)


class TestSelectProvider:
    def test_default_is_mock(self) -> None:
        assert isinstance(select_provider(), MockFaceAnalysisProvider)
        assert isinstance(select_provider(prefer_real=False), MockFaceAnalysisProvider)

    def test_prefer_real_unavailable_falls_back_to_mock_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(face_analysis, "real_tracking_available", lambda: False)
        with caplog.at_level(logging.WARNING, logger="openclip.ai.face_analysis"):
            provider = select_provider(prefer_real=True)
        assert isinstance(provider, MockFaceAnalysisProvider)
        assert any("Real face tracking unavailable" in r.message for r in caplog.records)

    def test_prefer_real_never_raises_importerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(face_analysis, "real_tracking_available", lambda: False)
        provider = select_provider(prefer_real=True)
        assert isinstance(provider, FaceAnalysisProvider)

    def test_prefer_real_uses_real_provider_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(face_analysis, "real_tracking_available", lambda: True)
        monkeypatch.setattr(
            face_analysis, "MediaPipeFaceAnalysisProvider", MockFaceAnalysisProvider
        )
        assert isinstance(select_provider(prefer_real=True), MockFaceAnalysisProvider)


class TestTrackingFromFaceAnalysis:
    def test_sample_stride(self) -> None:
        points = tracking_from_face_analysis(MockFaceAnalysisProvider(), fps=30.0, duration=2.0)
        assert len(points) == 10
        assert points[0].t == 0.0
        assert points[1].t == pytest.approx(0.2)
        assert all(isinstance(p, TrackingPoint) for p in points)

    def test_none_results_are_skipped(self) -> None:
        class _AlternatingProvider:
            def analyze_frame(
                self, frame_index: int, timestamp: float, frame: object = None
            ) -> FaceFrame | None:
                if frame_index % 2 == 0:
                    return None
                return _frame(timestamp)

        points = tracking_from_face_analysis(
            _AlternatingProvider(), fps=10.0, duration=1.0, sample_fps=10.0
        )
        assert [p.t for p in points] == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])

    def test_deterministic_output(self) -> None:
        a = tracking_from_face_analysis(MockFaceAnalysisProvider(), 30.0, 3.0)
        b = tracking_from_face_analysis(MockFaceAnalysisProvider(), 30.0, 3.0)
        assert a == b

    def test_invalid_args_raise(self) -> None:
        provider = MockFaceAnalysisProvider()
        with pytest.raises(ValueError):
            tracking_from_face_analysis(provider, fps=0.0, duration=1.0)
        with pytest.raises(ValueError):
            tracking_from_face_analysis(provider, fps=30.0, duration=-1.0)
        with pytest.raises(ValueError):
            tracking_from_face_analysis(provider, fps=30.0, duration=1.0, sample_fps=0.0)

    def test_zero_points_fall_back_to_center_anchor_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _NoneProvider:
            def analyze_frame(
                self, frame_index: int, timestamp: float, frame: object = None
            ) -> FaceFrame | None:
                return None

        with caplog.at_level(logging.WARNING, logger="openclip.ai.reframing"):
            points = tracking_from_face_analysis(_NoneProvider(), fps=30.0, duration=2.0)
        assert points == [TrackingPoint(t=0.0, x_center_norm=0.5, y_center_norm=0.5)]
        assert any("center anchor" in r.message for r in caplog.records)


class TestTrackingFromDefaultFaceAnalysis:
    def test_non_empty_sorted_points_for_10s(self) -> None:
        points = tracking_from_default_face_analysis(fps=30.0, duration=10.0)
        assert len(points) > 0
        assert all(isinstance(p, TrackingPoint) for p in points)
        times = [p.t for p in points]
        assert times == sorted(times)
        assert times[0] == 0.0


class TestFaceFrameMapping:
    """Pure landmark -> FaceFrame mapping (synthetic landmarks, no MediaPipe)."""

    def _landmarks(self, placements: dict[int, tuple[float, float]]) -> list[Point | None]:
        pts: list[Point | None] = [None] * 468
        for idx, xy in placements.items():
            pts[idx] = xy
        return pts

    def test_normalized_bounds_and_populated_regions(self) -> None:
        from openclip.ai.face_analysis import (
            FACE_OVAL_IDX,
            LEFT_EYE_IDX,
            LIPS_IDX,
            NOSE_IDX,
            RIGHT_EYE_IDX,
        )

        placements: dict[int, tuple[float, float]] = {}
        for i in FACE_OVAL_IDX:
            placements[i] = (0.3 + 0.01 * (i % 7), 0.3 + 0.01 * (i // 7))
        for i in LEFT_EYE_IDX:
            placements[i] = (0.35, 0.40)
        for i in RIGHT_EYE_IDX:
            placements[i] = (0.65, 0.40)
        for i in LIPS_IDX:
            placements[i] = (0.50, 0.70)
        for i in NOSE_IDX:
            placements[i] = (0.50, 0.55)

        frame = _build_face_frame(
            timestamp=1.0,
            landmarks_xy=self._landmarks(placements),
            width=320,
            height=240,
        )

        # center inside the face oval bbox (spread 0.30..0.47 roughly)
        assert 0.0 <= frame.center_x <= 1.0
        assert 0.0 <= frame.center_y <= 1.0

        # exclusion zones: left eye, right eye, lips, nostrils
        assert len(frame.exclusion_zones) == 4, frame.exclusion_zones
        for zone in frame.exclusion_zones:
            assert all(0.0 <= v <= 1.0 for v in zone), zone

        # jaw_neck_landmarks: face oval + 2 neck points
        assert len(frame.jaw_neck_landmarks) == len(FACE_OVAL_IDX) + 2
        for p in frame.jaw_neck_landmarks:
            if p is not None:
                assert 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0

        # skin mask cells: 4 cells
        assert len(frame.skin_mask_cells) == 4
        for cell in frame.skin_mask_cells:
            assert all(0.0 <= v <= 1.0 for v in cell), cell

    def test_all_none_landmarks_degenerate_but_valid(self) -> None:
        """No face detected -> degenerate frame (center at frame center), not None."""
        frame = _build_face_frame(
            timestamp=0.0, landmarks_xy=[None] * 468, width=320, height=240
        )
        assert frame.center_x == 0.5
        assert frame.center_y == 0.5
        assert frame.jaw_neck_landmarks == ()
        assert frame.skin_mask_cells == ()
        assert frame.exclusion_zones == ()


@pytest.mark.skipif(
    not real_tracking_available(), reason="MediaPipe backend (cv2+mediapipe+model) not available"
)
class TestMediaPipeFaceAnalysisProvider:
    """Real MediaPipe FaceLandmarker provider (gaze marker: cv2+mediapipe+model)."""

    def test_blank_image_returns_none(self) -> None:
        provider = MediaPipeFaceAnalysisProvider()
        try:
            import numpy as np

            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            assert provider.analyze_frame(0, 0.0, frame) is None
        finally:
            provider._landmarker.close()
