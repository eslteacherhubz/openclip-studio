"""Per-frame face analysis for reframing: anchor, landmarks, mask, exclusion zones.

The reframing layer consumes one :class:`FaceFrame` per video frame. All
coordinates are normalized to the unit square (0-1 relative to frame width
and height). Providers implement :class:`FaceAnalysisProvider`;
:class:`MockFaceAnalysisProvider` is the deterministic default so the pipeline
runs fully offline, and :class:`MediaPipeFaceAnalysisProvider` is a stub for
the real MediaPipe/OpenCV backend behind an optional, guarded import.

Raw per-frame detections jitter, so :class:`CachedFaceAnalysis` analyzes each
``frame_index`` at most once and smooths every coordinate stream with
:mod:`openclip.ai.smoothing` (streaming EWMA; optional point streams carry the
last value forward across dropouts, the semantics of
``smooth_optional_pairs``) before returning.

Pure stdlib; the only non-stdlib imports are the optional ``mediapipe``/``cv2``
inside the ``try``/``except ImportError`` below.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from openclip.ai.smoothing import DEFAULT_SMOOTH_ALPHA, EWMA

try:
    import cv2
    import mediapipe  # type: ignore[import-not-found]
except ImportError:
    cv2 = None  # type: ignore[assignment]
    mediapipe = None

logger = logging.getLogger(__name__)

#: Normalized 2D point ``(x, y)``, both components in 0-1.
Point = tuple[float, float]

#: Normalized rectangle ``(x, y, width, height)``, all components in 0-1.
Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class FaceFrame:
    """Analysis of a single video frame; every coordinate is normalized 0-1."""

    timestamp: float
    center_x: float
    center_y: float
    jaw_neck_landmarks: tuple[Point | None, ...]
    skin_mask_cells: tuple[Rect, ...]
    exclusion_zones: tuple[Rect, ...]


@runtime_checkable
class FaceAnalysisProvider(Protocol):
    """Source of per-frame face analysis."""

    def analyze_frame(self, frame_index: int, timestamp: float) -> FaceFrame | None:
        """Return the analysis for ``frame_index``, or ``None`` if no face is found."""
        ...


class _PointStream:
    """Streaming EWMA over optional 2D points; ``None`` carries the last value."""

    def __init__(self, alpha: float) -> None:
        self._sx = EWMA(alpha)
        self._sy = EWMA(alpha)

    def update(self, p: Point | None) -> Point | None:
        if p is None:
            if self._sx.value is not None and self._sy.value is not None:
                return (self._sx.value, self._sy.value)
            return None
        return (self._sx.update(p[0]), self._sy.update(p[1]))


class _RectStream:
    """Streaming coordinate-wise EWMA over normalized rectangles."""

    def __init__(self, alpha: float) -> None:
        self._ewmas = [EWMA(alpha) for _ in range(4)]

    def update(self, rect: Rect) -> Rect:
        vals = [ewma.update(v) for ewma, v in zip(self._ewmas, rect, strict=True)]
        return (vals[0], vals[1], vals[2], vals[3])


class CachedFaceAnalysis:
    """Caching, smoothing wrapper around a :class:`FaceAnalysisProvider`.

    Each ``frame_index`` is passed to the wrapped provider at most once; the
    smoothed result is cached, so repeated calls for the same frame return the
    identical :class:`FaceFrame` and no frame is ever analyzed twice.
    """

    def __init__(
        self,
        provider: FaceAnalysisProvider,
        alpha: float = DEFAULT_SMOOTH_ALPHA,
    ) -> None:
        self._provider = provider
        self._alpha = alpha
        self._cache: dict[int, FaceFrame | None] = {}
        self._cx = EWMA(alpha)
        self._cy = EWMA(alpha)
        self._landmark_streams: list[_PointStream] = []
        self._skin_streams: list[_RectStream] = []
        self._zone_streams: list[_RectStream] = []

    def analyze_frame(self, frame_index: int, timestamp: float) -> FaceFrame | None:
        """Return the smoothed, cached analysis for ``frame_index``."""
        if frame_index in self._cache:
            return self._cache[frame_index]
        raw = self._provider.analyze_frame(frame_index, timestamp)
        result = None if raw is None else self._smooth(raw)
        self._cache[frame_index] = result
        return result

    def _smooth(self, raw: FaceFrame) -> FaceFrame:
        landmarks: list[Point | None] = []
        for i, p in enumerate(raw.jaw_neck_landmarks):
            while len(self._landmark_streams) <= i:
                self._landmark_streams.append(_PointStream(self._alpha))
            landmarks.append(self._landmark_streams[i].update(p))
        return FaceFrame(
            timestamp=raw.timestamp,
            center_x=self._cx.update(raw.center_x),
            center_y=self._cy.update(raw.center_y),
            jaw_neck_landmarks=tuple(landmarks),
            skin_mask_cells=tuple(self._smooth_rects(self._skin_streams, raw.skin_mask_cells)),
            exclusion_zones=tuple(self._smooth_rects(self._zone_streams, raw.exclusion_zones)),
        )

    def _smooth_rects(self, streams: list[_RectStream], rects: tuple[Rect, ...]) -> list[Rect]:
        out: list[Rect] = []
        for i, rect in enumerate(rects):
            while len(streams) <= i:
                streams.append(_RectStream(self._alpha))
            out.append(streams[i].update(rect))
        return out


class MockFaceAnalysisProvider:
    """Deterministic synthetic provider; the default for offline runs.

    The subject pans left/right sinusoidally with a slight vertical bob, and
    landmarks, skin cells, and exclusion zones are derived from that anchor so
    the output is plausible and fully reproducible. Detection dropouts are
    simulated at fixed frame intervals (whole-frame ``None`` and a single
    landmark dropout).
    """

    PAN_PERIOD_FRAMES: int = 120
    BOB_PERIOD_FRAMES: int = 60
    DROP_FRAME_MODULUS: int = 97
    LANDMARK_DROP_MODULUS: int = 47

    def analyze_frame(self, frame_index: int, timestamp: float) -> FaceFrame | None:
        """Return the synthetic :class:`FaceFrame` for ``frame_index``."""
        if frame_index > 0 and frame_index % self.DROP_FRAME_MODULUS == 0:
            return None
        cx = 0.5 + 0.3 * math.sin(2.0 * math.pi * frame_index / self.PAN_PERIOD_FRAMES)
        cy = 0.45 + 0.03 * math.sin(
            2.0 * math.pi * frame_index / self.BOB_PERIOD_FRAMES + 1.0
        )
        return FaceFrame(
            timestamp=timestamp,
            center_x=cx,
            center_y=cy,
            jaw_neck_landmarks=self._landmarks(cx, cy, frame_index),
            skin_mask_cells=self._skin_cells(cx, cy),
            exclusion_zones=self._exclusion_zones(cx, cy),
        )

    def _landmarks(self, cx: float, cy: float, frame_index: int) -> tuple[Point | None, ...]:
        pts: list[Point | None] = []
        for k in range(6):
            angle = math.pi * (0.2 + 0.6 * k / 5.0)
            pts.append((cx + 0.09 * math.cos(angle), cy + 0.11 * math.sin(angle)))
        pts.append((cx - 0.035, cy + 0.16))
        pts.append((cx + 0.035, cy + 0.16))
        if frame_index > 0 and frame_index % self.LANDMARK_DROP_MODULUS == 0:
            pts[2] = None
        return tuple(pts)

    @staticmethod
    def _skin_cells(cx: float, cy: float) -> tuple[Rect, ...]:
        cells: list[Rect] = []
        for row in range(3):
            for col in range(3):
                cells.append((cx - 0.07 + col * 0.07, cy - 0.07 + row * 0.07, 0.06, 0.06))
        return tuple(cells)

    @staticmethod
    def _exclusion_zones(cx: float, cy: float) -> tuple[Rect, ...]:
        return (
            (cx - 0.060, cy - 0.050, 0.045, 0.030),
            (cx + 0.015, cy - 0.050, 0.045, 0.030),
            (cx - 0.030, cy + 0.055, 0.060, 0.035),
        )


class MediaPipeFaceAnalysisProvider:
    """STUB for the real MediaPipe FaceMesh + OpenCV backend (optional extra).

    No inference is implemented yet and nothing is downloaded. Constructing
    without the optional ``mediapipe``/``cv2`` packages raises ``ImportError``.
    """

    def __init__(self) -> None:
        if cv2 is None or mediapipe is None:
            raise ImportError(
                "MediaPipeFaceAnalysisProvider requires the optional "
                "'mediapipe' and 'cv2' (opencv-python) packages"
            )

    def analyze_frame(self, frame_index: int, timestamp: float) -> FaceFrame | None:
        """Stub: always raises until wired to ``mediapipe`` FaceMesh."""
        raise NotImplementedError(
            "MediaPipeFaceAnalysisProvider is a stub; wire to mediapipe FaceMesh"
        )


def default_provider() -> FaceAnalysisProvider:
    """Return the default provider: the deterministic :class:`MockFaceAnalysisProvider`."""
    return MockFaceAnalysisProvider()


def real_tracking_available() -> bool:
    """Return True if the optional real backend (``mediapipe`` + ``cv2``) is importable.

    Never raises and never imports at call time — the guarded import already
    ran at module load — so callers (e.g. the GUI status line) can poll it
    without pulling in MediaPipe themselves.
    """
    return cv2 is not None and mediapipe is not None


def select_provider(prefer_real: bool = False) -> FaceAnalysisProvider:
    """Pick a face-analysis provider, degrading gracefully to the offline mock.

    ``prefer_real=False`` (the default) returns :class:`MockFaceAnalysisProvider`.
    ``prefer_real=True`` returns :class:`MediaPipeFaceAnalysisProvider` when the
    optional dependencies are importable; otherwise it logs a warning and
    returns the mock. Never raises ``ImportError`` to callers and never triggers
    network access or downloads.
    """
    if prefer_real and real_tracking_available():
        return MediaPipeFaceAnalysisProvider()
    if prefer_real:
        logger.warning("Real face tracking unavailable — using offline fallback")
    return MockFaceAnalysisProvider()
