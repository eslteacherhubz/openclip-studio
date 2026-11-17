"""Per-frame face analysis for reframing: anchor, landmarks, mask, exclusion zones.

The reframing layer consumes one :class:`FaceFrame` per video frame. All
coordinates are normalized to the unit square (0-1 relative to frame width
and height). Providers implement :class:`FaceAnalysisProvider`;
:class:`MockFaceAnalysisProvider` is the deterministic default so the pipeline
runs fully offline, and :class:`MediaPipeFaceAnalysisProvider` is the real
MediaPipe/OpenCV backend behind an optional, guarded import.

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
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from openclip.ai.smoothing import DEFAULT_SMOOTH_ALPHA, EWMA

try:
    import cv2
    import mediapipe as _mp
    import numpy as np

    _HAS_MEDIAPIPE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _mp = None
    _HAS_MEDIAPIPE = False

logger = logging.getLogger(__name__)

#: Normalized 2D point ``(x, y)``, both components in 0-1.
Point = tuple[float, float]

#: Normalized rectangle ``(x, y, width, height)``, all components in 0-1.
Rect = tuple[float, float, float, float]

#: Path to the bundled MediaPipe FaceLandmarker model asset (task bundle).
#: No network/downloads happen at runtime — the model ships with the repo.
MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")

#: MediaPipe FaceMesh (468) landmark index sets, derived from
#: ``mediapipe.tasks.vision.FaceLandmarksConnections`` when available, with
#: hardcoded fallbacks so the mapping logic stays testable without mediapipe.


def _conn_indices(connections: Sequence[Any] | None) -> tuple[int, ...]:
    """Extract the unique landmark indices from a connections list."""
    idxs: set[int] = set()
    for conn in connections or ():
        idxs.add(int(getattr(conn, "start", 0)))
        idxs.add(int(getattr(conn, "end", 0)))
    return tuple(sorted(idxs))


_FLC = None
if _HAS_MEDIAPIPE and _mp is not None:
    try:
        _FLC = _mp.tasks.vision.FaceLandmarksConnections
    except Exception:  # pragma: no cover - environment-specific
        _FLC = None


def _region(name: str, fallback: tuple[int, ...]) -> tuple[int, ...]:
    """Resolve a FaceMesh region index set from mediapipe connections or fallback."""
    if _FLC is not None:
        conns = getattr(_FLC, name, None)
        if conns:
            return _conn_indices(conns)
    return fallback


#: Jaw / face-oval contour indices (used for jaw_neck_landmarks + face oval).
FACE_OVAL_IDX: tuple[int, ...] = _region(
    "FACE_LANDMARKS_FACE_OVAL",
    (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378),
)
#: Left eye region indices (for exclusion_zones).
LEFT_EYE_IDX: tuple[int, ...] = _region(
    "FACE_LANDMARKS_LEFT_EYE",
    (263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398),
)
#: Right eye region indices (for exclusion_zones).
RIGHT_EYE_IDX: tuple[int, ...] = _region(
    "FACE_LANDMARKS_RIGHT_EYE",
    (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
)
#: Lips / mouth region indices (for exclusion_zones).
LIPS_IDX: tuple[int, ...] = _region(
    "FACE_LANDMARKS_LIPS",
    (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317),
)
#: Nose indices (includes nostrils; used for exclusion_zones + center hint).
NOSE_IDX: tuple[int, ...] = _region(
    "FACE_LANDMARKS_NOSE",
    (168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 129, 15),
)


@dataclass(frozen=True)
class FaceFrame:
    """Analysis of a single video frame; every coordinate is normalized 0-1."""

    timestamp: float
    center_x: float
    center_y: float
    jaw_neck_landmarks: tuple[Point | None, ...]
    skin_mask_cells: tuple[Rect, ...]
    exclusion_zones: tuple[Rect, ...]


def _points_bbox(points: Sequence[Point]) -> Rect | None:
    """Return ``(x, y, w, h)`` normalized bbox of ``points``, or ``None`` if empty."""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _clip_rect(rect: Rect) -> Rect:
    """Clamp a normalized rect to the unit square with non-negative size."""
    x, y, w, h = rect
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    x2 = max(0.0, min(1.0, x + w))
    y2 = max(0.0, min(1.0, y + h))
    return (x, y, max(0.0, x2 - x), max(0.0, y2 - y))


def _inflate(rect: Rect, dx: float, dy: float) -> Rect:
    """Grow a normalized rect symmetrically, clamped to the unit square."""
    x, y, w, h = rect
    return _clip_rect((x - dx, y - dy, w + 2.0 * dx, h + 2.0 * dy))


def _build_face_frame(
    timestamp: float,
    landmarks_xy: Sequence[Point | None],
    width: int,
    height: int,
) -> FaceFrame:
    """Build a :class:`FaceFrame` from 468 MediaPipe FaceMesh landmarks.

    Pure function of the landmarks (+ image size): deterministic and testable
    with synthetic landmark arrays, no MediaPipe runtime required.
    ``width``/``height`` are accepted for API symmetry; landmarks from
    MediaPipe are already normalized 0-1 so the mapping is resolution-free.
    """

    def _pts(indices: tuple[int, ...]) -> list[Point]:
        out: list[Point] = []
        for i in indices:
            if i < len(landmarks_xy):
                p = landmarks_xy[i]
                if p is not None:
                    out.append((float(p[0]), float(p[1])))
        return out

    face_oval = _pts(FACE_OVAL_IDX)
    left_eye = _pts(LEFT_EYE_IDX)
    right_eye = _pts(RIGHT_EYE_IDX)
    lips = _pts(LIPS_IDX)
    nose = _pts(NOSE_IDX)

    # --- center: face-bbox center, falling back to nose, then frame center ---
    if face_oval:
        bbox = _points_bbox(face_oval)
        assert bbox is not None
        center_x = bbox[0] + bbox[2] / 2.0
        center_y = bbox[1] + bbox[3] / 2.0
    elif nose:
        bbox = _points_bbox(nose)
        assert bbox is not None
        center_x = bbox[0] + bbox[2] / 2.0
        center_y = bbox[1] + bbox[3] / 2.0
    else:
        center_x, center_y = 0.5, 0.5

    # --- jaw_neck_landmarks: face-oval points + two neck points below jaw ---
    jaw_neck: list[Point | None] = list(face_oval)
    if face_oval:
        bottom_y = max(p[1] for p in face_oval)
        cx = center_x
        jaw_neck.append((cx, min(1.0, bottom_y + 0.06)))  # neck top
        jaw_neck.append((cx, min(1.0, bottom_y + 0.14)))  # neck bottom
    jaw_neck_landmarks: tuple[Point | None, ...] = tuple(jaw_neck) if jaw_neck else ()

    # --- skin_mask_cells: forehead / left cheek / right cheek / chin ---
    if face_oval:
        f_bbox = _points_bbox(face_oval)
        assert f_bbox is not None
        f_w = f_bbox[2]
        f_h = f_bbox[3]
        f_cx = f_bbox[0] + f_w / 2.0
        f_cy = f_bbox[1] + f_h / 2.0
        cells = [
            _clip_rect((f_cx - 0.20 * f_w, f_bbox[1] - 0.10 * f_h, 0.40 * f_w, 0.20 * f_h)),
            _clip_rect((f_bbox[0] - 0.05 * f_w, f_cy - 0.15 * f_h, 0.25 * f_w, 0.40 * f_h)),
            _clip_rect((f_bbox[0] + 0.75 * f_w, f_cy - 0.15 * f_h, 0.25 * f_w, 0.40 * f_h)),
            _clip_rect((f_cx - 0.15 * f_w, f_bbox[3] - 0.05 * f_h, 0.30 * f_w, 0.20 * f_h)),
        ]
    else:
        cells = []
    skin_mask_cells: tuple[Rect, ...] = tuple(cells)

    # --- exclusion_zones: left eye, right eye, lips, nostrils ---
    zones: list[Rect] = []
    for pts in (left_eye, right_eye, lips):
        bbox = _points_bbox(pts)
        if bbox is not None:
            zones.append(_inflate(bbox, 0.02, 0.02))
    if nose:
        bbox = _points_bbox(nose)
        if bbox is not None:
            # nostrils: lower half of the nose bbox
            nostril_rect = (
                bbox[0],
                bbox[1] + 0.5 * bbox[3],
                bbox[2],
                max(1e-4, 0.5 * bbox[3]),
            )
            zones.append(_inflate(nostril_rect, 0.02, 0.02))
    exclusion_zones: tuple[Rect, ...] = tuple(zones)

    return FaceFrame(
        timestamp=timestamp,
        center_x=center_x,
        center_y=center_y,
        jaw_neck_landmarks=jaw_neck_landmarks,
        skin_mask_cells=skin_mask_cells,
        exclusion_zones=exclusion_zones,
    )


@runtime_checkable
class FaceAnalysisProvider(Protocol):
    """Source of per-frame face analysis."""

    def analyze_frame(
        self, frame_index: int, timestamp: float, frame: np.ndarray | None = None
    ) -> FaceFrame | None:
        """Return the analysis for ``frame_index``, or ``None`` if no face found.

        ``frame`` is the optional decoded source frame (BGR ndarray). Providers
        that need the pixels (e.g. MediaPipe) use it; providers that generate
        synthetic data (e.g. the mock) ignore it.
        """
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

    def analyze_frame(
        self, frame_index: int, timestamp: float, frame: np.ndarray | None = None
    ) -> FaceFrame | None:
        """Return the smoothed, cached analysis for ``frame_index``."""
        if frame_index in self._cache:
            return self._cache[frame_index]
        raw = self._provider.analyze_frame(frame_index, timestamp, frame)
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

    def analyze_frame(
        self, frame_index: int, timestamp: float, frame: np.ndarray | None = None
    ) -> FaceFrame | None:
        """Return the synthetic :class:`FaceFrame` for ``frame_index``.

        ``frame`` is ignored — the mock is deterministic and offline.
        """
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
    """Real MediaPipe FaceMesh backend (optional ``mediapipe`` extra).

    Uses :class:`mediapipe.tasks.vision.FaceLandmarker` with the bundled
    ``models/face_landmarker.task`` asset, so no network/downloads happen at
    runtime. Construct-time failures (missing deps or model asset) raise
    :class:`ImportError`; per-frame failures return ``None``.
    """

    def __init__(self) -> None:
        if not _HAS_MEDIAPIPE or _mp is None or cv2 is None or np is None:
            raise ImportError(
                "MediaPipeFaceAnalysisProvider requires the optional "
                "'mediapipe', 'cv2' (opencv-python) and 'numpy' packages"
            )
        if not os.path.exists(MODEL_PATH):
            raise ImportError(
                f"MediaPipe face landmarker model not found at {MODEL_PATH}; "
                "bundle models/face_landmarker.task with the repo"
            )
        # mediapipe is an untyped module; resolve task classes via attribute
        # access so mypy sees them as Any rather than import-untyped errors.
        BaseOptions = _mp.tasks.BaseOptions
        FaceLandmarker = _mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = _mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = _mp.tasks.vision.RunningMode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._video_timestamp_ms = 0

    def analyze_frame(
        self, frame_index: int, timestamp: float, frame: np.ndarray | None = None
    ) -> FaceFrame | None:
        """Run MediaPipe FaceLandmarker on ``frame`` and map to a :class:`FaceFrame`.

        Returns ``None`` when ``frame`` is not provided or no face is detected.
        ``timestamp`` (seconds) is converted to milliseconds for the video
        running mode.
        """
        if frame is None:
            return None
        try:
            self._video_timestamp_ms = round(timestamp * 1000.0)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(
                mp_image, self._video_timestamp_ms
            )
        except Exception:  # pragma: no cover - inference-time failure
            return None
        landmarks = result.face_landmarks
        if not landmarks:
            return None
        pts = landmarks[0]
        h, w = frame.shape[:2]
        landmarks_xy: list[Point | None] = []
        for p in pts:
            landmarks_xy.append((float(p.x), float(p.y)))
        return _build_face_frame(
            timestamp=timestamp,
            landmarks_xy=landmarks_xy,
            width=w,
            height=h,
        )


def default_provider() -> FaceAnalysisProvider:
    """Return the default provider: the deterministic :class:`MockFaceAnalysisProvider`."""
    return MockFaceAnalysisProvider()


def real_tracking_available() -> bool:
    """Return True if the optional real backend (``mediapipe`` + ``cv2`` + model) is usable.

    Never raises and never imports at call time — the guarded import already
    ran at module load — so callers (e.g. the GUI status line) can poll it
    without pulling in MediaPipe themselves.
    """
    return (
        _HAS_MEDIAPIPE
        and _mp is not None
        and cv2 is not None
        and np is not None
        and os.path.exists(MODEL_PATH)
    )


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
