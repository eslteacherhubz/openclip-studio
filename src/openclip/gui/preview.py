"""Video preview widget: real frame scrubbing + play/pause (CORE v1, no audio).

Decodes via OpenCV's bundled FFmpeg (cv2.VideoCapture) and renders each frame
onto a scaled QLabel. Playback is driven by a QTimer ticked at the source fps.
Seek jumps to an absolute frame index. Designed to degrade gracefully: when
OpenCV is unavailable the widget shows a placeholder and disables all controls
rather than crashing.

No audio in v1 — cosmetic playback only; crop/beauty overlays come later.
"""

from __future__ import annotations

import contextlib
import os

try:
    import cv2
    import numpy as np

    _HAS_VISION = True
except ImportError:  # pragma: no cover - environment-specific
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _HAS_VISION = False

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def vision_available() -> bool:
    """Return True if OpenCV + numpy are importable for frame decoding."""
    return _HAS_VISION


class VideoPreviewWidget(QWidget):
    """Real video preview: decode via cv2, display frames, scrub + play/pause."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cap: cv2.VideoCapture | None = None
        self._path: str | None = None
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._duration: float = 0.0
        self._current_index: int = 0
        self._playing: bool = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        # ---- UI ----
        self._video_label = QLabel("Preview requires opencv (vision extra)")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(160, 120)

        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_play)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self._on_seek)

        self.lbl_time = QLabel("00:00 / 00:00")

        layout = QVBoxLayout(self)
        layout.addWidget(self._video_label, 1)
        controls = QHBoxLayout()
        controls.addWidget(self.btn_play)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.lbl_time)
        layout.addLayout(controls)

        if not _HAS_VISION:
            self._set_disabled_state()

    # ---------- read-only properties ----------
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def is_playing(self) -> bool:
        return self._playing

    # ---------- disabled state (no vision) ----------
    def _set_disabled_state(self) -> None:
        self.btn_play.setEnabled(False)
        self.slider.setEnabled(False)
        self._video_label.setEnabled(False)

    # ---------- lifecycle ----------
    def load(self, path: str | os.PathLike[str]) -> None:
        """Open ``path`` with cv2 and prepare the preview (frame 0 shown)."""
        self.close_capture()
        if not _HAS_VISION:
            return
        self._path = os.fspath(path)
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self._cap = None
            self.btn_play.setEnabled(False)
            self.slider.setEnabled(False)
            self._video_label.setText("Cannot open video: " + self._path)
            return
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._duration = self._frame_count / self._fps if self._fps > 0 else 0.0
        self._current_index = 0
        self.slider.setRange(0, max(0, self._frame_count - 1))
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.btn_play.setEnabled(True)
        self._show_frame(0)
        self._update_readout()

    def close_capture(self) -> None:
        """Stop playback and release the cv2 capture (idempotent)."""
        self._playing = False
        self._timer.stop()
        self.btn_play.setText("Play")
        if self._cap is not None:
            with contextlib.suppress(Exception):
                self._cap.release()
            self._cap = None

    # ---------- frame display ----------
    def _show_frame(self, index: int) -> None:
        if self._cap is None or not _HAS_VISION:
            return
        last = max(0, self._frame_count - 1)
        index = max(0, min(index, last))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        self._current_index = index
        self.slider.setValue(index)
        self._render_frame(frame)
        self._update_readout()

    def _render_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ch = rgb.shape
        bytes_per_line = 3 * w
        # tobytes() copies pixel data so QImage owns a stable buffer.
        qimg = QImage(
            rgb.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)

    def _update_readout(self) -> None:
        cur = self._current_index / self._fps if self._fps > 0 else 0.0
        self.lbl_time.setText(f"{self._fmt(cur)} / {self._fmt(self._duration)}")

    @staticmethod
    def _fmt(seconds: float) -> str:
        seconds = max(0.0, seconds)
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    # ---------- playback controls ----------
    def _toggle_play(self) -> None:
        if not _HAS_VISION or self._cap is None:
            return
        self._playing = not self._playing
        self.btn_play.setText("Pause" if self._playing else "Play")
        if self._playing:
            interval = int(1000.0 / self._fps) if self._fps > 0 else 40
            self._timer.start(interval)
        else:
            self._timer.stop()

    def _on_tick(self) -> None:
        if self._cap is None:
            self._timer.stop()
            return
        self._show_frame(self._current_index + 1)
        if self._current_index >= max(0, self._frame_count - 1):
            self._playing = False
            self.btn_play.setText("Play")
            self._timer.stop()

    def _on_seek(self, value: int) -> None:
        if self._cap is None:
            return
        was_playing = self._playing
        if was_playing:
            self._timer.stop()
        self._show_frame(value)
        if was_playing:
            interval = int(1000.0 / self._fps) if self._fps > 0 else 40
            self._timer.start(interval)

    # ---------- Qt overrides ----------
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._cap is not None and _HAS_VISION:
            self._show_frame(self._current_index)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.close_capture()
        super().closeEvent(event)
