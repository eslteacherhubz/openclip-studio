"""Reframe preview widget: pure-vector time lane of the smart-reframe crop window.

Paints where the 9:16 crop window sits across the source frame over time,
derived only from :class:`~openclip.ai.reframing.TrackingPoint` data. No video
decoding, no network — safe to run fully offline.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from openclip.ai.reframing import (
    DEFAULT_TARGET_ASPECT,
    TrackingPoint,
    anchor_at,
    crop_size_for_aspect,
)

PLACEHOLDER_TEXT = "Load media to preview reframe"


class ReframePreviewWidget(QWidget):
    """Horizontal time lane showing the crop window band moving across the frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[TrackingPoint] = []
        self._duration: float = 0.0
        self._src_w: int = 1920
        self._src_h: int = 1080
        self._target_aspect: float = DEFAULT_TARGET_ASPECT
        self.setMinimumHeight(90)

    def set_tracking(
        self,
        points: list[TrackingPoint],
        duration: float,
        src_width: int,
        src_height: int,
        target_aspect_ratio: float = DEFAULT_TARGET_ASPECT,
    ) -> None:
        """Set the tracking data and media geometry, then repaint."""
        self._points = list(points)
        self._duration = duration
        self._src_w = max(1, src_width)
        self._src_h = max(1, src_height)
        self._target_aspect = target_aspect_ratio
        self.update()

    def clear(self) -> None:
        """Return to the neutral placeholder state."""
        self._points = []
        self._duration = 0.0
        self.update()

    @property
    def points(self) -> list[TrackingPoint]:
        return list(self._points)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        if not self._points or self._duration <= 0:
            painter.setPen(QColor(160, 160, 160))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), PLACEHOLDER_TEXT)
            painter.end()
            return
        margin = 12
        lane_h = 28
        lane_x0 = margin
        lane_x1 = max(margin + 1, rect.width() - margin)
        lane_w = lane_x1 - lane_x0
        lane_y = max(0, (rect.height() - lane_h) // 2 - 6)
        crop_w, crop_h = crop_size_for_aspect(self._src_w, self._src_h, self._target_aspect)
        painter.fillRect(lane_x0, lane_y, lane_w, lane_h, QColor(40, 44, 52))
        band_w = max(2, round(lane_w * crop_w / self._src_w))
        overlay = QColor(80, 160, 250, 60)
        steps = 60
        for i in range(steps + 1):
            t = self._duration * i / steps
            ax, _ay = anchor_at(self._points, t, self._src_w, self._src_h, crop_w, crop_h)
            x = lane_x0 + round(lane_w * ax / self._src_w)
            painter.fillRect(x, lane_y, band_w, lane_h, overlay)
        last_t = self._points[-1].t
        ax, _ay = anchor_at(self._points, last_t, self._src_w, self._src_h, crop_w, crop_h)
        x = lane_x0 + round(lane_w * ax / self._src_w)
        painter.setPen(QPen(QColor(120, 200, 255), 2))
        painter.drawRect(x, lane_y, band_w, lane_h)
        painter.setPen(QColor(200, 200, 200))
        ticks = 5
        for i in range(ticks + 1):
            t = self._duration * i / ticks
            x = lane_x0 + round(lane_w * t / self._duration)
            painter.drawLine(x, lane_y + lane_h, x, lane_y + lane_h + 6)
            painter.drawText(x - 20, lane_y + lane_h + 20, f"{t:.1f}s")
        painter.end()
