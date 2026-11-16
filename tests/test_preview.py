"""Tests for openclip.gui.preview.

- gui marker: widget construction + cv2-absent degradation (PySide6 only).
- gaze marker: real clip decode/seek/play-pause (PySide6 + cv2).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

try:
    from PySide6.QtWidgets import QApplication

    _QT_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-specific
    QApplication = None  # type: ignore[assignment]
    _QT_AVAILABLE = False


@pytest.fixture(scope="module")
def qapp():
    if not _QT_AVAILABLE:
        pytest.skip("PySide6 (gui extra) not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.gui
class TestPreviewConstruct:
    """Construction + cv2-absent degradation (gui job: PySide6 only)."""

    def test_widget_constructs_without_media(self, qapp) -> None:
        from openclip.gui.preview import VideoPreviewWidget

        w = VideoPreviewWidget()
        assert w.fps == 0.0
        assert w.frame_count == 0
        assert w.duration == 0.0
        assert w.current_index == 0
        assert not w.is_playing
        assert not w.btn_play.isEnabled()
        assert not w.slider.isEnabled()
        assert "Preview requires opencv" in w._video_label.text()
        w.close()

    def test_cv2_absent_shows_placeholder_and_disables(
        self, qapp, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import openclip.gui.preview as preview_mod

        monkeypatch.setattr(preview_mod, "_HAS_VISION", False)
        monkeypatch.setattr(preview_mod, "vision_available", lambda: False)
        from openclip.gui.preview import VideoPreviewWidget

        assert not preview_mod.vision_available()
        w = VideoPreviewWidget()
        assert not w.btn_play.isEnabled()
        assert not w.slider.isEnabled()
        assert "Preview requires opencv (vision extra)" in w._video_label.text()
        # load() must not crash without cv2
        w.load("/nonexistent/clip.mp4")
        assert w._cap is None
        w.close()


@pytest.mark.gaze
class TestPreviewRealClip:
    """Real decode/seek/play-pause (gaze job: cv2 present; skips without PySide6)."""

    @staticmethod
    def _make_clip(path: Path, duration_s: float = 2.0, fps: int = 25,
                   width: int = 320, height: int = 240) -> None:
        pytest.importorskip("cv2")
        args = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            f"testsrc=duration={duration_s}:size={width}x{height}:rate={fps}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-an",
            str(path),
        ]
        res = subprocess.run(args, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr.strip() or res.stdout.strip()
        assert path.exists() and path.stat().st_size > 0

    def test_load_reads_fps_and_frame_count(
        self, qapp, tmp_path: Path
    ) -> None:
        pytest.importorskip("cv2")
        from openclip.gui.preview import VideoPreviewWidget

        clip = tmp_path / "clip.mp4"
        self._make_clip(clip, duration_s=2.0, fps=25, width=320, height=240)
        w = VideoPreviewWidget()
        try:
            w.load(str(clip))
            assert w.fps == 25.0, w.fps
            assert w.frame_count == 50, w.frame_count
            assert abs(w.duration - 2.0) < 0.1, w.duration
            assert w.current_index == 0
        finally:
            w.close_capture()
            w.close()

    def test_seek_updates_index(self, qapp, tmp_path: Path) -> None:
        pytest.importorskip("cv2")
        from openclip.gui.preview import VideoPreviewWidget

        clip = tmp_path / "clip.mp4"
        self._make_clip(clip, duration_s=2.0, fps=25, width=320, height=240)
        w = VideoPreviewWidget()
        try:
            w.load(str(clip))
            w._on_seek(12)
            assert w.current_index == 12
            assert w.slider.value() == 12
            w._on_seek(0)
            assert w.current_index == 0
        finally:
            w.close_capture()
            w.close()

    def test_play_pause_toggles_timer(self, qapp, tmp_path: Path) -> None:
        pytest.importorskip("cv2")
        from openclip.gui.preview import VideoPreviewWidget

        clip = tmp_path / "clip.mp4"
        self._make_clip(clip, duration_s=2.0, fps=25, width=320, height=240)
        w = VideoPreviewWidget()
        try:
            w.load(str(clip))
            assert not w.is_playing
            assert not w._timer.isActive()
            w._toggle_play()
            assert w.is_playing
            assert w._timer.isActive()
            assert w.btn_play.text() == "Pause"
            w._toggle_play()
            assert not w.is_playing
            assert not w._timer.isActive()
            assert w.btn_play.text() == "Play"
        finally:
            w.close_capture()
            w.close()
