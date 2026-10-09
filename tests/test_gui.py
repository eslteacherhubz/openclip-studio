"""GUI tests: offscreen-safe, skipped without the ``gui`` extra."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Must be set before QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.gui


def _qt_available() -> bool:
    try:
        import PySide6  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def qapp():
    if not _qt_available():
        pytest.skip("PySide6 (gui extra) not installed")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def lesson_media(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from openclip.engine.synth import generate_lesson

    return generate_lesson(tmp_path_factory.mktemp("gui") / "lesson.mp4").path


@pytest.fixture
def window(qapp, lesson_media, tmp_path):
    from openclip.engine.project import Project
    from openclip.gui.main_window import MainWindow

    win = MainWindow()
    win.project = Project.create(source=lesson_media, script_text="")
    from openclip.engine.ffmpeg import probe

    win.media = probe(lesson_media)
    yield win
    win.close()


def test_main_window_constructs(window) -> None:
    assert window.windowTitle() == "Eterna OpenClip Studio"
    assert window.script_edit is not None
    assert window.cut_table.columnCount() == 5


def test_script_sync_and_settings(window) -> None:
    window.script_edit.setPlainText("Hello class :: 大家好\n\nSecond line")
    window.cmb_animation.setCurrentText("pop")
    window.chk_karaoke.setChecked(True)
    window.cmb_audio.setCurrentText("denoise")
    assert window._sync_script_into_project()
    assert len(window.project.script) == 2
    assert window.project.script.lines[0].support == "大家好"
    assert window.project.captions.animation == "pop"
    assert window.project.captions.karaoke is True
    assert window.project.audio_preset == "denoise"


def test_analysis_populates_cut_table(window) -> None:
    from openclip.engine.pipeline import run_pipeline, spans_to_hyps
    from openclip.engine.synth import SynthLesson, lesson_plan, lesson_word_hypotheses

    segments, total = lesson_plan()
    script_text = "\n\n".join(
        f"{s.primary} :: {s.support}" for s in segments if s.kind == "speech"
    )
    window.script_edit.setPlainText(script_text)
    window._sync_script_into_project()
    synth = SynthLesson(path=window.media.path, duration_s=total, segments=segments)
    hyps = spans_to_hyps(lesson_word_hypotheses(synth))
    run_pipeline(window.project, media=window.media, transcriber="mock", hyps=hyps)
    window._refresh_cut_table()
    assert window.cut_table.rowCount() >= 4
    kinds = {window.cut_table.item(r, 1).text() for r in range(window.cut_table.rowCount())}
    assert "silence" in kinds and "offscript" in kinds

    # Toggle first row to KEEP, then sync acceptance.
    from PySide6.QtCore import Qt

    window.cut_table.item(0, 0).setCheckState(Qt.CheckState.Checked)
    window._cut_acceptance_from_table()
    assert window.project.cut_plan.cuts[0].accepted is False


def test_gui_entry_declines_without_media() -> None:
    from openclip.gui.main_window import MainWindow

    win = MainWindow()
    assert win.project is None and win.media is None
    assert not win.btn_export.isEnabled()
    win.close()
