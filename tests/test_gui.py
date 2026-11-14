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
    script_text = "\n\n".join(f"{s.primary} :: {s.support}" for s in segments if s.kind == "speech")
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


def test_ai_review_table_populates_and_toggles(window) -> None:
    import json

    from PySide6.QtCore import Qt

    from openclip.ai import (
        MockAIProvider,
        generate_plan,
        load_provider_config,
        plan_to_review_items,
    )

    provider = MockAIProvider(load_provider_config("local_mock", env={}))
    plan = generate_plan(provider, "short clip", "transcript", 14.0)
    window._ai_plan = plan
    window._refresh_ai_table()

    items = plan_to_review_items(plan)
    assert window.ai_table.rowCount() == len(items)
    kinds = {window.ai_table.item(r, 1).text() for r in range(window.ai_table.rowCount())}
    assert {"clip", "cut", "caption", "zoom", "broll"} <= kinds
    # every proposal starts PENDING → checkbox checked (proposed accept)
    assert all(
        window.ai_table.item(r, 0).checkState() == Qt.CheckState.Checked
        for r in range(window.ai_table.rowCount())
    )
    assert all(window._ai_acceptance())

    # reject the first row; acceptance list must reflect the toggle
    window.ai_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    acceptance = window._ai_acceptance()
    assert acceptance[0] is False
    assert all(acceptance[1:])

    # decisions built from the plan stay pending until the user renders
    from openclip.ai import ai_plan_to_director_decisions

    decisions = ai_plan_to_director_decisions(plan)
    assert all(d.status == "pending" for d in decisions)
    assert (
        len(decisions)
        == len(json.loads(json.dumps(plan.to_dict()))["clips"])
        + sum(len(plan.to_dict()[k]) for k in ("cuts", "captions", "broll", "zooms"))
        + 2
    )  # audio + export


def test_ai_panel_has_byok_provider_selector_and_secure_key(window) -> None:
    from PySide6.QtWidgets import QLineEdit

    from openclip.ai.config import PROVIDER_NAMES

    providers = [window.cmb_ai_provider.itemText(i) for i in range(window.cmb_ai_provider.count())]
    assert providers == list(PROVIDER_NAMES)
    for expected in ("local_mock", "openai", "anthropic", "ollama"):
        assert expected in providers
    assert window.cmb_ai_provider.currentText() == "local_mock"
    # API key field masks input (BYOK) and starts empty
    assert window.edit_ai_key.echoMode() == QLineEdit.EchoMode.Password
    assert window.edit_ai_key.text() == ""


def test_provider_env_overrides_maps_byok_key() -> None:
    if not _qt_available():
        pytest.skip("PySide6 (gui extra) not installed")
    from openclip.gui.workers import provider_env_overrides

    assert provider_env_overrides("openai", None) == {}
    assert provider_env_overrides("openai", "") == {}
    assert provider_env_overrides("openai", "sk-abc") == {"OPENAI_API_KEY": "sk-abc"}
    assert provider_env_overrides("anthropic", "sk-ant") == {"ANTHROPIC_API_KEY": "sk-ant"}
    assert provider_env_overrides("ollama", "anything") == {}  # local: no key var
    assert provider_env_overrides("local_mock", "x") == {}


def test_transcript_editor_apply_feeds_ai_review(window) -> None:
    original = {
        "segments": [
            {"text": "Hello class", "start": 0.0, "end": 2.0},
            {"text": "Second line", "start": 2.0, "end": 4.0},
            {"text": "Third line", "start": 4.0, "end": 6.0},
        ]
    }
    window._original_transcript = original
    window.transcript_edit.setPlainText("Hello class\n\nThird line")  # deleted block 2
    window.apply_transcript_edits()

    cuts = [d for d in window._transcript_decisions if d.kind == "cut"]
    captions = [d for d in window._transcript_decisions if d.kind == "caption"]
    assert len(cuts) == 1
    assert cuts[0].source_span == (2.0, 4.0)
    # kept blocks auto-generate caption decisions (Objective 1)
    assert len(captions) == 2
    assert {c.source_span for c in captions} == {(0.0, 2.0), (4.0, 6.0)}

    # the cut is surfaced in the AI review panel, pending, accepted-by-default checkbox
    assert window.ai_table.rowCount() == 3  # 1 cut + 2 captions
    kinds = {window.ai_table.item(r, 1).text() for r in range(3)}
    assert kinds == {"cut", "caption"}
    cut_row = next(r for r in range(3) if window.ai_table.item(r, 1).text() == "cut")
    assert "Second line" in window.ai_table.item(cut_row, 4).text()
    assert window.btn_ai_render.isEnabled()

    # rejecting the row is reflected in the acceptance list
    from PySide6.QtCore import Qt

    window.ai_table.item(cut_row, 0).setCheckState(Qt.CheckState.Unchecked)
    assert window._ai_acceptance()[cut_row] is False


def test_transcript_editor_no_deletions(window) -> None:
    original = {"segments": [{"text": "Only line", "start": 0.0, "end": 2.0}]}
    window._original_transcript = original
    window.transcript_edit.setPlainText("Only line")
    window.apply_transcript_edits()
    # no cuts, but the single kept block still yields one caption decision
    assert [d.kind for d in window._transcript_decisions] == ["caption"]
    assert window.ai_table.rowCount() == 1
    assert window.btn_ai_render.isEnabled()


def test_gui_entry_declines_without_media() -> None:
    if not _qt_available():
        pytest.skip("PySide6 (gui extra) not installed")
    from openclip.gui.main_window import MainWindow

    win = MainWindow()
    assert win.project is None and win.media is None
    assert not win.btn_export.isEnabled()
    win.close()
