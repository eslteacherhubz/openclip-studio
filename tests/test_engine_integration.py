"""Integration tests: real FFmpeg renders on synthetic media (generated here)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openclip.engine.ffmpeg import detect_silences, probe, run_ffmpeg_capture
from openclip.engine.pipeline import run_pipeline, spans_to_hyps
from openclip.engine.project import Project
from openclip.engine.render import RenderOptions, render_project
from openclip.engine.synth import SynthLesson, generate_lesson, lesson_word_hypotheses


@pytest.fixture(scope="module")
def lesson(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The deterministic synthetic lesson shared by this module."""
    out = tmp_path_factory.mktemp("lesson") / "lesson.mp4"
    generate_lesson(out)
    return out


def test_probe_synthetic_lesson(lesson: Path) -> None:
    info = probe(lesson)
    assert info.has_video and info.has_audio
    assert info.width == 320 and info.height == 180
    assert abs(info.duration_s - 14.0) < 0.35
    assert info.vcodec == "h264"
    assert info.acodec == "aac"
    assert 14.0 <= info.fps <= 16.0
    # Both probe paths must agree.
    from openclip.engine.ffmpeg import _probe_ffmpeg

    fallback = _probe_ffmpeg(lesson)
    assert abs(fallback.duration_s - info.duration_s) < 0.5
    assert fallback.width == info.width


def test_silence_detection_on_synthetic_gaps(lesson: Path) -> None:
    spans = detect_silences(lesson, noise_db=-35.0, min_duration_s=0.5)
    # Plan has 4 silences: 0.6, 0.8, 1.6, 1.8 seconds.
    assert len(spans) == 4
    mid = [s for s in spans if abs(s.start_s - 8.6) < 0.15]
    assert mid, f"long mid-lesson gap not found in {[ (s.start_s, s.end_s) for s in spans]}"
    assert abs(mid[0].end_s - 10.2) < 0.15
    tail = [s for s in spans if s.start_s > 11.0]
    assert tail and tail[0].duration > 1.5


def test_full_pipeline_render(lesson: Path, tmp_path: Path) -> None:
    from openclip.engine.synth import lesson_plan

    segments, _total = lesson_plan()
    script_text = "\n\n".join(
        f"{s.primary} :: {s.support}" if s.support else s.primary
        for s in segments
        if s.kind == "speech"
    )
    project = Project.create(source=lesson, script_text=script_text)
    media = probe(lesson)
    synth = SynthLesson(path=lesson, duration_s=14.0, segments=segments)
    hyps = spans_to_hyps(lesson_word_hypotheses(synth))
    outcome = run_pipeline(project, media=media, transcriber="mock", hyps=hyps)

    assert outcome.transcriber == "mock"
    assert outcome.words_found > 0
    kinds = {c.kind for c in outcome.cut_plan.cuts}
    assert "silence" in kinds
    assert "offscript" in kinds
    # All 4 sentences aligned within tolerance.
    aligned = [ln for ln in outcome.alignment.lines if ln.confidence > 0]
    assert len(aligned) == 4
    speech_segments = [s for s in segments if s.kind == "speech"]
    for ln, seg in zip(aligned, speech_segments, strict=True):
        assert abs(ln.start_s - seg.start_s) < 0.4, f"line {ln.index} start drift"
        assert abs(ln.end_s - seg.end_s) < 0.4, f"line {ln.index} end drift"

    out = tmp_path / "export.mp4"
    result = render_project(
        project, media, out, RenderOptions(burn_captions=True, audio_preset="teaching")
    )
    exported = probe(out)
    assert exported.vcodec == "h264"
    assert exported.acodec == "aac"
    assert exported.pix_fmt == "yuv420p"
    assert exported.has_audio and exported.has_video
    assert exported.width == 320 and exported.height == 180
    # Duration must match planned kept time within ~0.5s.
    assert abs(exported.duration_s - result.kept_duration_s) < 0.5
    # The mistake span (5.2..6.4) and long silence (8.6..10.2) must be gone.
    assert result.cuts_applied >= 4
    # Off-script cut should exist around the flub.
    offscript = [c for c in outcome.cut_plan.cuts if c.kind == "offscript"]
    assert offscript and offscript[0].start_s > 4.5 and offscript[0].end_s < 7.0


def test_captions_actually_burned(lesson: Path, tmp_path: Path) -> None:
    """Render twice; the captioned export must differ where captions live."""
    from openclip.engine.synth import lesson_plan

    segments, _total = lesson_plan()
    script_text = "\n\n".join(
        f"{s.primary} :: {s.support}" if s.support else s.primary
        for s in segments
        if s.kind == "speech"
    )
    project = Project.create(source=lesson, script_text=script_text)
    media = probe(lesson)
    synth = SynthLesson(path=lesson, duration_s=14.0, segments=segments)
    hyps = spans_to_hyps(lesson_word_hypotheses(synth))
    run_pipeline(project, media=media, transcriber="mock", hyps=hyps)

    out_plain = tmp_path / "plain.mp4"
    out_caps = tmp_path / "caps.mp4"
    render_project(project, media, out_plain, RenderOptions(burn_captions=False))
    render_project(project, media, out_caps, RenderOptions(burn_captions=True))
    assert out_plain.stat().st_size > 0 and out_caps.stat().st_size > 0

    # PSNR between the two renders: identical video except burned text.
    proc = run_ffmpeg_capture(
        [
            "-i",
            str(out_plain),
            "-i",
            str(out_caps),
            "-lavfi",
            "psnr",
            "-f",
            "null",
            "-",
        ]
    )
    assert proc.returncode == 0
    import re

    m = re.search(r"average:([\d.]+)", proc.stderr, re.IGNORECASE)
    assert m, "no PSNR average in output"
    psnr = float(m.group(1))
    # Captions must change pixels (finite PSNR) but not wreck the video (<50 dB).
    assert 12.0 < psnr < 50.0


def test_project_roundtrip(lesson: Path, tmp_path: Path) -> None:
    project = Project.create(
        source=lesson, script_text="Hello :: 你好\n\nBye :: 再见"
    )
    project.audio_preset = "denoise"
    project.captions.karaoke = True
    path = tmp_path / "p.json"
    project.save(path)
    loaded = Project.load(path)
    assert loaded.source == Path(lesson)
    assert len(loaded.script) == 2
    assert loaded.script.lines[0].support == "你好"
    assert loaded.audio_preset == "denoise"
    assert loaded.captions.karaoke is True


def test_cli_end_to_end(lesson: Path, tmp_path: Path) -> None:
    """The exact headless path CI will exercise: synth -> new -> auto."""
    from openclip.cli import main as cli_main

    script = tmp_path / "script.txt"
    script.write_text(
        "Hello everyone, welcome to the lesson. :: 大家好，欢迎来到课程。\n"
        "\n"
        "Today we will learn five new words. :: 今天我们学习五个新单词。\n"
        "\n"
        "Please listen and repeat after me. :: 请听录音然后跟读。\n"
        "\n"
        "Great job, see you next time. :: 做得好，下次见。\n",
        encoding="utf-8",
    )
    project = tmp_path / "project.json"
    assert (
        cli_main(["new", "--media", str(lesson), "--script", str(script), "-o", str(project)])
        == 0
    )
    out = tmp_path / "final.mp4"
    assert cli_main(["auto", str(project), "-o", str(out), "-t", "even"]) == 0
    exported = probe(out)
    assert exported.vcodec == "h264" and exported.acodec == "aac"
    data = json.loads(project.read_text(encoding="utf-8"))
    assert data["cut_plan"], "even-mode pipeline must still produce silence cuts"
