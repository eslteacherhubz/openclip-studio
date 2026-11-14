"""End-to-end tests for the `openclip ai-edit` CLI command (local_mock, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openclip.cli import main
from openclip.engine.ffmpeg import probe
from openclip.engine.synth import generate_lesson


@pytest.fixture(scope="module")
def lesson(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_lesson(tmp_path_factory.mktemp("cli-ai") / "lesson.mp4").path


def test_ai_edit_without_auto_accept_prints_review_json(
    lesson: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out.mp4"
    rc = main(
        [
            "ai-edit",
            "--media",
            str(lesson),
            "--prompt",
            "Make a short clip about the main topic",
            "-o",
            str(out),
            "--provider",
            "local_mock",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "local_mock"
    assert "Make a short clip about the main topic" in payload["prompt"]
    assert "plan" in payload and "review" in payload
    # mock plan: clip + cut + caption + zoom + broll
    kinds = {item["kind"] for item in payload["review"]}
    assert {"clip", "cut", "caption", "zoom", "broll"} <= kinds
    assert all(item["status"] == "pending" for item in payload["review"])
    assert not out.exists()  # no render without --auto-accept


def test_ai_edit_with_transcript_file(
    lesson: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("Hello everyone :: 大家好\n", encoding="utf-8")
    out = tmp_path / "out.mp4"
    rc = main(
        [
            "ai-edit",
            "--media",
            str(lesson),
            "--prompt",
            "highlight the greeting",
            "--transcript",
            str(transcript),
            "-o",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["source_duration_seconds"] == 14.0


def test_ai_edit_auto_accept_renders_mp4(lesson: Path, tmp_path: Path) -> None:
    out = tmp_path / "edited.mp4"
    rc = main(
        [
            "ai-edit",
            "--media",
            str(lesson),
            "--prompt",
            "Remove the silent pause and keep the rest",
            "-o",
            str(out),
            "--provider",
            "local_mock",
            "--auto-accept",
        ]
    )
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    info = probe(out)
    assert (info.width, info.height) == (320, 180)
    # mock plan keeps only the clip span 0–10s and cuts 2.0–3.0s inside it → ~9s
    assert 8.0 < info.duration_s < 10.0


def test_ai_edit_auto_accept_with_captions_burns_text(lesson: Path, tmp_path: Path) -> None:
    from openclip.engine.ffmpeg import run_ffmpeg

    def _frame(video: Path, t: float, out: Path) -> bytes:
        run_ffmpeg(
            ["-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(out)],
            timeout=120,
        )
        return out.read_bytes()

    base_args = [
        "ai-edit",
        "--media",
        str(lesson),
        "--prompt",
        "caption the greeting",
        "--provider",
        "local_mock",
        "--auto-accept",
    ]
    out_cap = tmp_path / "cap.mp4"
    rc = main([*base_args, "-o", str(out_cap), "--captions"])
    assert rc == 0
    out_plain = tmp_path / "plain.mp4"
    rc = main([*base_args, "-o", str(out_plain)])
    assert rc == 0
    # mock caption "Hello / Hola" covers source 0–2s → export 0–2s (inside clip 0–10)
    assert _frame(out_cap, 1.0, tmp_path / "c.png") != _frame(out_plain, 1.0, tmp_path / "p.png")
    # outside the caption span frames are identical
    assert _frame(out_cap, 6.0, tmp_path / "c2.png") == _frame(out_plain, 6.0, tmp_path / "p2.png")
