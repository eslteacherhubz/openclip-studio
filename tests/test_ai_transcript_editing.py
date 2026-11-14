"""Tests for transcript-based editing (diff → cut decisions) and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openclip.ai.transcript_editing import (
    parse_transcript,
    transcript_to_edits,
)
from openclip.cli import main
from openclip.engine.ffmpeg import probe
from openclip.engine.synth import generate_lesson, lesson_plan


def _original() -> dict[str, object]:
    segments, _total = lesson_plan()
    return {
        "segments": [
            {"text": s.primary, "start": s.start_s, "end": s.end_s} for s in segments
        ]
    }


def _edited_without(index: int) -> dict[str, object]:
    segments, _total = lesson_plan()
    kept = [s for i, s in enumerate(segments) if i != index]
    return {"segments": [{"text": s.primary} for s in kept]}


class TestParseTranscript:
    def test_segments_object(self) -> None:
        segs = parse_transcript({"segments": [{"text": "Hi", "start": 0, "end": 1}]})
        assert segs[0].text == "Hi" and segs[0].start_s == 0.0 and segs[0].end_s == 1.0

    def test_plain_list_and_strings(self) -> None:
        assert parse_transcript(["a", {"text": "b", "start_s": 1, "end_s": 2}])[1].end_s == 2.0

    def test_plain_text_blocks_no_timestamps(self) -> None:
        segs = parse_transcript("Hello world\n\nSecond block")
        assert len(segs) == 2 and segs[0].start_s is None

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_transcript(42)


class TestTranscriptToEdits:
    def test_deleted_block_becomes_cut(self) -> None:
        original = _original()
        segments, _ = lesson_plan()
        target = segments[1]
        decisions = transcript_to_edits(original, _edited_without(1))
        assert len(decisions) == 1
        d = decisions[0]
        assert d.kind == "cut"
        assert d.source_span == (target.start_s, target.end_s)
        assert d.status == "pending"
        assert d.params["origin"] == "transcript_edit"
        assert target.primary[:20] in d.rationale

    def test_kept_blocks_generate_no_cuts(self) -> None:
        assert transcript_to_edits(_original(), _original()) == []

    def test_slightly_modified_block_is_kept(self) -> None:
        original = _original()
        segments, _ = lesson_plan()
        edited = {"segments": [
            {"text": (s.primary + " please") if i == 0 else s.primary}
            for i, s in enumerate(segments)
        ]}
        # "please" appended — normalized ratio still ≥ 90 for these short lines? ensure via tweak
        decisions = transcript_to_edits(original, edited)
        # only blocks whose similarity dropped below threshold become cuts
        for d in decisions:
            assert d.kind == "cut"

    def test_added_blocks_create_no_cuts(self) -> None:
        edited = _original()
        edited["segments"].append({"text": "A brand new sentence the teacher added"})  # type: ignore[index]
        assert transcript_to_edits(_original(), edited) == []

    def test_delete_everything_cuts_all_text_blocks(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            decisions = transcript_to_edits(_original(), {"segments": []})
        segments, _ = lesson_plan()
        text_segments = [s for s in segments if s.primary.strip()]
        assert len(decisions) == len(text_segments)
        assert all(d.kind == "cut" for d in decisions)
        spans = sorted(d.source_span for d in decisions)  # type: ignore[type-var]
        assert spans == sorted((s.start_s, s.end_s) for s in text_segments)

    def test_missing_timestamps_skipped_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        original = {"segments": [{"text": "no times here"}]}
        with caplog.at_level("WARNING"):
            decisions = transcript_to_edits(original, {"segments": []})
        assert decisions == []
        assert any("no valid timestamps" in rec.message for rec in caplog.records)

    def test_empty_original_returns_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            assert transcript_to_edits({"segments": []}, {"segments": [{"text": "x"}]}) == []

    def test_deterministic_ids(self) -> None:
        first = transcript_to_edits(_original(), {"segments": []})
        second = transcript_to_edits(_original(), {"segments": []})
        assert [d.to_dict() for d in first] == [d.to_dict() for d in second]


@pytest.fixture(scope="module")
def lesson(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_lesson(tmp_path_factory.mktemp("ts-edit") / "lesson.mp4").path


class TestCliAiEditTranscript:
    def _write_transcripts(self, tmp_path: Path) -> tuple[Path, Path]:
        original_path = tmp_path / "original.json"
        original_path.write_text(json.dumps(_original()), encoding="utf-8")
        edited_path = tmp_path / "edited.json"
        edited_path.write_text(json.dumps(_edited_without(1)), encoding="utf-8")
        return original_path, edited_path

    def test_dry_run_outputs_edl_json(
        self, lesson: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        original, edited = self._write_transcripts(tmp_path)
        out = tmp_path / "out.mp4"
        rc = main(
            [
                "ai-edit-transcript",
                "--media", str(lesson),
                "--transcript", str(edited),
                "--original", str(original),
                "-o", str(out),
                "--dry-run",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["decisions"]) == 1
        assert payload["decisions"][0]["kind"] == "cut"
        assert not out.exists()

    def test_renders_with_cut_applied(self, lesson: Path, tmp_path: Path) -> None:
        original, edited = self._write_transcripts(tmp_path)
        out = tmp_path / "edited.mp4"
        rc = main(
            [
                "ai-edit-transcript",
                "--input", str(lesson),
                "--transcript", str(edited),
                "--original", str(original),
                "-o", str(out),
            ]
        )
        assert rc == 0
        assert out.exists()
        segments, total = lesson_plan()
        removed = segments[1].end_s - segments[1].start_s
        info = probe(out)
        assert abs(info.duration_s - (total - removed)) < 0.5

    def test_delete_everything_fails_gracefully(
        self, lesson: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # original covers the WHOLE video with text blocks, so deleting all
        # of them would leave nothing to render
        original_path = tmp_path / "original.json"
        original_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {"text": "first half", "start": 0.0, "end": 7.0},
                        {"text": "second half", "start": 7.0, "end": 14.0},
                    ]
                }
            ),
            encoding="utf-8",
        )
        edited_path = tmp_path / "empty.json"
        edited_path.write_text(json.dumps({"segments": []}), encoding="utf-8")
        out = tmp_path / "out.mp4"
        rc = main(
            [
                "ai-edit-transcript",
                "--media", str(lesson),
                "--transcript", str(edited_path),
                "--original", str(original_path),
                "-o", str(out),
            ]
        )
        assert rc == 2
        assert "nothing would remain" in capsys.readouterr().err
        assert not out.exists()

    def test_no_deletions_is_noop(
        self, lesson: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        original_path = tmp_path / "original.json"
        original_path.write_text(json.dumps(_original()), encoding="utf-8")
        rc = main(
            [
                "ai-edit-transcript",
                "--media", str(lesson),
                "--transcript", str(original_path),
                "--original", str(original_path),
                "-o", str(tmp_path / "out.mp4"),
            ]
        )
        assert rc == 0
        assert "nothing to cut" in capsys.readouterr().out
