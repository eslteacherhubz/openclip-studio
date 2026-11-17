"""Tests for rule-based hook/viral-moment detection and the `ai-clips` CLI."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from openclip.ai import ai_plan_to_director_decisions, plan_to_review_items
from openclip.ai.hook_detection import (
    HOOK_TYPES,
    Hook,
    HookDetector,
    hooks_to_director_plan,
)
from openclip.cli import main

FILLER = "Let's continue with today's lesson. Open your books to page ten."


def _seg(text: str, start: float, end: float, speaker: str | None = None) -> dict[str, object]:
    seg: dict[str, object] = {"text": text, "start": start, "end": end}
    if speaker is not None:
        seg["speaker"] = speaker
    return seg


def _hook_transcript() -> list[dict[str, object]]:
    """Five well-separated hook blocks (~120s apart) with neutral filler between."""
    segs: list[dict[str, object]] = []
    blocks: list[list[tuple[str, float, float]]] = [
        # question hook
        [
            ("What is the most important skill in life?", 0.0, 8.0),
            ("Why does it matter so much?", 8.0, 16.0),
            ("How can we actually learn it?", 16.0, 25.0),
        ],
        # shock hook
        [
            ("This is unbelievable! Nobody tells you this secret!", 120.0, 130.0),
            ("It is shocking, truly unreal!", 130.0, 140.0),
        ],
        # emotional hook
        [
            ("I love this! It is the most amazing thing ever.", 240.0, 250.0),
            ("Absolutely brilliant and fantastic, wow!", 250.0, 260.0),
        ],
        # pattern interrupt
        [
            ("Okay now we talk about grammar rules.", 360.0, 370.0),
            ("Listen, here's the thing — everything changes now.", 370.0, 382.0),
        ],
        # keyword-dense hook
        [
            ("Make money with this free hack for passive income.", 480.0, 490.0),
            ("Viral growth, more clients, success and power!", 490.0, 500.0),
        ],
    ]
    for block in blocks:
        block_start = block[0][1]
        block_end = block[-1][2]
        # neutral filler around each block so topic-shift has context
        if block_start >= 10.0:
            segs.append(_seg(FILLER, block_start - 10.0, block_start))
        for text, start, end in block:
            segs.append(_seg(text, start, end, speaker="teacher"))
        segs.append(_seg(FILLER, block_end, block_end + 10.0))
    segs.sort(key=lambda s: (s["start"], s["end"]))  # type: ignore[return-value, arg-type]
    return segs


class TestHookDetector:
    def test_detects_all_five_hook_types(self) -> None:
        hooks = HookDetector(_hook_transcript()).find_hooks(top_n=5)
        assert len(hooks) == 5
        assert {h.hook_type for h in hooks} == set(HOOK_TYPES)
        for hook in hooks:
            assert 15.0 <= hook.duration <= 90.0
            assert 0.0 < hook.score <= 1.0
            assert hook.transcript_snippet
            assert hook.end > hook.start

    def test_hooks_sorted_by_start_and_non_overlapping(self) -> None:
        hooks = HookDetector(_hook_transcript()).find_hooks(top_n=5)
        starts = [h.start for h in hooks]
        assert starts == sorted(starts)
        for a, b in itertools.pairwise(hooks):
            assert a.end <= b.start

    def test_top_n_limits_results(self) -> None:
        hooks = HookDetector(_hook_transcript()).find_hooks(top_n=2)
        assert len(hooks) == 2

    def test_ranking_prefers_higher_score(self) -> None:
        transcript = [
            _seg("Boring filler talk about pages.", 0.0, 20.0),
            _seg(
                "This is unbelievable! A secret nobody knows — "
                "shocking, insane, unreal, mind-blowing!",
                100.0,
                120.0,
            ),
        ]
        hooks = HookDetector(transcript).find_hooks(top_n=1)
        assert len(hooks) == 1
        assert hooks[0].start == 100.0
        assert hooks[0].hook_type == "shock"
        assert hooks[0].score == 1.0

    def test_deterministic(self) -> None:
        detector = HookDetector(_hook_transcript())
        assert detector.find_hooks(top_n=5) == detector.find_hooks(top_n=5)

    def test_empty_transcript(self) -> None:
        assert HookDetector([]).find_hooks() == []

    def test_untimed_segments_ignored(self) -> None:
        transcript: list[dict[str, object]] = [{"text": "What is this?"}]  # no start/end
        assert HookDetector(transcript).find_hooks() == []

    def test_start_end_aliases(self) -> None:
        transcript: list[dict[str, object]] = [
            {"text": "Why does this matter so much?", "start_s": 0.0, "end_s": 20.0},
            {"text": "How do we fix it?", "start_seconds": 20.0, "end_seconds": 35.0},
        ]
        hooks = HookDetector(transcript).find_hooks(top_n=1)
        assert len(hooks) == 1
        assert hooks[0].hook_type == "question"

    def test_invalid_params_raise(self) -> None:
        detector = HookDetector(_hook_transcript())
        with pytest.raises(ValueError):
            detector.find_hooks(min_duration=0.0)
        with pytest.raises(ValueError):
            detector.find_hooks(min_duration=90.0, max_duration=15.0)
        with pytest.raises(ValueError):
            detector.find_hooks(top_n=0)

    def test_non_list_transcript_raises(self) -> None:
        with pytest.raises(ValueError):
            HookDetector({"segments": []})  # type: ignore[arg-type]

    def test_hook_to_dict(self) -> None:
        hook = Hook(
            start=1.0, end=20.0, hook_type="question", score=0.9, transcript_snippet="Why?"
        )
        d = hook.to_dict()
        assert d["start"] == 1.0 and d["end"] == 20.0 and d["duration"] == 19.0
        assert d["hook_type"] == "question" and d["score"] == 0.9 and d["snippet"] == "Why?"


class TestHooksToDirectorPlan:
    def test_hooks_become_clip_decisions(self) -> None:
        hooks = HookDetector(_hook_transcript()).find_hooks(top_n=3)
        plan = hooks_to_director_plan(hooks, source_duration_seconds=600.0)
        assert plan.provider == "hook_detection"
        assert len(plan.clips) == 3
        assert plan.source_duration_seconds == 600.0

        items = plan_to_review_items(plan)
        assert {item.kind for item in items} == {"clip"}
        for item, hook in zip(items, sorted(hooks, key=lambda h: h.start), strict=True):
            assert item.start_seconds == hook.start
            assert item.end_seconds == hook.end
            assert item.payload["extra"]["hook_type"] == hook.hook_type  # type: ignore[index]

        decisions = ai_plan_to_director_decisions(plan)
        assert all(d.kind == "clip" for d in decisions)
        assert all(d.status == "pending" for d in decisions)  # review-gated
        assert all(d.source_span is not None for d in decisions)


class TestAiClipsCLI:
    @pytest.fixture(scope="module")
    def lesson(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        from openclip.engine.synth import generate_lesson

        return generate_lesson(tmp_path_factory.mktemp("cli-clips") / "lesson.mp4").path

    @pytest.fixture
    def transcript_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "transcript.json"
        path.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "text": "What is the key question here? Why does it matter?",
                            "start": 1.0,
                            "end": 3.5,
                        },
                        {
                            "text": "This is unbelievable! A secret nobody knows!",
                            "start": 5.0,
                            "end": 7.5,
                        },
                        {
                            "text": "Make money with this free viral hack!",
                            "start": 9.0,
                            "end": 12.0,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_ai_clips_returns_top_n_with_hook_types(
        self,
        lesson: Path,
        transcript_file: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "clips.json"
        rc = main(
            [
                "ai-clips",
                "--media", str(lesson),
                "--transcript", str(transcript_file),
                "--top", "3",
                "--min-duration", "2",
                "--max-duration", "6",
                "-o", str(out),
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["top"] == 3
        clips = payload["clips"]
        assert len(clips) == 3
        assert {c["hook_type"] for c in clips} == {"question", "shock", "keyword_dense"}
        for clip in clips:
            assert 0.0 <= clip["start"] < clip["end"] <= payload["duration_s"]
            assert 2.0 <= clip["duration"] <= 6.0
            assert 0.0 < clip["score"] <= 1.0
            assert clip["snippet"]
        assert json.loads(out.read_text(encoding="utf-8"))["clips"] == clips

    def test_ai_clips_top_limits(
        self, lesson: Path, transcript_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(
            [
                "ai-clips",
                "--media", str(lesson),
                "--transcript", str(transcript_file),
                "--top", "1",
                "--min-duration", "2",
                "--max-duration", "6",
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["clips"]) == 1

    def test_ai_clips_render(
        self,
        lesson: Path,
        transcript_file: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from shutil import which

        if not which("ffmpeg"):
            pytest.skip("ffmpeg not available")
        render_dir = tmp_path / "rendered"
        rc = main(
            [
                "ai-clips",
                "--media", str(lesson),
                "--transcript", str(transcript_file),
                "--top", "2",
                "--min-duration", "2",
                "--max-duration", "6",
                "--render",
                "--render-dir", str(render_dir),
            ]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["clips"]) == 2
        rendered = [Path(p) for p in payload["rendered"]]
        assert len(rendered) == 2
        for path in rendered:
            assert path.exists() and path.stat().st_size > 0
            assert path.parent == render_dir
