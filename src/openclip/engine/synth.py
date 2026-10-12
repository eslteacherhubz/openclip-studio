"""Synthetic bilingual lesson generator (FFmpeg lavfi) for tests and demos.

No fixtures in git: every media file used by the test suite is generated
on the fly here. The generator is deterministic so tests can assert exact
timeline ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openclip.engine.ffmpeg import run_ffmpeg


@dataclass
class SynthSegment:
    """One span of the synthetic timeline."""

    kind: str  # "speech" | "silence" | "mistake"
    start_s: float
    end_s: float
    primary: str = ""
    support: str = ""


@dataclass
class SynthLesson:
    path: Path
    duration_s: float
    segments: list[SynthSegment]
    width: int = 320
    height: int = 180
    fps: int = 15

    @property
    def sentences(self) -> list[SynthSegment]:
        return [s for s in self.segments if s.kind == "speech"]

    def script_blocks(self) -> list[dict[str, object]]:
        return [
            {"primary": s.primary, "support": s.support, "start": s.start_s, "end": s.end_s}
            for s in self.sentences
        ]

    def ground_truth_json(self) -> str:
        return json.dumps(
            {
                "path": str(self.path),
                "duration_s": self.duration_s,
                "segments": [
                    {
                        "kind": s.kind,
                        "start_s": s.start_s,
                        "end_s": s.end_s,
                        "primary": s.primary,
                        "support": s.support,
                    }
                    for s in self.segments
                ],
            },
            indent=2,
        )


# Deterministic bilingual lesson plan. Tone frequencies differ per sentence so
# downstream tools (and humans debugging) can tell segments apart. The plan
# deliberately contains a flub (off-script, 150 Hz) and a long silence, which
# the cut planner is expected to remove.
_PLAN: list[tuple[str, float, str, str, float]] = [
    # (kind, duration, primary, support, tone_hz)
    ("silence", 0.60, "", "", 0.0),
    ("speech", 1.80, "Hello everyone, welcome to the lesson.", "大家好，欢迎来到课程。", 440.0),
    ("silence", 0.80, "", "", 0.0),
    ("speech", 2.00, "Today we will learn five new words.", "今天我们学习五个新单词。", 494.0),
    ("mistake", 1.20, "", "", 150.0),
    ("speech", 2.20, "Please listen and repeat after me.", "请听录音然后跟读。", 554.0),
    ("silence", 1.60, "", "", 0.0),
    ("speech", 2.00, "Great job, see you next time.", "做得好，下次见。", 622.0),
    ("silence", 1.80, "", "", 0.0),
]


def lesson_plan() -> tuple[list[SynthSegment], float]:
    """The deterministic plan (segments, total duration) without rendering."""
    segments: list[SynthSegment] = []
    t = 0.0
    for kind, dur, primary, support, _tone in _PLAN:
        segments.append(SynthSegment(kind, t, t + dur, primary, support))
        t += dur
    return segments, t


def generate_lesson(out: str | Path) -> SynthLesson:
    """Render the deterministic synthetic lesson to `out` (an .mp4 path)."""
    out_path = Path(out)
    segments, total = lesson_plan()
    tones = [p[4] for p in _PLAN]

    args: list[str] = [
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=320x180:rate=15:duration={total:.3f}",
    ]
    filter_parts: list[str] = []
    labels: list[str] = []
    for idx, (seg, tone) in enumerate(zip(segments, tones, strict=True), start=1):
        dur = seg.end_s - seg.start_s
        if seg.kind == "silence":
            args += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:duration={dur:.3f}"]
            labels.append(f"[{idx}:a]")
        else:
            vol = "0.45" if seg.kind == "mistake" else "0.9"
            args += ["-f", "lavfi", "-i", f"sine=frequency={tone:.1f}:duration={dur:.3f}"]
            filter_parts.append(f"[{idx}:a]volume={vol}[a{idx}]")
            labels.append(f"[a{idx}]")
    filter_parts.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[aout]")

    args += [
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(out_path),
    ]
    run_ffmpeg(args, timeout=300)
    return SynthLesson(path=out_path, duration_s=total, segments=segments)


@dataclass
class WordSpan:
    word: str
    start_s: float
    end_s: float


def lesson_word_hypotheses(lesson: SynthLesson) -> list[WordSpan]:
    """Mock ASR: expand each speech sentence into evenly spaced word hypotheses.

    Mistake spans get filler pseudo-words. This provides deterministic, exact
    ground truth for aligner/cut-planner tests without any model.
    """
    hyps: list[WordSpan] = []
    for seg in lesson.segments:
        dur = seg.end_s - seg.start_s
        if seg.kind == "speech":
            words = seg.primary.split()
            per = dur / len(words)
            for i, w in enumerate(words):
                hyps.append(WordSpan(w, seg.start_s + i * per, seg.start_s + (i + 1) * per))
        elif seg.kind == "mistake":
            n = max(2, int(dur / 0.4))
            per = dur / n
            for i in range(n):
                hyps.append(WordSpan("um", seg.start_s + i * per, seg.start_s + (i + 1) * per))
    return hyps
