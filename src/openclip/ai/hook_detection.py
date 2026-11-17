"""Hook / viral-moment detection over a timestamped transcript.

Rule-based, deterministic, no LLM and no network: slides 15–90s windows over
the transcript blocks and scores each window for five hook types — questions,
shocking statements, emotional peaks, pattern interrupts, and viral-keyword
density — then returns the top-N non-overlapping windows as :class:`Hook`s.

The hooks convert into AI Director ``clip`` proposals via
:func:`hooks_to_director_plan`, so they flow through the existing
review/accept workflow and the multi-clip export (``clip`` decisions keep only
those spans).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from openclip.ai.edl import AI_PLAN_SCHEMA_VERSION, AIClip, AIDirectorPlan, AISpan

HookType = Literal["question", "shock", "emotional", "pattern_interrupt", "keyword_dense"]

HOOK_TYPES: tuple[HookType, ...] = (
    "question",
    "shock",
    "emotional",
    "pattern_interrupt",
    "keyword_dense",
)

QUESTION_WORDS: tuple[str, ...] = (
    "what", "why", "how", "when", "where", "who", "which",
    "can", "could", "would", "should", "is", "are", "was", "do", "does", "did",
)
SHOCK_WORDS: tuple[str, ...] = (
    "shocking", "unbelievable", "insane", "crazy", "secret", "never", "always",
    "nobody", "everyone", "mind-blowing", "mindblowing", "wild", "unreal",
    "jaw-dropping", "jaw-dropping", "impossible", "incredible",
)
EMOTION_POSITIVE: tuple[str, ...] = (
    "amazing", "incredible", "love", "loved", "best", "awesome", "fantastic",
    "brilliant", "perfect", "excited", "wow", "beautiful", "exciting",
)
EMOTION_NEGATIVE: tuple[str, ...] = (
    "hate", "hated", "worst", "terrible", "awful", "angry", "disaster",
    "horrible", "shame", "fail", "failed", "worried", "scared", "frustrating",
)
INTERRUPT_PHRASES: tuple[str, ...] = (
    "here's the thing", "here is the thing", "listen", "watch this",
    "plot twist", "but wait", "suddenly", "the secret is", "the trick is",
    "pay attention", "look at this", "the key is", "guess what",
)
VIRAL_KEYWORDS: tuple[str, ...] = (
    "money", "love", "success", "secret", "hack", "viral", "million", "rich",
    "free", "mistake", "truth", "power", "habit", "tiktok", "reels", "shorts",
    "algorithm", "growth", "clients", "income", "passive", "warning",
    "nobody tells you", "game changer", "life hack", "pro tip",
)

#: windows scoring below this are not hooks at all
MIN_HOOK_SCORE = 0.2
#: seconds of surrounding context used for topic-shift detection
TOPIC_SHIFT_CONTEXT_S = 10.0

_WORD_RE = re.compile(r"[a-z']+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True)
class Hook:
    """One proposed viral moment on the source timeline."""

    start: float
    end: float
    hook_type: HookType
    score: float  # 0..1
    transcript_snippet: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, object]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "hook_type": self.hook_type,
            "score": round(self.score, 4),
            "snippet": self.transcript_snippet,
        }


@dataclass(frozen=True)
class _Segment:
    text: str
    start: float
    end: float
    speaker: str | None


def _parse_segments(transcript: list[dict[str, object]]) -> list[_Segment]:
    segments: list[_Segment] = []
    for item in transcript:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("primary") or ""
        start = _optional_float(
            item.get("start", item.get("start_s", item.get("start_seconds")))
        )
        end = _optional_float(item.get("end", item.get("end_s", item.get("end_seconds"))))
        if not isinstance(text, str) or not text.strip():
            continue
        if start is None or end is None or not 0.0 <= start < end:
            continue  # untimed blocks cannot be placed on the timeline
        speaker_raw = item.get("speaker")
        segments.append(
            _Segment(
                text=text.strip(),
                start=start,
                end=end,
                speaker=str(speaker_raw) if isinstance(speaker_raw, str) else None,
            )
        )
    segments.sort(key=lambda s: (s.start, s.end, s.text))
    return segments


def _is_question(segment_text: str) -> bool:
    text = segment_text.strip().lower()
    if text.endswith("?"):
        return True
    first = _words(text)[:1]
    return bool(first) and first[0] in QUESTION_WORDS


def _score_question(window: list[_Segment]) -> float:
    questions = sum(1 for s in window if _is_question(s.text))
    if questions == 0:
        return 0.0
    ratio = questions / len(window)
    return min(1.0, 0.6 + 0.4 * ratio)


def _score_shock(text: str, words: list[str]) -> float:
    hits = sum(1 for w in words if w in SHOCK_WORDS)
    bangs = text.count("!")
    if hits == 0 and bangs < 2:
        return 0.0
    return min(1.0, 0.4 * hits + 0.2 * min(bangs, 3))


def _score_emotional(words: list[str]) -> float:
    lexicon = set(EMOTION_POSITIVE) | set(EMOTION_NEGATIVE)
    hits = sum(1 for w in words if w in lexicon)
    if hits == 0:
        return 0.0
    return min(1.0, hits / 3.0)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _score_pattern_interrupt(
    window: list[_Segment],
    all_segments: list[_Segment],
    window_start: float,
    window_end: float,
    text: str,
) -> float:
    lower = text.lower()
    phrase = any(p in lower for p in INTERRUPT_PHRASES)
    neighbors = [
        s
        for s in all_segments
        if (s.end <= window_start + 1e-6 and s.end >= window_start - TOPIC_SHIFT_CONTEXT_S)
        or (s.start >= window_end - 1e-6 and s.start <= window_end + TOPIC_SHIFT_CONTEXT_S)
    ]
    window_words = {w for s in window for w in _words(s.text)}
    if neighbors:
        neighbor_words = {w for s in neighbors for w in _words(s.text)}
        shift = 1.0 - _jaccard(window_words, neighbor_words)
    else:
        shift = 0.0
    if not phrase and shift < 0.3:
        return 0.0
    return min(1.0, 0.6 * (1.0 if phrase else 0.0) + 0.4 * shift)


def _score_keyword_dense(words: list[str]) -> float:
    hits = sum(1 for w in words if w in VIRAL_KEYWORDS)
    if hits == 0 or not words:
        return 0.0
    density = hits / len(words)
    return min(1.0, 0.25 * hits + 5.0 * density)


class HookDetector:
    """Deterministic rule-based hook detector over a timestamped transcript.

    ``transcript`` is a list of dicts with ``text``, ``start``, ``end``
    (seconds; ``start_s``/``start_seconds`` aliases accepted) and optional
    ``speaker``. Blocks without valid timestamps are ignored.
    """

    def __init__(self, transcript: list[dict[str, object]]) -> None:
        if not isinstance(transcript, list):
            raise ValueError("transcript must be a list of segment dicts")
        self._segments = _parse_segments(transcript)

    @property
    def segments(self) -> list[_Segment]:
        return list(self._segments)

    def find_hooks(
        self,
        min_duration: float = 15.0,
        max_duration: float = 90.0,
        top_n: int = 5,
    ) -> list[Hook]:
        """Return up to ``top_n`` non-overlapping hooks, ordered by start time.

        Slides windows of [min_duration, max_duration] over the transcript,
        scores each for the five hook types, keeps the best-scoring
        non-overlapping windows. Deterministic: same input → same output.
        """
        if min_duration <= 0.0:
            raise ValueError(f"min_duration must be > 0, got {min_duration}")
        if max_duration < min_duration:
            raise ValueError(
                f"max_duration ({max_duration}) must be >= min_duration ({min_duration})"
            )
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        segs = self._segments
        candidates: list[Hook] = []
        for i in range(len(segs)):
            window: list[_Segment] = []
            for j in range(i, len(segs)):
                window.append(segs[j])
                span = segs[j].end - segs[i].start
                if span > max_duration:
                    break
                if span < min_duration:
                    continue
                hook = self._score_window(window, segs)
                if hook is not None:
                    candidates.append(hook)

        # Greedy by score (ties: earlier start, then longer window) — non-overlapping.
        candidates.sort(key=lambda h: (-h.score, h.start, -h.end))
        selected: list[Hook] = []
        for cand in candidates:
            if len(selected) >= top_n:
                break
            if any(cand.start < s.end and s.start < cand.end for s in selected):
                continue
            selected.append(cand)
        selected.sort(key=lambda h: h.start)
        return selected

    def _score_window(self, window: list[_Segment], all_segments: list[_Segment]) -> Hook | None:
        text = " ".join(s.text for s in window)
        words = _words(text)
        scores: dict[HookType, float] = {
            "question": _score_question(window),
            "shock": _score_shock(text, words),
            "emotional": _score_emotional(words),
            "pattern_interrupt": _score_pattern_interrupt(
                window, all_segments, window[0].start, window[-1].end, text
            ),
            "keyword_dense": _score_keyword_dense(words),
        }
        best_type: HookType = "question"
        best_score = 0.0
        for hook_type in HOOK_TYPES:  # fixed order → deterministic tie-break
            if scores[hook_type] > best_score:
                best_type = hook_type
                best_score = scores[hook_type]
        if best_score < MIN_HOOK_SCORE:
            return None
        snippet = text if len(text) <= 200 else text[:197] + "..."
        return Hook(
            start=window[0].start,
            end=window[-1].end,
            hook_type=best_type,
            score=round(best_score, 4),
            transcript_snippet=snippet,
        )


def hooks_to_director_plan(
    hooks: list[Hook],
    source_duration_seconds: float | None = None,
) -> AIDirectorPlan:
    """Convert detected hooks into an AI Director plan of ``clip`` proposals.

    The clips enter the standard review workflow: accept them in the review
    table and the multi-clip export keeps only the accepted spans.
    """
    clips = [
        AIClip(
            span=AISpan(start_seconds=hook.start, end_seconds=hook.end),
            id=f"hook-{index:03d}",
            title=f"{hook.hook_type} hook",
            reason=hook.transcript_snippet,
            score=hook.score,
            extra={
                "hook_type": hook.hook_type,
                "score": hook.score,
                "snippet": hook.transcript_snippet,
            },
        )
        for index, hook in enumerate(hooks)
    ]
    return AIDirectorPlan(
        version=AI_PLAN_SCHEMA_VERSION,
        provider="hook_detection",
        prompt="Find viral moments (rule-based hook detection)",
        source_duration_seconds=source_duration_seconds,
        clips=clips,
        notes=[f"{len(clips)} hook(s) detected"],
    )


__all__ = [
    "HOOK_TYPES",
    "Hook",
    "HookDetector",
    "HookType",
    "hooks_to_director_plan",
]
