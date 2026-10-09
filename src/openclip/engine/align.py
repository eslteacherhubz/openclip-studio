"""Script↔timeline alignment.

Two sources of word hypotheses ("transcribers"):

- ``MockScheduleTranscriber`` / direct hypothesis lists — deterministic,
  used by tests and by projects that already carry word timings.
- ``WhisperTranscriber`` — optional local ASR via faster-whisper
  (extra ``asr``); downloads a model on first use, then fully offline.

The aligner itself is fuzzy and monotonic: script lines are matched, in
order, to contiguous windows of hypotheses using token similarity
(rapidfuzz). Lines with no good match stay unaligned — those spans become
cut candidates.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from openclip.engine.script import LessonScript, tokenize
from openclip.engine.synth import WordSpan

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _norm(word: str) -> str:
    return word.lower().translate(_PUNCT_TABLE).strip()


@dataclass
class WordHyp:
    word: str
    start_s: float
    end_s: float

    @property
    def norm(self) -> str:
        return _norm(self.word)


class Transcriber(Protocol):
    """Anything that can turn media into word hypotheses."""

    name: str

    def transcribe(self, media: Path, language: str | None) -> list[WordHyp]: ...


@dataclass
class MockScheduleTranscriber:
    """Returns pre-computed hypotheses (tests, cached timings)."""

    hyps: list[WordHyp]
    name: str = "mock"

    def transcribe(self, media: Path, language: str | None) -> list[WordHyp]:
        return self.hyps


def spans_to_hyps(spans: list[WordSpan]) -> list[WordHyp]:
    return [WordHyp(s.word, s.start_s, s.end_s) for s in spans]


@dataclass
class WhisperTranscriber:
    """Local ASR via faster-whisper. Requires the ``asr`` extra."""

    model_size: str = "small"
    device: str = "auto"
    compute_type: str = "default"
    name: str = "whisper"

    def transcribe(self, media: Path, language: str | None) -> list[WordHyp]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "The 'asr' extra is not installed. Run: pip install eterna-openclip-studio[asr]"
            ) from exc
        model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        segments, _info = model.transcribe(
            str(media),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )
        hyps: list[WordHyp] = []
        for seg in segments:
            for w in seg.words or []:
                token = w.word.strip()
                if not token:
                    continue
                hyps.append(WordHyp(token, float(w.start), float(w.end)))
        return hyps


@dataclass
class LineAlignment:
    index: int
    start_s: float
    end_s: float
    confidence: float
    matched_words: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "confidence": round(self.confidence, 3),
            "matched_words": self.matched_words,
        }


@dataclass
class Alignment:
    lines: list[LineAlignment] = field(default_factory=list)
    transcriber: str = "even"

    @property
    def matched_indices(self) -> set[int]:
        return {ln.index for ln in self.lines if ln.confidence > 0.0}

    def to_dict(self) -> list[dict[str, object]]:
        return [ln.to_dict() for ln in self.lines]


def align_script(
    script: LessonScript,
    hyps: list[WordHyp],
    min_score: float = 55.0,
) -> Alignment:
    """Monotonic fuzzy alignment of script lines against word hypotheses."""
    from rapidfuzz import fuzz

    words = [h.norm for h in hyps]
    n = len(hyps)
    pos = 0
    lines: list[LineAlignment] = []

    for ln in script.lines:
        toks = ln.tokens()
        if not toks:
            continue
        target = " ".join(toks)
        best_score = -1.0
        best_span: tuple[int, int] | None = None
        min_w = max(1, int(len(toks) * 0.5))
        max_w = int(len(toks) * 2.0) + 4
        for start in range(pos, n):
            if words[start] == "" and start + 1 < n:
                continue
            for width in range(min_w, max_w + 1):
                end = start + width
                if end > n:
                    break
                window = " ".join(words[start:end])
                # token_sort_ratio (not token_set_ratio): length-sensitive, so
                # windows that merely contain the target do not score 100.
                score = fuzz.token_sort_ratio(target, window)
                if score > best_score:
                    best_score = score
                    best_span = (start, end)
                if score >= 97.0:
                    break
            if best_score >= 97.0:
                break

        if best_span is not None and best_score >= min_score:
            s, e = best_span
            lines.append(
                LineAlignment(
                    index=ln.index,
                    start_s=hyps[s].start_s,
                    end_s=hyps[e - 1].end_s,
                    confidence=best_score / 100.0,
                    matched_words=[h.word for h in hyps[s:e]],
                )
            )
            pos = e
        else:
            lines.append(
                LineAlignment(index=ln.index, start_s=-1.0, end_s=-1.0, confidence=0.0)
            )
    return Alignment(lines=lines, transcriber="fuzzy")


def even_alignment(script: LessonScript, duration_s: float) -> Alignment:
    """Fallback: distribute lines evenly by token count over the media.

    Used when no ASR is available. Times are estimates; silence-based cuts
    are still accurate (they come from the audio, not the script).
    """
    weights = [max(1, len(ln.tokens())) for ln in script.lines]
    total_w = sum(weights) or 1
    lines: list[LineAlignment] = []
    t = 0.0
    for ln, w in zip(script.lines, weights, strict=False):
        dur = duration_s * (w / total_w)
        lines.append(
            LineAlignment(
                index=ln.index,
                start_s=t,
                end_s=t + dur,
                confidence=0.0,
                matched_words=[],
            )
        )
        t += dur
    return Alignment(lines=lines, transcriber="even")


def alignment_from_dict(data: list[dict[str, object]]) -> Alignment:
    from openclip.engine.jsonio import as_float, as_int, as_list, as_str

    lines = [
        LineAlignment(
            index=as_int(d, "index"),
            start_s=as_float(d, "start_s"),
            end_s=as_float(d, "end_s"),
            confidence=as_float(d, "confidence", 0.0),
            matched_words=[str(w) for w in as_list(d, "matched_words")],
        )
        for d in data
    ]
    transcriber = as_str(data[0], "transcriber", "fuzzy") if data else "even"
    return Alignment(lines=lines, transcriber=transcriber)


def tokenize_line(text: str) -> list[str]:
    return tokenize(text)
