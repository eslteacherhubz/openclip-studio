"""Intelligent cut planning.

Given an alignment (script ↔ timeline) and word hypotheses, propose cuts:

- ``silence``   — from ffmpeg silencedetect spans (audio-driven, no ASR needed)
- ``filler``    — hypothesis words in a filler list ("um", "uh", ...)
- ``offscript`` — speech that matches no script line (flubs, asides, retakes)

Every decision carries a human-readable reason and a confidence, so the GUI
can show exactly why a cut is proposed. Cuts are just proposals until the
user (or `--accept-all`) accepts them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openclip.engine.align import Alignment, WordHyp
from openclip.engine.ffmpeg import SilenceSpan

DEFAULT_FILLERS = {
    "um", "uh", "umm", "uhh", "erm", "hmm", "mmm", "ah", "eh", "er",
}


@dataclass
class CutOptions:
    silences: bool = True
    silence_keep_s: float = 0.15
    fillers: bool = True
    filler_words: set[str] = field(default_factory=lambda: set(DEFAULT_FILLERS))
    offscript: bool = True
    offscript_min_s: float = 0.45
    pad_s: float = 0.08
    min_kept_s: float = 0.6

    def to_dict(self) -> dict[str, object]:
        return {
            "silences": self.silences,
            "silence_keep_s": self.silence_keep_s,
            "fillers": self.fillers,
            "filler_words": sorted(self.filler_words),
            "offscript": self.offscript,
            "offscript_min_s": self.offscript_min_s,
            "pad_s": self.pad_s,
            "min_kept_s": self.min_kept_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CutOptions:
        from openclip.engine.jsonio import as_bool, as_float, as_list

        fillers = as_list(data, "filler_words", list(DEFAULT_FILLERS))
        return cls(
            silences=as_bool(data, "silences", True),
            silence_keep_s=as_float(data, "silence_keep_s", 0.15),
            fillers=as_bool(data, "fillers", True),
            filler_words={str(w) for w in fillers},
            offscript=as_bool(data, "offscript", True),
            offscript_min_s=as_float(data, "offscript_min_s", 0.45),
            pad_s=as_float(data, "pad_s", 0.08),
            min_kept_s=as_float(data, "min_kept_s", 0.6),
        )


@dataclass
class CutDecision:
    kind: str  # "silence" | "filler" | "offscript"
    start_s: float
    end_s: float
    reason: str
    confidence: float
    accepted: bool = True

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CutDecision:
        from openclip.engine.jsonio import as_bool, as_float, as_str

        return cls(
            kind=as_str(data, "kind"),
            start_s=as_float(data, "start_s"),
            end_s=as_float(data, "end_s"),
            reason=as_str(data, "reason"),
            confidence=as_float(data, "confidence", 0.0),
            accepted=as_bool(data, "accepted", True),
        )


@dataclass
class CutPlan:
    cuts: list[CutDecision] = field(default_factory=list)

    def accepted_cuts(self) -> list[CutDecision]:
        return [c for c in self.cuts if c.accepted]

    def to_dict(self) -> list[dict[str, object]]:
        return [c.to_dict() for c in self.cuts]

    @classmethod
    def from_dict(cls, data: list[dict[str, object]]) -> CutPlan:
        return cls(cuts=[CutDecision.from_dict(d) for d in data])


def plan_cuts(
    alignment: Alignment,
    hyps: list[WordHyp],
    duration_s: float,
    silences: list[SilenceSpan],
    options: CutOptions | None = None,
) -> CutPlan:
    opts = options or CutOptions()
    cuts: list[CutDecision] = []

    if opts.silences:
        for span in silences:
            s = span.start_s + opts.silence_keep_s
            e = span.end_s - opts.silence_keep_s
            if e - s >= 0.20:
                cuts.append(
                    CutDecision(
                        kind="silence",
                        start_s=max(0.0, s),
                        end_s=min(duration_s, e),
                        reason=f"Silence for {span.duration:.2f}s (keep "
                        f"{opts.silence_keep_s:.2f}s breathing room)",
                        confidence=0.95,
                    )
                )

    if opts.fillers and hyps:
        cur_start: float | None = None
        cur_end = 0.0
        for h in hyps:
            is_filler = h.norm in opts.filler_words
            if is_filler:
                if cur_start is None:
                    cur_start = max(0.0, h.start_s - opts.pad_s)
                cur_end = min(duration_s, h.end_s + opts.pad_s)
            elif cur_start is not None:
                cuts.append(
                    CutDecision(
                        kind="filler",
                        start_s=cur_start,
                        end_s=cur_end,
                        reason="Filler word run",
                        confidence=0.7,
                    )
                )
                cur_start = None
        if cur_start is not None:
            cuts.append(
                CutDecision(
                    kind="filler",
                    start_s=cur_start,
                    end_s=cur_end,
                    reason="Filler word run",
                    confidence=0.7,
                )
            )

    if opts.offscript and hyps and alignment.lines:
        matched = sorted(
            (ln.start_s, ln.end_s)
            for ln in alignment.lines
            if ln.confidence > 0.0 and ln.start_s >= 0.0
        )
        # Unmatched hypothesis runs are off-script speech.
        runs: list[list[WordHyp]] = []
        prev_matched = False
        for h in hyps:
            in_match = any(s <= h.start_s and h.end_s <= e for s, e in matched)
            if not in_match:
                if runs and not prev_matched:
                    runs[-1].append(h)
                else:
                    runs.append([h])
            prev_matched = in_match
        for run in runs:
            s = max(0.0, run[0].start_s - 2 * opts.pad_s)
            e = min(duration_s, run[-1].end_s + opts.pad_s)
            if e - s >= opts.offscript_min_s:
                words_preview = " ".join(h.word for h in run[:8])
                cuts.append(
                    CutDecision(
                        kind="offscript",
                        start_s=s,
                        end_s=e,
                        reason=f"Off-script speech: \"{words_preview}\"",
                        confidence=0.75,
                    )
                )

    merged = _merge_cuts(cuts, gap=0.05)
    merged = _absorb_tiny_kept(merged, duration_s, opts.min_kept_s)
    return CutPlan(cuts=merged)


def _merge_cuts(cuts: list[CutDecision], gap: float) -> list[CutDecision]:
    if not cuts:
        return []
    cuts = sorted(cuts, key=lambda c: c.start_s)
    out = [cuts[0]]
    for c in cuts[1:]:
        last = out[-1]
        if c.start_s <= last.end_s + gap:
            if c.end_s > last.end_s:
                last.end_s = c.end_s
            kinds = {last.kind, c.kind}
            last.kind = last.kind if len(kinds) == 1 else "offscript"
            last.reason = f"{last.reason}; {c.reason}"
            last.confidence = max(last.confidence, c.confidence)
        else:
            out.append(c)
    return out


def _absorb_tiny_kept(
    cuts: list[CutDecision], duration_s: float, min_kept_s: float
) -> list[CutDecision]:
    """Extend cuts so no kept interval is shorter than min_kept_s."""
    cuts = sorted(cuts, key=lambda c: c.start_s)
    for _ in range(8):  # fixed-point iteration, tiny lists converge fast
        kept = kept_intervals(cuts, duration_s)
        too_small = [(s, e) for s, e in kept if e - s < min_kept_s]
        if not too_small:
            break
        s, e = too_small[0]
        prev_cut = max((c for c in cuts if c.end_s <= s), key=lambda c: c.end_s, default=None)
        next_cut = min((c for c in cuts if c.start_s >= e), key=lambda c: c.start_s, default=None)
        if prev_cut is not None:
            prev_cut.end_s = e
        elif next_cut is not None:
            next_cut.start_s = s
        else:
            # Single tiny island: drop all cuts (keep whole media) rather than
            # render a strobing result.
            return []
    return cuts


def kept_intervals(cuts: list[CutDecision], duration_s: float) -> list[tuple[float, float]]:
    """Complement of accepted cuts over [0, duration]."""
    accepted = sorted(
        (c.start_s, c.end_s) for c in cuts if c.accepted and c.end_s > c.start_s
    )
    kept: list[tuple[float, float]] = []
    pos = 0.0
    for s, e in accepted:
        s = max(0.0, s)
        e = min(duration_s, e)
        if s > pos:
            kept.append((pos, s))
        pos = max(pos, e)
    if pos < duration_s:
        kept.append((pos, duration_s))
    return kept
