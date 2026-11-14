"""Transcript-based editing: diff an edited transcript against the original
timestamped transcript and produce director `Decision`s.

The user edits the transcript text (the "killer feature": edit video by
editing text). Blocks they deleted become `cut` decisions mapped to their
original media timestamps; blocks that remain (even slightly reworded) are
kept. Pure and deterministic; no network, no IO — parsing accepts decoded
JSON (dict/list) or a plain string. Edge cases (empty original, empty edited,
missing timestamps) log warnings and never crash.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from rapidfuzz import fuzz

from openclip.director.models import Decision

logger = logging.getLogger(__name__)

#: fuzzy-match threshold (0–100) above which a reworded block counts as kept
FUZZY_KEEP_THRESHOLD = 90.0


@dataclass
class TranscriptSegment:
    """One transcript block; timestamps are optional (seconds, source timeline)."""

    text: str
    start_s: float | None
    end_s: float | None


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def parse_transcript(data: object) -> list[TranscriptSegment]:
    """Parse a transcript from decoded JSON (``{"segments": [...]}`` or a list)
    or a plain string (blocks separated by blank lines; no timestamps)."""
    if isinstance(data, str):
        blocks = [b.strip() for b in data.split("\n\n") if b.strip()]
        return [TranscriptSegment(text=b, start_s=None, end_s=None) for b in blocks]
    if isinstance(data, dict):
        raw: object = data.get("segments", [])
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError(
            f"transcript must be a JSON object/list or plain text, got {type(data).__name__}"
        )
    if not isinstance(raw, list):
        raise ValueError("transcript 'segments' must be a JSON array")
    segments: list[TranscriptSegment] = []
    for item in raw:
        if isinstance(item, str):
            segments.append(TranscriptSegment(text=item, start_s=None, end_s=None))
            continue
        if not isinstance(item, dict):
            raise ValueError(
                f"transcript segment must be a string or object, got {type(item).__name__}"
            )
        text = item.get("text") or item.get("primary") or ""
        start = _optional_float(
            item.get("start", item.get("start_s", item.get("start_seconds")))
        )
        end = _optional_float(item.get("end", item.get("end_s", item.get("end_seconds"))))
        segments.append(TranscriptSegment(text=str(text), start_s=start, end_s=end))
    return segments


def transcript_to_edits(
    original_transcript: object, edited_transcript: object
) -> list[Decision]:
    """Diff `edited_transcript` against `original_transcript` → cut Decisions.

    Matching is exact on normalized text first, then fuzzy (rapidfuzz ratio ≥
    ``FUZZY_KEEP_THRESHOLD``) so slightly reworded blocks still count as kept.
    Original blocks with no match in the edited transcript were deleted by the
    user → one pending ``cut`` decision each, at the original timestamps.
    Segments without valid timestamps are skipped with a warning. Additions in
    the edited transcript never create cuts. Deterministic and side-effect
    free (logging only).
    """
    original = parse_transcript(original_transcript)
    edited = parse_transcript(edited_transcript)
    if not original:
        logger.warning("transcript edit: original transcript is empty; no edits generated")
        return []
    edited_norms = [n for n in (_normalize(s.text) for s in edited) if n]
    if not edited_norms:
        logger.warning(
            "transcript edit: edited transcript is empty — every original block becomes a cut"
        )
    remaining = Counter(edited_norms)

    decisions: list[Decision] = []
    for index, seg in enumerate(original):
        norm = _normalize(seg.text)
        if not norm:
            continue
        if remaining.get(norm, 0) > 0:
            remaining[norm] -= 1
            continue  # kept verbatim
        best_text, best_score = "", 0.0
        for candidate in edited_norms:
            if remaining.get(candidate, 0) <= 0:
                continue
            score = fuzz.ratio(norm, candidate)
            if score > best_score:
                best_text, best_score = candidate, score
        if best_score >= FUZZY_KEEP_THRESHOLD:
            remaining[best_text] -= 1
            continue  # slightly reworded → still kept
        if (
            seg.start_s is None
            or seg.end_s is None
            or not (0.0 <= seg.start_s < seg.end_s)
        ):
            logger.warning(
                "transcript edit: block %d was deleted but has no valid timestamps; "
                "no cut generated for %r",
                index,
                seg.text[:60],
            )
            continue
        decisions.append(
            Decision(
                id=f"transcript-cut-{index:03d}",
                kind="cut",
                source_span=(seg.start_s, seg.end_s),
                params={"origin": "transcript_edit", "text": seg.text[:200]},
                rationale=f"Deleted from transcript: {seg.text[:80]!r}",
                status="pending",
            )
        )
    return decisions


def decisions_to_json(decisions: list[Decision]) -> str:
    """JSON rendering of the generated EDL decisions (for CLI dry-run output)."""
    import json

    return json.dumps(
        {"decisions": [d.to_dict() for d in decisions]},
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )


__all__ = [
    "FUZZY_KEEP_THRESHOLD",
    "TranscriptSegment",
    "decisions_to_json",
    "parse_transcript",
    "transcript_to_edits",
]
