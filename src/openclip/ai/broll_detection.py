"""B-roll opportunity detection over a timestamped transcript.

Rule-based, deterministic, no LLM and no network: scans transcript blocks and
identifies windows where B-roll would enhance the visual narrative. Detection
reasons: "visual_language" (keywords: imagine, look at, picture, for example),
"concept_intro" (noun-heavy concept introductions), "topic_shift" (semantic
boundaries between segments). The Face-First rule strictly excludes any window
that overlaps with Face-First hook windows.
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

from openclip.ai.broll_types import BrollOpportunity, BrollSource, ReasonType
from openclip.ai.edl import AI_PLAN_SCHEMA_VERSION, AIBRoll, AIDirectorPlan, AISpan
from openclip.ai.transcript_editing import TranscriptSegment

logger = logging.getLogger(__name__)

VISUAL_LANGUAGE_KEYWORDS: tuple[str, ...] = (
    "imagine",
    "look at",
    "picture",
    "for example",
    "for instance",
    "visualize",
    "see how",
    "watch",
    "notice",
    "observe",
    "here we see",
    "this shows",
    "as you can see",
)

CONCEPT_INTRO_PATTERNS: tuple[str, ...] = (
    r"\b(what is|what are|define|definition of|concept of|idea of)\b",
    r"\b(the concept|the idea|the principle|the theory|the model)\b",
    r"\b(introducing|let me introduce|we introduce|this is)\b",
    r"\b(key concept|main idea|core concept|fundamental)\b",
)

TOPIC_SHIFT_MIN_DURATION = 5.0
TOPIC_SHIFT_MAX_DURATION = 30.0
VISUAL_LANGUAGE_MIN_DURATION = 3.0
VISUAL_LANGUAGE_MAX_DURATION = 20.0
CONCEPT_INTRO_MIN_DURATION = 4.0
CONCEPT_INTRO_MAX_DURATION = 25.0

# Priority bases (higher = more preferred in greedy selection)
VISUAL_LANGUAGE_BASE_PRIORITY = 0.75
CONCEPT_INTRO_BASE_PRIORITY = 0.75
TOPIC_SHIFT_BASE_PRIORITY = 0.25
TOPIC_SHIFT_MAX_PRIORITY = 0.7

_WORD_RE = re.compile(r"[a-z']+")

# Common stop words to filter out for topic shift detection
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
    "can", "this", "that", "these", "those", "it", "its", "they", "them", "their", "we",
    "our", "you", "your", "i", "me", "my", "he", "she", "his", "her", "him", "us",
    "let", "now", "then", "so", "very", "just", "also", "even", "still",
}


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _content_words(text: str) -> set[str]:
    """Return content words (non-stop-words) from text."""
    return {w for w in _words(text) if w not in _STOP_WORDS and len(w) > 2}


def _noun_ratio(text: str) -> float:
    """Heuristic: ratio of capitalized words and common noun suffixes."""
    # Check capitalization on original text, not lowercased
    original_words = re.findall(r"[A-Za-z']+", text)
    lower_words = _words(text)
    if not lower_words:
        return 0.0
    noun_indicators = 0
    for orig, low in zip(original_words, lower_words, strict=True):
        is_capitalized = orig[0].isupper()
        has_noun_suffix = low.endswith(
            ("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism", "ist", "er", "or")
        )
        if is_capitalized or has_noun_suffix:
            noun_indicators += 1
    return noun_indicators / len(lower_words)


class BrollDetector:
    """Deterministic rule-based B-roll opportunity detector over a timestamped transcript.

    ``blocks`` is a list of :class:`TranscriptSegment` (from ``transcript_editing.py``).
    Only blocks with valid timestamps are considered.

    The ``exclude_hooks`` parameter enforces the Face-First rule: any B-roll window
    that overlaps with a hook window is strictly excluded.
    """

    def __init__(self, blocks: list[TranscriptSegment]) -> None:
        if not isinstance(blocks, list):
            raise ValueError("blocks must be a list of TranscriptSegment")
        filtered = [
            b
            for b in blocks
            if b.start_s is not None and b.end_s is not None and b.start_s < b.end_s
        ]
        filtered.sort(key=lambda b: (b.start_s, b.end_s, b.text))
        self._blocks = filtered

    def find_opportunities(
        self,
        exclude_hooks: tuple[tuple[float, float], ...] = (),
    ) -> list[BrollOpportunity]:
        """Return sorted list of non-overlapping B-roll opportunities.

        Opportunities are detected for three reasons:
        - "visual_language": speaker uses visual cue words
        - "concept_intro": speaker introduces a new concept (noun-heavy)
        - "topic_shift": semantic boundary between adjacent segments

        Windows overlapping any range in ``exclude_hooks`` are discarded.
        Result is sorted by start time, non-overlapping (greedy by priority).
        """
        if not self._blocks:
            return []

        candidates: list[BrollOpportunity] = []

        # 1. Visual language detection
        candidates.extend(self._find_visual_language_opportunities(exclude_hooks))

        # 2. Concept introduction detection
        candidates.extend(self._find_concept_intro_opportunities(exclude_hooks))

        # 3. Topic shift detection
        candidates.extend(self._find_topic_shift_opportunities(exclude_hooks))

        # Greedy selection: sort by priority (desc), then start (asc), then duration (desc)
        # Pick non-overlapping
        candidates.sort(key=lambda o: (-o.priority, o.start, -o.duration))
        selected: list[BrollOpportunity] = []
        for cand in candidates:
            if any(cand.start < s.end and s.start < cand.end for s in selected):
                continue
            selected.append(cand)

        selected.sort(key=lambda o: o.start)
        return selected

    def _overlaps_exclude(
        self, start: float, end: float, exclude_hooks: tuple[tuple[float, float], ...]
    ) -> bool:
        return any(start < ex_end and ex_start < end for ex_start, ex_end in exclude_hooks)

    def _find_visual_language_opportunities(
        self, exclude_hooks: tuple[tuple[float, float], ...]
    ) -> list[BrollOpportunity]:
        opportunities: list[BrollOpportunity] = []
        for block in self._blocks:
            assert block.start_s is not None
            assert block.end_s is not None
            text_lower = block.text.lower()
            matched_keywords = tuple(kw for kw in VISUAL_LANGUAGE_KEYWORDS if kw in text_lower)
            if not matched_keywords:
                continue

            start = block.start_s
            end = block.end_s
            duration = end - start

            # Extend window slightly if too short
            if duration < VISUAL_LANGUAGE_MIN_DURATION:
                end = start + VISUAL_LANGUAGE_MIN_DURATION
            elif duration > VISUAL_LANGUAGE_MAX_DURATION:
                end = start + VISUAL_LANGUAGE_MAX_DURATION

            if self._overlaps_exclude(start, end, exclude_hooks):
                continue

            # Priority based on number of visual keywords matched
            priority = min(1.0, VISUAL_LANGUAGE_BASE_PRIORITY + 0.1 * len(matched_keywords))
            concept_query = block.text[:120]

            opportunities.append(
                BrollOpportunity(
                    start=start,
                    end=end,
                    concept_query=concept_query,
                    keywords=matched_keywords,
                    priority=priority,
                    reason="visual_language",
                )
            )
        return opportunities

    def _find_concept_intro_opportunities(
        self, exclude_hooks: tuple[tuple[float, float], ...]
    ) -> list[BrollOpportunity]:
        opportunities: list[BrollOpportunity] = []
        for block in self._blocks:
            assert block.start_s is not None
            assert block.end_s is not None
            text_lower = block.text.lower()

            # Check for concept intro patterns
            pattern_match = any(re.search(pat, text_lower) for pat in CONCEPT_INTRO_PATTERNS)
            if not pattern_match:
                continue

            # Also require decent noun ratio
            if _noun_ratio(block.text) < 0.15:
                continue

            start = block.start_s
            end = block.end_s
            duration = end - start

            if duration < CONCEPT_INTRO_MIN_DURATION:
                end = start + CONCEPT_INTRO_MIN_DURATION
            elif duration > CONCEPT_INTRO_MAX_DURATION:
                end = start + CONCEPT_INTRO_MAX_DURATION

            if self._overlaps_exclude(start, end, exclude_hooks):
                continue

            # Extract key nouns as keywords (use original text for capitalization check)
            original_words = re.findall(r"[A-Za-z']+", block.text)
            lower_words = _words(block.text)
            nouns = []
            for orig, low in zip(original_words, lower_words, strict=True):
                is_capitalized = orig[0].isupper()
                has_noun_suffix = low.endswith(
                    ("tion", "sion", "ment", "ness", "ity", "cept", "ple")
                )
                if is_capitalized or has_noun_suffix:
                    nouns.append(low)
            keywords = tuple(nouns[:5]) if nouns else tuple(lower_words[:5])

            priority = min(1.0, CONCEPT_INTRO_BASE_PRIORITY + 0.1 * len(keywords))
            concept_query = block.text[:120]

            opportunities.append(
                BrollOpportunity(
                    start=start,
                    end=end,
                    concept_query=concept_query,
                    keywords=keywords,
                    priority=priority,
                    reason="concept_intro",
                )
            )
        return opportunities

    def _find_topic_shift_opportunities(
        self, exclude_hooks: tuple[tuple[float, float], ...]
    ) -> list[BrollOpportunity]:
        opportunities: list[BrollOpportunity] = []
        if len(self._blocks) < 2:
            return opportunities

        for i in range(len(self._blocks) - 1):
            curr = self._blocks[i]
            nxt = self._blocks[i + 1]
            assert curr.start_s is not None
            assert curr.end_s is not None
            assert nxt.start_s is not None
            assert nxt.end_s is not None

            # Skip if there's a large gap (likely a natural break, not a shift)
            gap = nxt.start_s - curr.end_s
            if gap > 10.0:
                continue

            curr_words = _content_words(curr.text)
            nxt_words = _content_words(nxt.text)

            if not curr_words or not nxt_words:
                continue

            # Jaccard distance as topic shift measure
            intersection = len(curr_words & nxt_words)
            union = len(curr_words | nxt_words)
            if union == 0:
                continue
            jaccard_sim = intersection / union
            shift_score = 1.0 - jaccard_sim

            # Threshold for significant topic shift - higher to avoid
            # false positives on similar content
            if shift_score < 0.85:
                continue

            # Window around the boundary
            start = max(0.0, curr.end_s - 2.0)
            end = nxt.start_s + 3.0
            duration = end - start

            if duration < TOPIC_SHIFT_MIN_DURATION:
                end = start + TOPIC_SHIFT_MIN_DURATION
            elif duration > TOPIC_SHIFT_MAX_DURATION:
                end = start + TOPIC_SHIFT_MAX_DURATION

            if self._overlaps_exclude(start, end, exclude_hooks):
                continue

            # Keywords from both segments (content words only)
            combined_words = curr_words | nxt_words
            keywords = tuple(sorted(combined_words, key=lambda w: -len(w))[:6])

            priority = min(TOPIC_SHIFT_MAX_PRIORITY, TOPIC_SHIFT_BASE_PRIORITY + 0.55 * shift_score)
            concept_query = f"{curr.text[:60]} ... {nxt.text[:60]}"

            opportunities.append(
                BrollOpportunity(
                    start=start,
                    end=end,
                    concept_query=concept_query,
                    keywords=keywords,
                    priority=priority,
                    reason="topic_shift",
                )
            )
        return opportunities


# ---------------------------------------------------------------------------
# Bridge to AI Director plan
# ---------------------------------------------------------------------------


def broll_to_director_plan(
    opportunities: list[BrollOpportunity],
    sources: list[BrollSource],
    source_duration_seconds: float | None,
) -> AIDirectorPlan:
    """Convert detected B-roll opportunities + resolved sources into an AI Director plan.

    Each opportunity is paired with its corresponding source (by index) and becomes
    an AIBRoll entry in the plan. The plan can then flow through the existing
    review/render pipeline via `ai_plan_to_director_decisions`.
    """
    broll_items: list[AIBRoll] = []
    for index, (opp, src) in enumerate(zip(opportunities, sources, strict=True)):
        broll_items.append(
            AIBRoll(
                span=AISpan(start_seconds=opp.start, end_seconds=opp.end),
                id=f"broll-{index:03d}",
                query=opp.concept_query,
                source=src.match_type,
                mode="overlay",
                extra={
                    "asset": src.path,
                    "confidence": src.confidence,
                    "reason": opp.reason,
                    "keywords": list(opp.keywords),
                },
            )
        )
    return AIDirectorPlan(
        version=AI_PLAN_SCHEMA_VERSION,
        provider="broll_detection",
        prompt="Rule-based B-roll opportunity detection with local library sourcing",
        source_duration_seconds=source_duration_seconds,
        broll=broll_items,
        notes=[f"{len(broll_items)} B-roll opportunity(ies) detected"],
    )


# ---------------------------------------------------------------------------
# Generated card rendering (for "generated" match_type fallback)
# ---------------------------------------------------------------------------


def _render_generated_card(
    concept_query: str,
    duration: float,
    target_width: int = 1920,
    target_height: int = 1080,
) -> str:
    """Render a temporary generated B-roll card as a PNG image.

    Creates a dark gradient background with the concept query text centered.
    Returns the path to the temporary file. The caller is responsible for
    cleaning up the temp file after rendering completes.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("cv2/numpy not available; cannot render generated B-roll card")
        # Return a placeholder that will fail gracefully in FFmpeg
        return f"generated:broll/{concept_query[:40]}.png"

    # Create image with dark gradient background
    img = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Dark gradient: from dark gray to black
    for y in range(target_height):
        ratio = y / target_height
        # Interpolate between dark gray (40,40,40) and black (0,0,0)
        val = int(40 * (1.0 - ratio * 0.5))
        img[y, :] = (val, val, val)

    # Add subtle grid pattern for visual interest
    grid_color = (30, 30, 30)
    for x in range(0, target_width, 80):
        cv2.line(img, (x, 0), (x, target_height), grid_color, 1)
    for y in range(0, target_height, 80):
        cv2.line(img, (0, y), (target_width, y), grid_color, 1)

    # Render concept query text centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(target_width, target_height) / 800.0
    thickness = max(2, int(font_scale * 3))
    color = (255, 255, 255)  # white text

    # Wrap text to multiple lines if needed
    max_width = int(target_width * 0.85)
    words = concept_query.split()
    lines: list[str] = []
    current_line: list[str] = []
    for word in words:
        test_line = " ".join([*current_line, word])
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if text_width > max_width and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    if current_line:
        lines.append(" ".join(current_line))

    # Calculate total text height
    line_height = int(40 * font_scale)
    total_text_height = len(lines) * line_height
    start_y = (target_height - total_text_height) // 2 + line_height

    for i, line in enumerate(lines):
        (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = (target_width - text_width) // 2
        y = start_y + i * line_height
        # Draw text with subtle shadow for readability
        cv2.putText(
            img,
            line,
            (x + 2, y + 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            line,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    # Add "B-ROLL" label at bottom
    label = "B-ROLL PLACEHOLDER"
    label_scale = font_scale * 0.5
    label_thickness = max(1, int(label_scale * 2))
    (label_width, _), _ = cv2.getTextSize(label, font, label_scale, label_thickness)
    cv2.putText(
        img,
        label,
        ((target_width - label_width) // 2, target_height - 40),
        font,
        label_scale,
        (128, 128, 128),
        label_thickness,
        cv2.LINE_AA,
    )

    # Save to temp file
    safe_query = "".join(c if c.isalnum() else "_" for c in concept_query[:40])
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"openclip_broll_{safe_query}_{duration:.1f}s.png"
    cv2.imwrite(str(temp_path), img)
    logger.debug("Rendered generated B-roll card to %s", temp_path)
    return str(temp_path)


__all__ = [
    "VISUAL_LANGUAGE_KEYWORDS",
    "BrollDetector",
    "BrollOpportunity",
    "ReasonType",
    "_render_generated_card",
    "broll_to_director_plan",
]
