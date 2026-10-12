"""Lesson script model: ordered bilingual sentence pairs.

Scripts may contain *marker* lines (``Section:``, ``Keyword:``, ``Example:``,
``Diagram:``) that direct automated compositions but are never spoken.
Markers are detected here — one shared definition — so the aligner can skip
them (they must not consume ASR words) and the Production Director can
attach them to the next spoken line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_SUPPORT_SEP = " :: "

_MARKER_RES = (
    re.compile(r"^\s*(?:Section|#)\s*[:：]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\s*(?:Keyword|Example)\s*[:：]\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\s*Diagram\s*[:：]\s*(\S+)$", re.IGNORECASE),
)


def is_marker_text(primary: str) -> bool:
    """True when a script line is a composition marker, not spoken content."""
    return any(rx.match(primary) for rx in _MARKER_RES)


def marker_payload(primary: str) -> tuple[str, str] | None:
    """(marker_kind, payload) for a marker line, else None."""
    for kind, rx in zip(("section", "keyword", "diagram"), _MARKER_RES, strict=True):
        if (m := rx.match(primary)):
            return kind, m.group(1).strip()
    return None


@dataclass
class ScriptLine:
    index: int
    primary: str
    support: str = ""

    @property
    def is_marker(self) -> bool:
        return is_marker_text(self.primary)

    def tokens(self) -> list[str]:
        return tokenize(self.primary)


def tokenize(text: str) -> list[str]:
    """Lowercase, punctuation-stripped word tokens (CJK chars kept whole)."""
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return [t for t in _WS_RE.split(cleaned) if t]


@dataclass
class LessonScript:
    lines: list[ScriptLine] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.lines)

    def to_dict(self) -> list[dict[str, object]]:
        return [
            {"index": ln.index, "primary": ln.primary, "support": ln.support}
            for ln in self.lines
        ]

    @classmethod
    def from_dict(cls, data: list[dict[str, object]]) -> LessonScript:
        from openclip.engine.jsonio import as_int, as_str

        lines = []
        for i, item in enumerate(data):
            lines.append(
                ScriptLine(
                    index=as_int(item, "index", i),
                    primary=as_str(item, "primary"),
                    support=as_str(item, "support"),
                )
            )
        return cls(lines=lines)


def parse_script(text: str) -> LessonScript:
    """Parse a teaching script.

    Blocks are separated by blank lines. Each block is either:

    - one line ``primary text :: support text`` (bilingual pair), or
    - two lines: primary on the first, support on the second.

    A single-line block with no separator is a primary-only line.
    """
    lines: list[ScriptLine] = []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    for _i, block in enumerate(blocks):
        rows = [r.strip() for r in block.splitlines() if r.strip()]
        primary = rows[0]
        support = ""
        if len(rows) == 2:
            support = rows[1]
        elif _SUPPORT_SEP in primary:
            primary, _, support = (p.strip() for p in primary.partition(_SUPPORT_SEP))
        lines.append(ScriptLine(index=len(lines), primary=primary, support=support))
    assert len(lines) == len(blocks)
    return LessonScript(lines=lines)


def script_from_lesson_blocks(blocks: list[dict[str, object]]) -> LessonScript:
    """Build a script from synth/plan ``{"primary","support"}`` blocks."""
    return LessonScript.from_dict(
        [{"primary": b.get("primary", ""), "support": b.get("support", "")} for b in blocks]
    )
