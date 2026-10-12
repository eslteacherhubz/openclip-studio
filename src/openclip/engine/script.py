"""Lesson script model: ordered bilingual sentence pairs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_SUPPORT_SEP = " :: "


@dataclass
class ScriptLine:
    index: int
    primary: str
    support: str = ""

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
