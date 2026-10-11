"""Typed decision model for the Production Director.

Every decision is an inspectable record: what to do, which script sentence
justifies it, why, with which parameters, and whether a human accepted it.
The EDL is deterministic and versionable JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openclip.engine.jsonio import (
    as_dict,
    as_int,
    as_list,
    as_str,
)

DECISION_KINDS = (
    "cut",  # remove a time span (from script-divergence/silence/filler analysis)
    "caption",  # bilingual animated caption for an aligned script line
    "callout",  # keyword/example overlay, layer above captions
    "section_title",  # title card at a script section boundary
    "diagram_slot",  # reviewed image overlay during an aligned line (v2)
    "color_preset",  # lighting/colour chain (v2)
    "reframe",  # presenter auto-framing (v2)
    "background",  # background replacement (v3)
    "emphasis",  # karaoke-highlight of a target term (v2)
)

STATUSES = ("pending", "accepted", "rejected")


@dataclass
class Decision:
    kind: str
    rationale: str
    id: str
    source_line: int | None = None  # script line index that justifies it
    source_span: tuple[float, float] | None = None  # timeline range
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source_line": self.source_line,
            "source_span": (
                [round(self.source_span[0], 3), round(self.source_span[1], 3)]
                if self.source_span is not None
                else None
            ),
            "params": self.params,
            "rationale": self.rationale,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Decision:
        from openclip.engine.jsonio import as_optional_int

        span = data.get("source_span")
        source_span: tuple[float, float] | None = None
        if (
            isinstance(span, list)
            and len(span) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in span)
        ):
            source_span = (float(span[0]), float(span[1]))
        kind = as_str(data, "kind")
        if kind not in DECISION_KINDS:
            raise ValueError(f"Unknown decision kind {kind!r}")
        status = as_str(data, "status", "pending")
        if status not in STATUSES:
            raise ValueError(f"Unknown decision status {status!r}")
        return cls(
            id=as_str(data, "id"),
            kind=kind,
            source_line=as_optional_int(data, "source_line"),
            source_span=source_span,
            params=dict(as_dict(data, "params")),
            rationale=as_str(data, "rationale"),
            status=status,
        )


EDL_SCHEMA_VERSION = 1


@dataclass
class EditDecisionList:
    project_source: str
    decisions: list[Decision] = field(default_factory=list)
    schema_version: int = EDL_SCHEMA_VERSION

    def ordered(self) -> list[Decision]:
        """Deterministic order: source timeline, then kind, then id."""
        return sorted(
            self.decisions,
            key=lambda d: (
                d.source_span[0] if d.source_span else float("inf"),
                d.kind,
                d.id,
            ),
        )

    def accepted(self) -> list[Decision]:
        return [d for d in self.ordered() if d.status == "accepted"]

    def has_pending(self) -> bool:
        return any(d.status == "pending" for d in self.decisions)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": EDL_SCHEMA_VERSION,
            "project_source": self.project_source,
            "decisions": [d.to_dict() for d in self.ordered()],
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EditDecisionList:
        if as_int(data, "schema", 0) != EDL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported EDL schema {data.get('schema')!r} (expected {EDL_SCHEMA_VERSION})"
            )
        decisions = [Decision.from_dict(item) for item in as_list(data, "decisions")]
        return cls(
            project_source=as_str(data, "project_source"),
            decisions=decisions,
        )

    @classmethod
    def from_json(cls, text: str) -> EditDecisionList:
        import json

        return cls.from_dict(json.loads(text))


def check_invariants(edl: EditDecisionList) -> list[str]:
    """Return a list of invariant violations (empty list = safe).

    Enforces the architecture rules from docs/PRODUCTION_DIRECTOR.md:
    unique ids, known kinds/statuses, source attribution, and — critically —
    that `cut` decisions carry a rationale traceable to analysis, not an
    opaque generative instruction.
    """
    problems: list[str] = []
    seen_ids: set[str] = set()
    for d in edl.ordered():
        if d.id in seen_ids:
            problems.append(f"duplicate decision id {d.id!r}")
        seen_ids.add(d.id)
        if d.kind not in DECISION_KINDS:
            problems.append(f"{d.id}: unknown kind {d.kind!r}")
        if d.status not in STATUSES:
            problems.append(f"{d.id}: unknown status {d.status!r}")
        if d.kind == "cut":
            if d.source_span is None:
                problems.append(f"{d.id}: cut without source_span")
            if not d.rationale.strip():
                problems.append(f"{d.id}: cut without rationale (not inspectable)")
        overlay_kinds = ("caption", "callout", "section_title", "diagram_slot", "emphasis")
        if d.kind in overlay_kinds and d.source_line is None:
            problems.append(f"{d.id}: {d.kind} not tied to a script line")
    return problems


def summarize(edl: EditDecisionList) -> str:
    counts: dict[str, int] = {}
    for d in edl.decisions:
        counts[d.kind] = counts.get(d.kind, 0) + 1
    parts = [f"{n} {kind}" for kind, n in sorted(counts.items())]
    pending = sum(1 for d in edl.decisions if d.status == "pending")
    return f"{len(edl.decisions)} decisions ({', '.join(parts)}), {pending} pending"
