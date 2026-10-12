"""Deterministic rule planner: project state → EditDecisionList.

Pure function of the project (script, alignment, cut plan). No randomness,
no wall-clock, no network. Script markers drive compositions:

    Section: Present tense basics     → section_title
    Keyword: photosynthesis           → callout
    Example: "I go to school"         → callout
    Diagram: assets/tense_map.png     → diagram_slot (v2: needs overlay executor)

The planner never invents cuts: `cut` decisions are derived 1:1 from the
project's reviewed cut plan, preserving the meaning-preservation rule.
"""

from __future__ import annotations

from openclip.director.models import Decision, EditDecisionList
from openclip.engine.jsonio import as_float, as_int
from openclip.engine.project import Project
from openclip.engine.script import marker_payload, tokenize


def _line_alignment(project: Project) -> dict[int, tuple[float, float]]:
    """Aligned (start, end) per script line index from the cached alignment."""
    spans: dict[int, tuple[float, float]] = {}
    if project.alignment is None:
        return spans
    for entry in project.alignment:
        idx = as_int(entry, "index", -1)
        conf = as_float(entry, "confidence", 0.0)
        start = as_float(entry, "start_s", -1.0)
        end = as_float(entry, "end_s", -1.0)
        if conf > 0.0 and start >= 0.0 and end > start:
            spans[idx] = (start, end)
    return spans


def plan_from_project(project: Project) -> EditDecisionList:
    """Produce the EDL for a project (analysis must have run first)."""
    aligned = _line_alignment(project)
    lines = project.script.lines
    marker_flags = [ln.is_marker for ln in lines]

    def next_aligned_spoken(index: int) -> tuple[float, float] | None:
        """Span of the next aligned, non-marker line at or after `index`."""
        for j in range(index, len(lines)):
            if marker_flags[j]:
                continue
            if (span := aligned.get(lines[j].index)) is not None:
                return span
        return None

    decisions: list[Decision] = []

    if project.cut_plan is not None:
        for i, cut in enumerate(project.cut_plan.cuts):
            if not cut.accepted:
                continue
            decisions.append(
                Decision(
                    id=f"cut-{i:03d}",
                    kind="cut",
                    source_line=None,
                    source_span=(cut.start_s, cut.end_s),
                    params={"confidence": cut.confidence, "kind": cut.kind},
                    rationale=cut.reason,
                    status="accepted",
                )
            )

    for line in lines:
        is_marker = line.is_marker
        payload = marker_payload(line.primary)
        span = aligned.get(line.index)

        if not is_marker and span is not None:
            decisions.append(
                Decision(
                    id=f"caption-{line.index:03d}",
                    kind="caption",
                    source_line=line.index,
                    source_span=span,
                    params={"primary": line.primary, "support": line.support},
                    rationale=(
                        f"Bilingual caption for script line {line.index} "
                        f"({len(tokenize(line.primary))} words)"
                    ),
                    status="pending",
                )
            )

        if is_marker:
            # Markers are not spoken: they attach to the next spoken line's
            # aligned span. Unanchored markers (nothing follows) are ignored.
            target = next_aligned_spoken(line.index)
            if target is None:
                continue
            kind, value = payload if payload else ("", "")
            if kind == "section":
                decisions.append(
                    Decision(
                        id=f"section-{line.index:03d}",
                        kind="section_title",
                        source_line=line.index,
                        source_span=target,
                        params={"title": value},
                        rationale=f"Section marker on line {line.index}",
                        status="pending",
                    )
                )
            if kind == "keyword":
                term = value.strip("\"'")
                decisions.append(
                    Decision(
                        id=f"callout-{line.index:03d}",
                        kind="callout",
                        source_line=line.index,
                        source_span=target,
                        params={"term": term, "position": "lower_third"},
                        rationale=f"Keyword/example marker on line {line.index}",
                        status="pending",
                    )
                )
            if kind == "diagram":
                decisions.append(
                    Decision(
                        id=f"diagram-{line.index:03d}",
                        kind="diagram_slot",
                        source_line=line.index,
                        source_span=target,
                        params={"asset": value},
                        rationale=f"Diagram marker on line {line.index}",
                        status="pending",
                    )
                )

    return EditDecisionList(project_source=str(project.source), decisions=decisions)
