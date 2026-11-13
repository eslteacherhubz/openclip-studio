"""Tests for the AI plan → Production Director bridge."""

from __future__ import annotations

import json
from pathlib import Path

from openclip.ai import (
    ai_plan_to_director_decisions,
    normalize_ai_plan,
    parse_ai_plan,
)
from openclip.director.models import STATUSES, Decision

FIXTURE = Path(__file__).parent / "fixtures" / "ai" / "mock_plan.json"


def _plan() -> object:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return normalize_ai_plan(parse_ai_plan(data))


class TestDirectorBridge:
    def test_valid_plan_does_not_raise(self) -> None:
        decisions = ai_plan_to_director_decisions(_plan())  # type: ignore[arg-type]
        assert len(decisions) == 4  # 2 cuts + 2 captions (clips/broll/zooms not mapped yet)

    def test_returns_director_decisions(self) -> None:
        decisions = ai_plan_to_director_decisions(_plan())  # type: ignore[arg-type]
        assert all(isinstance(d, Decision) for d in decisions)
        assert {d.kind for d in decisions} == {"cut", "caption"}

    def test_reasons_preserved_in_rationale(self) -> None:
        decisions = ai_plan_to_director_decisions(_plan())  # type: ignore[arg-type]
        rationales = {d.rationale for d in decisions}
        assert "filler words and pause" in rationales
        assert "off-topic tangent, not in script" in rationales

    def test_nothing_auto_accepted(self) -> None:
        decisions = ai_plan_to_director_decisions(_plan())  # type: ignore[arg-type]
        assert all(d.status == "pending" for d in decisions)
        assert all(d.status in STATUSES for d in decisions)

    def test_spans_and_ids_mapped(self) -> None:
        decisions = ai_plan_to_director_decisions(_plan())  # type: ignore[arg-type]
        cut = next(d for d in decisions if d.id == "cut-filler")
        assert cut.source_span == (12.5, 15.0)

    def test_deterministic_and_does_not_mutate_plan(self) -> None:
        from openclip.ai import AIDirectorPlan

        plan = _plan()
        assert isinstance(plan, AIDirectorPlan)
        before = plan.to_json()
        first = [d.to_dict() for d in ai_plan_to_director_decisions(plan)]
        second = [d.to_dict() for d in ai_plan_to_director_decisions(plan)]
        assert first == second
        assert plan.to_json() == before

    def test_stub_contract_clip_broll_zoom_not_mapped(self) -> None:
        """Documented stub: kinds without a v1 DECISION_KINDS equivalent are skipped."""
        decisions = ai_plan_to_director_decisions(_plan())  # type: ignore[arg-type]
        ids = {d.id for d in decisions}
        assert "clip-intro" not in ids
        assert "broll-classroom" not in ids
        assert "zoom-example" not in ids
