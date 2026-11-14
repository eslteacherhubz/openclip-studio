"""Tests for the AI Director EDL schema: parsing, validation, normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openclip.ai import (
    AIDirectorPlan,
    AIPlanError,
    normalize_ai_plan,
    parse_ai_plan,
    parse_ai_plan_json,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ai" / "mock_plan.json"


def _fixture_dict() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _minimal_plan() -> dict[str, object]:
    return {"version": 1, "provider": "local_mock", "prompt": "do something"}


class TestParsing:
    def test_valid_fixture_parses(self) -> None:
        plan = parse_ai_plan(_fixture_dict())
        assert plan.version == 1
        assert plan.provider == "local_mock"
        assert plan.source_duration_seconds == 90.0
        assert len(plan.clips) == 2
        assert len(plan.cuts) == 2
        assert len(plan.captions) == 2
        assert len(plan.broll) == 1
        assert len(plan.zooms) == 1
        assert plan.audio is not None and plan.audio.enabled is True
        assert plan.export is not None and plan.export.format == "mp4"
        assert len(plan.notes) == 2

    def test_minimal_plan_parses_with_empty_optionals(self) -> None:
        plan = parse_ai_plan(_minimal_plan())
        assert plan.clips == []
        assert plan.cuts == []
        assert plan.captions == []
        assert plan.broll == []
        assert plan.zooms == []
        assert plan.audio is None
        assert plan.export is None
        assert plan.notes == []

    def test_non_dict_raises(self) -> None:
        with pytest.raises(AIPlanError):
            parse_ai_plan([1, 2, 3])

    def test_missing_version_raises(self) -> None:
        data = _minimal_plan()
        del data["version"]
        with pytest.raises(AIPlanError):
            parse_ai_plan(data)

    def test_unsupported_version_raises(self) -> None:
        data = _minimal_plan()
        data["version"] = 99
        with pytest.raises(AIPlanError):
            parse_ai_plan(data)

    def test_missing_provider_raises(self) -> None:
        data = _minimal_plan()
        del data["provider"]
        with pytest.raises(AIPlanError):
            parse_ai_plan(data)

    def test_missing_prompt_raises(self) -> None:
        data = _minimal_plan()
        del data["prompt"]
        with pytest.raises(AIPlanError):
            parse_ai_plan(data)

    def test_negative_start_raises(self) -> None:
        data = _minimal_plan()
        data["cuts"] = [{"start_seconds": -1.0, "end_seconds": 2.0, "reason": "x"}]
        with pytest.raises(AIPlanError):
            parse_ai_plan(data)

    def test_end_before_start_raises(self) -> None:
        data = _minimal_plan()
        data["cuts"] = [{"start_seconds": 5.0, "end_seconds": 2.0, "reason": "x"}]
        with pytest.raises(AIPlanError):
            parse_ai_plan(data)

    def test_equal_start_and_end_raises(self) -> None:
        data = _minimal_plan()
        data["cuts"] = [{"start_seconds": 3.0, "end_seconds": 3.0, "reason": "x"}]
        with pytest.raises(AIPlanError):
            parse_ai_plan(data)

    def test_non_finite_time_raises(self) -> None:
        data = _minimal_plan()
        data["cuts"] = [{"start_seconds": 0.0, "end_seconds": float("inf"), "reason": "x"}]
        with pytest.raises(AIPlanError):
            parse_ai_plan(data)

    def test_invalid_score_raises(self) -> None:
        data = _minimal_plan()
        data["clips"] = [{"start_seconds": 0.0, "end_seconds": 1.0, "score": 1.5}]
        with pytest.raises(AIPlanError):
            parse_ai_plan(data)

    def test_invalid_confidence_raises(self) -> None:
        data = _minimal_plan()
        data["cuts"] = [{"start_seconds": 0.0, "end_seconds": 1.0, "confidence": -0.1}]
        with pytest.raises(AIPlanError):
            parse_ai_plan(data)

    def test_invalid_strength_raises(self) -> None:
        data = _minimal_plan()
        data["zooms"] = [{"start_seconds": 0.0, "end_seconds": 1.0, "strength": 2.0}]
        with pytest.raises(AIPlanError):
            parse_ai_plan(data)

    def test_unknown_fields_preserved_in_extra(self) -> None:
        data = _minimal_plan()
        data["mood"] = "energetic"
        data["clips"] = [
            {"start_seconds": 0.0, "end_seconds": 1.0, "title": "t", "custom_field": 42}
        ]
        plan = parse_ai_plan(data)
        assert plan.extra["mood"] == "energetic"
        assert plan.clips[0].extra["custom_field"] == 42

    def test_flat_and_nested_span_both_accepted(self) -> None:
        data = _minimal_plan()
        data["cuts"] = [
            {"start_seconds": 1.0, "end_seconds": 2.0, "reason": "flat"},
            {"span": {"start_seconds": 3.0, "end_seconds": 4.0}, "reason": "nested"},
        ]
        plan = parse_ai_plan(data)
        assert plan.cuts[0].span.start_seconds == 1.0
        assert plan.cuts[1].span.end_seconds == 4.0

    def test_parse_ai_plan_json_round_trip(self) -> None:
        plan = parse_ai_plan(_fixture_dict())
        text = plan.to_json()
        reparsed = parse_ai_plan_json(text)
        assert reparsed.to_dict() == plan.to_dict()

    def test_parse_ai_plan_json_invalid_json(self) -> None:
        with pytest.raises(AIPlanError):
            parse_ai_plan_json("{nope")


class TestNormalization:
    def _plan(self) -> AIDirectorPlan:
        return parse_ai_plan(_fixture_dict())

    def test_items_sorted_by_start_time(self) -> None:
        data = _minimal_plan()
        data["cuts"] = [
            {"id": "b", "start_seconds": 9.0, "end_seconds": 10.0, "reason": "late"},
            {"id": "a", "start_seconds": 1.0, "end_seconds": 2.0, "reason": "early"},
        ]
        plan = normalize_ai_plan(parse_ai_plan(data))
        assert [c.id for c in plan.cuts] == ["a", "b"]
        assert [c.span.start_seconds for c in plan.cuts] == [1.0, 9.0]

    def test_missing_ids_generated_deterministically(self) -> None:
        data = _minimal_plan()
        data["clips"] = [
            {"start_seconds": 5.0, "end_seconds": 6.0, "title": "second"},
            {"start_seconds": 1.0, "end_seconds": 2.0, "title": "first"},
        ]
        first = normalize_ai_plan(parse_ai_plan(data))
        second = normalize_ai_plan(parse_ai_plan(data))
        assert [c.id for c in first.clips] == ["clip-000", "clip-001"]
        assert first.to_dict() == second.to_dict()
        # sorted: earliest span gets clip-000
        assert first.clips[0].title == "first"

    def test_exact_duplicate_spans_removed(self) -> None:
        data = _minimal_plan()
        data["cuts"] = [
            {"start_seconds": 1.0, "end_seconds": 2.0, "reason": "dup"},
            {"start_seconds": 1.0, "end_seconds": 2.0, "reason": "dup again"},
            {"start_seconds": 1.0, "end_seconds": 3.0, "reason": "overlap kept"},
        ]
        plan = normalize_ai_plan(parse_ai_plan(data))
        assert len(plan.cuts) == 2
        assert plan.cuts[0].reason == "dup"

    def test_overlapping_but_not_identical_spans_kept(self) -> None:
        data = _minimal_plan()
        data["zooms"] = [
            {"start_seconds": 1.0, "end_seconds": 4.0},
            {"start_seconds": 2.0, "end_seconds": 5.0},
        ]
        plan = normalize_ai_plan(parse_ai_plan(data))
        assert len(plan.zooms) == 2

    def test_duration_clamps_and_drops(self) -> None:
        data = _minimal_plan()
        data["source_duration_seconds"] = 10.0
        data["clips"] = [
            {"id": "partial", "start_seconds": 8.0, "end_seconds": 20.0, "title": "clamped"},
            {"id": "beyond", "start_seconds": 12.0, "end_seconds": 15.0, "title": "dropped"},
            {"id": "inside", "start_seconds": 1.0, "end_seconds": 2.0, "title": "kept"},
        ]
        plan = normalize_ai_plan(parse_ai_plan(data))
        assert [c.id for c in plan.clips] == ["inside", "partial"]
        partial = plan.clips[1]
        assert partial.span.end_seconds == 10.0

    def test_normalization_is_pure(self) -> None:
        plan = self._plan()
        before = plan.to_json()
        normalize_ai_plan(plan)
        assert plan.to_json() == before

    def test_normalized_fixture_is_deterministic(self) -> None:
        a = normalize_ai_plan(self._plan())
        b = normalize_ai_plan(self._plan())
        assert a.to_json() == b.to_json()
