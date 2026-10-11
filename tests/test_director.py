"""Production Director tests: determinism, invariants, review gating, render."""

from __future__ import annotations

from pathlib import Path

import pytest

from openclip.director.executor import compose_ass, render_edl
from openclip.director.models import Decision, EditDecisionList, check_invariants, summarize
from openclip.director.planner import plan_from_project
from openclip.engine.ffmpeg import probe
from openclip.engine.pipeline import run_pipeline, spans_to_hyps
from openclip.engine.project import Project
from openclip.engine.render import RenderOptions
from openclip.engine.synth import (
    SynthLesson,
    generate_lesson,
    lesson_plan,
    lesson_word_hypotheses,
)


@pytest.fixture(scope="module")
def lesson(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_lesson(tmp_path_factory.mktemp("director") / "lesson.mp4").path


def _project_with_markers(lesson_path: Path) -> Project:
    script_text = (
        "Section: Greetings :: Section: 问候\n"
        "\n"
        "Hello everyone, welcome to the lesson. :: 大家好，欢迎来到课程。\n"
        "\n"
        "Keyword: five words :: 关键词：五个单词\n"
        "\n"
        "Today we will learn five new words. :: 今天我们学习五个新单词。\n"
        "\n"
        "Diagram: assets/tense_map.png\n"
        "\n"
        "Please listen and repeat after me. :: 请听录音然后跟读。\n"
        "\n"
        "Example: I go to school :: 例句：I go to school\n"
        "\n"
        "Great job, see you next time. :: 做得好，下次见。\n"
    )
    project = Project.create(source=lesson_path, script_text=script_text)
    segments, _total = lesson_plan()
    # The synth lesson has 4 sentences; our script has 8 blocks of which 4
    # match the spoken lines. Alignment will match what it can.
    synth = SynthLesson(path=lesson_path, duration_s=14.0, segments=segments)
    hyps = spans_to_hyps(lesson_word_hypotheses(synth))
    run_pipeline(project, transcriber="mock", hyps=hyps)
    return project


class TestPlanner:
    def test_deterministic_byte_identical_edl(self, lesson: Path) -> None:
        project = _project_with_markers(lesson)
        first = plan_from_project(project).to_json()
        second = plan_from_project(project).to_json()
        assert first == second

    def test_markers_become_typed_decisions(self, lesson: Path) -> None:
        project = _project_with_markers(lesson)
        edl = plan_from_project(project)
        kinds = {d.id.split("-")[0] for d in edl.decisions}
        assert {"caption", "cut"} <= kinds
        by_id = {d.id: d for d in edl.decisions}
        # Section marker on line 0 is aligned (it is the spoken sentence 1).
        assert by_id["section-000"].kind == "section_title"
        assert by_id["section-000"].params["title"] == "Greetings"
        assert by_id["callout-002"].kind == "callout"
        assert by_id["callout-002"].params["term"] == "five words"
        assert by_id["diagram-004"].params["asset"] == "assets/tense_map.png"
        assert by_id["callout-006"].params["term"] == "I go to school"

    def test_every_decision_has_rationale_and_attribution(self, lesson: Path) -> None:
        project = _project_with_markers(lesson)
        edl = plan_from_project(project)
        assert check_invariants(edl) == []
        for d in edl.decisions:
            assert d.rationale, d.id
            if d.kind == "cut":
                assert d.source_span is not None
            else:
                assert d.source_line is not None

    def test_cuts_derived_only_from_accepted_plan(self, lesson: Path) -> None:
        project = _project_with_markers(lesson)
        assert project.cut_plan is not None
        # Reject one cut, re-plan: the rejected span must not appear.
        project.cut_plan.cuts[0].accepted = False
        edl = plan_from_project(project)
        cut_spans = {
            d.source_span for d in edl.decisions if d.kind == "cut"
        }
        assert (project.cut_plan.cuts[0].start_s, project.cut_plan.cuts[0].end_s) not in cut_spans
        for cut in project.cut_plan.cuts[1:]:
            if cut.accepted:
                assert (cut.start_s, cut.end_s) in cut_spans

    def test_roundtrip_preserves_decisions(self, lesson: Path) -> None:
        project = _project_with_markers(lesson)
        edl = plan_from_project(project)
        restored = EditDecisionList.from_json(edl.to_json())
        assert restored.to_json() == edl.to_json()

    def test_summarize_counts(self, lesson: Path) -> None:
        project = _project_with_markers(lesson)
        text = summarize(plan_from_project(project))
        assert "decisions" in text and "pending" in text


class TestInvariants:
    def test_duplicate_ids_flagged(self) -> None:
        d = Decision(
            id="x", kind="caption", source_line=0, source_span=(0.0, 1.0), rationale="r"
        )
        problems = check_invariants(EditDecisionList("p", [d, d]))
        assert any("duplicate" in p for p in problems)

    def test_cut_without_rationale_flagged(self) -> None:
        d = Decision(id="c", kind="cut", source_span=(1.0, 2.0), rationale="")
        assert any("rationale" in p for p in check_invariants(EditDecisionList("p", [d])))

    def test_overlay_without_script_line_flagged(self) -> None:
        d = Decision(id="k", kind="callout", rationale="r")
        assert any("script line" in p for p in check_invariants(EditDecisionList("p", [d])))

    def test_unknown_kind_rejected_at_parse(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            Decision.from_dict(
                {"id": "z", "kind": "vibes", "rationale": "r", "status": "pending"}
            )


class TestReviewGating:
    def test_render_blocks_on_pending(self, lesson: Path, tmp_path: Path) -> None:
        project = _project_with_markers(lesson)
        edl = plan_from_project(project)
        assert edl.has_pending()
        media = probe(lesson)
        with pytest.raises(ValueError, match="pending"):
            render_edl(project, media, edl, tmp_path / "out.mp4")

    def test_unsupported_kind_fails_explicitly(self, lesson: Path, tmp_path: Path) -> None:
        project = _project_with_markers(lesson)
        edl = plan_from_project(project)
        for d in edl.decisions:
            d.status = "accepted"
        edl.decisions.append(
            Decision(
                id="reframe-999",
                kind="reframe",
                source_span=(0.0, 2.0),
                rationale="test",
                status="accepted",
            )
        )
        media = probe(lesson)
        with pytest.raises(NotImplementedError, match="reframe"):
            render_edl(project, media, edl, tmp_path / "out.mp4")


class TestExecutor:
    def test_render_edl_end_to_end(self, lesson: Path, tmp_path: Path) -> None:
        project = _project_with_markers(lesson)
        edl = plan_from_project(project)
        for d in edl.decisions:
            # diagram_slot has no executor primitive yet (v2) — the teacher
            # leaves it rejected; everything else ships.
            d.status = "rejected" if d.kind == "diagram_slot" else "accepted"
        media = probe(lesson)
        out = tmp_path / "directed.mp4"
        result = render_edl(
            project, media, edl, out, RenderOptions(burn_captions=True), require_review=False
        )
        assert result.captions_burned
        exported = probe(out)
        assert exported.vcodec == "h264" and exported.acodec == "aac"
        assert abs(exported.duration_s - result.kept_duration_s) < 0.5

    def test_compose_ass_includes_callout_and_title(self, lesson: Path) -> None:
        project = _project_with_markers(lesson)
        edl = plan_from_project(project)
        for d in edl.decisions:
            d.status = "accepted"
        media = probe(lesson)
        text = compose_ass(project, media, [], edl)
        assert text is not None
        assert "Style: Callout" in text
        assert "Greetings" in text  # section title
        assert "five words" in text  # keyword callout
        assert "Layer" not in text.split("[Events]")[0]  # styles intact

    def test_compose_ass_without_alignment_still_overlays(self, lesson: Path) -> None:
        project = Project.create(source=lesson, script_text="")
        edl = EditDecisionList(
            project_source=str(lesson),
            decisions=[
                Decision(
                    id="callout-000",
                    kind="callout",
                    source_line=0,
                    source_span=(1.0, 3.0),
                    params={"term": "welcome"},
                    rationale="manual",
                    status="accepted",
                )
            ],
        )
        media = probe(lesson)
        text = compose_ass(project, media, [], edl)
        assert text is not None
        assert "welcome" in text
