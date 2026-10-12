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


class TestReviewEnforcement:
    """Rejected/pending captions must not render; burn_captions honored."""

    def _edl_with_caption_statuses(
        self, project: Project, statuses: dict[str, str]
    ) -> EditDecisionList:
        edl = plan_from_project(project)
        for d in edl.decisions:
            d.status = statuses.get(d.id, d.status)
        return edl

    def test_rejected_caption_excluded_from_ass(self, lesson: Path) -> None:
        project = _project_with_markers(lesson)
        edl = self._edl_with_caption_statuses(
            project, {"caption-001": "rejected"}
        )
        for d in edl.decisions:
            if d.status == "pending":
                d.status = "accepted" if d.kind != "diagram_slot" else "rejected"
        text = compose_ass(project, probe(lesson), [], edl)
        assert text is not None
        assert "Hello everyone" not in text, "rejected caption leaked into the ASS"
        assert "Today we will learn" in text, "accepted caption missing"

    def test_pending_caption_not_rendered_even_without_review_gate(
        self, lesson: Path
    ) -> None:
        project = _project_with_markers(lesson)
        edl = self._edl_with_caption_statuses(
            project, {"caption-001": "pending", "caption-005": "accepted"}
        )
        for d in edl.decisions:
            if d.status == "pending" and d.id != "caption-001":
                d.status = "accepted" if d.kind != "diagram_slot" else "rejected"
        text = compose_ass(project, probe(lesson), [], edl)
        assert text is not None
        assert "Hello everyone" not in text, "pending caption rendered"
        assert "Today we will learn" in text

    def test_burn_captions_false_suppresses_captions_only(self, lesson: Path) -> None:
        project = _project_with_markers(lesson)
        edl = self._edl_with_caption_statuses(project, {})
        for d in edl.decisions:
            d.status = "accepted" if d.kind != "diagram_slot" else "rejected"
        media = probe(lesson)
        text = compose_ass(project, media, [], edl, burn_captions=False)
        assert text is not None
        assert ",Primary,," not in text, "caption event rendered with burn_captions=False"
        assert ",Support,," not in text
        assert ",Callout,," in text, "callout must survive burn_captions=False"

    def test_burn_captions_false_without_overlays_skips_video_pass(
        self, lesson: Path
    ) -> None:
        project = _project_with_markers(lesson)
        edl = self._edl_with_caption_statuses(project, {})
        for d in edl.decisions:
            if d.kind == "caption":
                d.status = "accepted"
        text = compose_ass(project, probe(lesson), [], edl, burn_captions=False)
        assert text is None

    def test_mixed_states_render_only_accepted(
        self, lesson: Path, tmp_path: Path
    ) -> None:
        project = _project_with_markers(lesson)
        edl = self._edl_with_caption_statuses(
            project, {"caption-001": "rejected", "caption-005": "accepted"}
        )
        for d in edl.decisions:
            if d.status == "pending":
                d.status = "rejected"
        media = probe(lesson)
        out = tmp_path / "mixed.mp4"
        result = render_edl(project, media, edl, out, require_review=False)
        assert out.exists()
        assert result.segments >= 1


class TestOverlayTimelineMapping:
    """Overlays must use export-timeline timestamps after accepted cuts."""

    def _bare_project(self, lesson: Path) -> Project:
        return Project.create(source=lesson, script_text="")

    def _overlay_edl(self, source: Path, spans: list[tuple[str, float, float]]) -> EditDecisionList:
        decisions = [
            Decision(
                id=f"callout-{i:03d}",
                kind="callout",
                source_line=i,
                source_span=(s, e),
                params={"term": term},
                rationale="test overlay",
                status="accepted",
            )
            for i, (term, s, e) in enumerate(spans)
        ]
        return EditDecisionList(project_source=str(source), decisions=decisions)

    def test_callout_after_cut_uses_mapped_export_time(self, lesson: Path) -> None:
        project = self._bare_project(lesson)
        edl = self._overlay_edl(lesson, [("AFTER_CUT", 5.0, 6.0)])
        media = probe(lesson)
        text = compose_ass(project, media, [(1.0, 2.0)], edl)
        assert text is not None
        assert "Dialogue: 2,0:00:04.00,0:00:05.05,Callout" in text

    def test_overlay_fully_inside_cut_is_dropped(self, lesson: Path) -> None:
        project = self._bare_project(lesson)
        edl = self._overlay_edl(
            lesson, [("KEPT_ONE", 0.2, 0.8), ("LEAKED_TWO", 1.2, 1.8)]
        )
        media = probe(lesson)
        text = compose_ass(project, media, [(1.0, 2.0)], edl)
        assert text is not None
        assert "KEPT_ONE" in text
        assert "LEAKED_TWO" not in text, "overlay inside a cut must not render"

    def test_overlay_straddling_cut_start_clamps(self, lesson: Path) -> None:
        project = self._bare_project(lesson)
        edl = self._overlay_edl(lesson, [("STRADDLE", 0.5, 1.5)])
        media = probe(lesson)
        text = compose_ass(project, media, [(1.0, 2.0)], edl)
        assert text is not None
        # Clipped to (0.5, 1.0): no cut before 1.0, so export times are raw
        # (with the +0.05s display tail the executor adds).
        assert "Dialogue: 2,0:00:00.50,0:00:01.05,Callout" in text

    def test_overlay_after_overlapping_cuts_counts_union_once(
        self, lesson: Path
    ) -> None:
        project = self._bare_project(lesson)
        edl = self._overlay_edl(lesson, [("UNION_TEST", 10.0, 10.5)])
        media = probe(lesson)
        # Overlapping cuts [3,7]+[5,9] = 6 s removed before the overlay.
        text = compose_ass(project, media, [(3.0, 7.0), (5.0, 9.0)], edl)
        assert text is not None
        assert "Dialogue: 2,0:00:04.00,0:00:04.55,Callout" in text

    def test_kept_from_overlapping_accepted_cuts(self, lesson: Path) -> None:
        from openclip.director.executor import kept_intervals_from_spans

        kept = kept_intervals_from_spans([(3.0, 7.0), (5.0, 9.0)], 12.0)
        assert sum(e - s for s, e in kept) == 6.0  # 12 - union(6)

    def test_all_overlays_cut_away_returns_none(self, lesson: Path) -> None:
        project = self._bare_project(lesson)
        edl = self._overlay_edl(lesson, [("GONE", 3.5, 4.5)])
        media = probe(lesson)
        assert compose_ass(project, media, [(3.0, 5.0)], edl) is None


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
