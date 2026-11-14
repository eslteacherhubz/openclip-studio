"""Tests for the prompt-to-plan pipeline, prompts, and director extension."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openclip.ai import (
    AIDirectorPlan,
    AIPlanError,
    MockAIProvider,
    ProviderConfig,
    build_director_system_prompt,
    build_director_user_prompt,
    extract_json_object,
    generate_plan,
    load_provider_config,
    parse_ai_plan,
)
from openclip.director.models import DECISION_KINDS, Decision, EditDecisionList, check_invariants

FIXTURE = Path(__file__).parent / "fixtures" / "ai" / "mock_plan.json"


class _WrappedProvider:
    """Fake provider returning canned text, to exercise JSON extraction."""

    def __init__(self, response: str, name: str = "local_mock") -> None:
        self._response = response
        self._name = name
        self.calls: list[dict[str, object]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> ProviderConfig:
        return ProviderConfig(provider=self._name)

    def is_available(self) -> bool:
        return True

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        response_format: str | None = None,
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "system": system, "response_format": response_format}
        )
        return self._response


class TestExtractJson:
    def test_markdown_json_block(self) -> None:
        payload = '{"version": 1, "provider": "x", "prompt": "p"}'
        text = f"Here is your plan:\n```json\n{payload}\n```\nDone."
        assert extract_json_object(text) == {"version": 1, "provider": "x", "prompt": "p"}

    def test_plain_fence(self) -> None:
        text = '```\n{"a": 1}\n```'
        assert extract_json_object(text) == {"a": 1}

    def test_raw_json(self) -> None:
        assert extract_json_object('{"a": 2}') == {"a": 2}

    def test_json_embedded_in_prose(self) -> None:
        text = 'Sure! {"b": {"c": "}"}} hope that helps.'
        assert extract_json_object(text) == {"b": {"c": "}"}}

    def test_invalid_raises(self) -> None:
        with pytest.raises(AIPlanError):
            extract_json_object("no json here at all")

    def test_unterminated_raises(self) -> None:
        with pytest.raises(AIPlanError):
            extract_json_object('{"version": 1, "provider":')

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(AIPlanError):
            extract_json_object('{"version": 1,}')


class TestPrompts:
    def test_system_prompt_forces_json_schema(self) -> None:
        prompt = build_director_system_prompt()
        assert "OpenClip AI Director" in prompt
        assert "ONLY valid JSON" in prompt
        fields = (
            "version", "provider", "prompt", "clips", "cuts",
            "captions", "broll", "zooms", "audio", "export",
        )
        for field in fields:
            assert field in prompt
        assert "source_duration_seconds" in prompt

    def test_user_prompt_includes_inputs(self) -> None:
        msg = build_director_user_prompt("make a reel", "transcript text", 42.0)
        assert "make a reel" in msg
        assert "transcript text" in msg
        assert "42.0" in msg


class TestGeneratePlan:
    def test_mock_provider_end_to_end(self) -> None:
        provider = MockAIProvider(load_provider_config("local_mock", env={}))
        plan = generate_plan(provider, "highlight reel", "transcript...", 60.0)
        assert isinstance(plan, AIDirectorPlan)
        assert plan.provider == "local_mock"
        assert plan.source_duration_seconds == 60.0
        assert plan.clips and plan.cuts and plan.captions
        assert all(c.id for c in plan.clips)  # normalized IDs

    def test_markdown_wrapped_response(self) -> None:
        fixture = FIXTURE.read_text(encoding="utf-8")
        provider = _WrappedProvider(f"```json\n{fixture}\n```")
        plan = generate_plan(provider, "edit this", "transcript", 90.0)
        assert isinstance(plan, AIDirectorPlan)
        assert len(plan.clips) == 2
        assert plan.source_duration_seconds == 90.0
        # system prompt was sent and json response requested
        assert provider.calls[0]["response_format"] == "json"
        assert "OpenClip AI Director" in str(provider.calls[0]["system"])

    def test_raw_json_response(self) -> None:
        fixture = FIXTURE.read_text(encoding="utf-8")
        provider = _WrappedProvider(fixture)
        plan = generate_plan(provider, "edit this", "transcript", 90.0)
        assert len(plan.cuts) == 2

    def test_invalid_json_raises_plan_error(self) -> None:
        provider = _WrappedProvider("I cannot help with that.")
        with pytest.raises(AIPlanError):
            generate_plan(provider, "edit", "t", 10.0)

    def test_schema_invalid_plan_raises(self) -> None:
        bad = (
            '{"version": 1, "provider": "x", "prompt": "p", '
            '"cuts": [{"start_seconds": 5.0, "end_seconds": 1.0}]}'
        )
        provider = _WrappedProvider(bad)
        with pytest.raises(AIPlanError):
            generate_plan(provider, "edit", "t", 10.0)

    def test_missing_metadata_filled(self) -> None:
        payload = '{"clips": [{"span": {"start_seconds": 0.0, "end_seconds": 1.0}}]}'
        provider = _WrappedProvider(payload, name="openai")
        plan = generate_plan(provider, "my prompt", "t", 30.0)
        assert plan.version == 1
        assert plan.provider == "openai"
        assert plan.prompt == "my prompt"
        assert plan.source_duration_seconds == 30.0


class TestDirectorExtension:
    def test_new_kinds_registered(self) -> None:
        for kind in ("clip", "broll", "zoom", "audio_enhance", "export_config"):
            assert kind in DECISION_KINDS

    def test_new_kinds_pass_invariants(self) -> None:
        decisions = [
            Decision(
                id="z1", kind="zoom", source_span=(1.0, 2.0), rationale="why", status="pending"
            ),
            Decision(
                id="b1", kind="broll", source_span=(3.0, 4.0), rationale="why", status="pending"
            ),
            Decision(id="a1", kind="audio_enhance", rationale="why", status="pending"),
        ]
        edl = EditDecisionList(project_source="p", decisions=decisions)
        assert check_invariants(edl) == []

    def test_executor_skips_unimplemented_ai_kinds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Accepted clip/broll/zoom/audio/export decisions must not crash render_edl."""
        from openclip.director import executor
        from openclip.engine.ffmpeg import MediaInfo
        from openclip.engine.project import Project
        from openclip.engine.render import RenderResult

        captured: dict[str, object] = {}

        def _fake_render_export(**kwargs: object) -> RenderResult:
            captured.update(kwargs)
            return RenderResult(
                out=Path(str(kwargs["out_path"])),
                kept_duration_s=10.0,
                segments=1,
                cuts_applied=0,
                captions_burned=False,
                audio_enhanced=False,
            )

        monkeypatch.setattr(executor, "render_export", _fake_render_export)

        project = Project.create(source=tmp_path / "in.mp4", script_text="Hello :: Hola\n")
        media = MediaInfo(
            path=tmp_path / "in.mp4", duration_s=10.0, has_video=True, width=640, height=360
        )
        decisions = [
            Decision(
                id="z1", kind="zoom", source_span=(1.0, 2.0),
                rationale="zoom here", status="accepted",
            ),
            Decision(
                id="b1", kind="broll", source_span=(3.0, 4.0),
                rationale="b-roll", status="accepted",
            ),
            Decision(
                id="c1", kind="clip", source_span=(0.0, 5.0),
                rationale="keep only 0-5s", status="accepted",
            ),
            Decision(
                id="a1", kind="audio_enhance", rationale="denoise", status="accepted"
            ),
            Decision(id="e1", kind="export_config", rationale="mp4", status="accepted"),
        ]
        edl = EditDecisionList(project_source="in.mp4", decisions=decisions)
        result = executor.render_edl(project, media, edl, tmp_path / "out.mp4")
        assert result.out == tmp_path / "out.mp4"
        # clip 0–5s accepted → everything outside it is removed (Phase 6)
        assert captured["cut_spans"] == [(5.0, 10.0)]
        assert captured["kept"] == [(0.0, 5.0)]

    def test_executor_still_raises_for_legacy_unsupported(self, tmp_path: Path) -> None:
        from openclip.director import executor
        from openclip.engine.ffmpeg import MediaInfo
        from openclip.engine.project import Project

        project = Project.create(source=tmp_path / "in.mp4", script_text="Hello :: Hola\n")
        media = MediaInfo(path=tmp_path / "in.mp4", duration_s=10.0, has_video=True)
        edl = EditDecisionList(
            project_source="in.mp4",
            decisions=[Decision(id="r1", kind="reframe", rationale="x", status="accepted")],
        )
        with pytest.raises(NotImplementedError, match="reframe"):
            executor.render_edl(project, media, edl, tmp_path / "out.mp4")

    def test_bridge_maps_all_kinds(self) -> None:
        from openclip.ai import ai_plan_to_director_decisions, normalize_ai_plan

        plan = normalize_ai_plan(parse_ai_plan(json.loads(FIXTURE.read_text(encoding="utf-8"))))
        decisions = ai_plan_to_director_decisions(plan)
        kinds = {d.kind for d in decisions}
        expected = {"clip", "cut", "caption", "broll", "zoom", "audio_enhance", "export_config"}
        assert kinds == expected
        assert all(d.status == "pending" for d in decisions)
        assert all(d.params.get("origin") == "ai" for d in decisions)
        edl = EditDecisionList(project_source="p", decisions=decisions)
        assert check_invariants(edl) == []
