"""Tests for the deterministic offline mock AI provider."""

from __future__ import annotations

import json

from openclip.ai import AIProvider, MockAIProvider, load_provider_config


def _provider() -> MockAIProvider:
    return MockAIProvider(load_provider_config("local_mock", env={}))


class TestMockProvider:
    def test_implements_protocol(self) -> None:
        assert isinstance(_provider(), AIProvider)

    def test_no_api_key_required_and_available(self) -> None:
        provider = _provider()
        assert provider.config.api_key is None
        assert provider.is_available() is True
        assert provider.name == "local_mock"

    def test_complete_returns_valid_json(self) -> None:
        result = _provider().complete("Summarize this lesson")
        plan = json.loads(result)
        assert isinstance(plan, dict)

    def test_complete_top_level_fields(self) -> None:
        plan = json.loads(_provider().complete("Make a highlight reel"))
        assert plan["version"] == 1
        assert plan["provider"] == "local_mock"
        assert plan["prompt"] == "Make a highlight reel"
        for key in ("clips", "cuts", "captions"):
            assert key in plan
            assert isinstance(plan[key], list)
            assert len(plan[key]) >= 1

    def test_complete_clip_structure(self) -> None:
        plan = json.loads(_provider().complete("p"))
        clip = plan["clips"][0]
        assert clip["title"] == "Mock clip"
        assert clip["start_seconds"] == 0.0
        assert clip["end_seconds"] == 10.0
        assert clip["reason"] == "Mock reason"
        assert clip["score"] == 0.8

    def test_complete_is_deterministic(self) -> None:
        provider = _provider()
        first = provider.complete("same prompt")
        second = provider.complete("same prompt")
        assert first == second

    def test_complete_echoes_prompt(self) -> None:
        plan = json.loads(_provider().complete("unique prompt 12345"))
        assert plan["prompt"] == "unique prompt 12345"

    def test_complete_accepts_optional_args(self) -> None:
        result = _provider().complete("p", system="sys", response_format="json")
        assert json.loads(result)["prompt"] == "p"
