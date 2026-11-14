"""Tests for the real provider HTTP clients — urllib is mocked, no network."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from openclip.ai import AIRuntimeError, ProviderConfig
from openclip.ai.clients import AnthropicClient, OpenAICompatibleClient


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _openai_config() -> ProviderConfig:
    return ProviderConfig(
        provider="openai",
        api_key="sk-test",
        base_url="https://api.openai.test/v1",
        model="gpt-test",
        timeout_seconds=5.0,
    )


def _anthropic_config() -> ProviderConfig:
    return ProviderConfig(
        provider="anthropic",
        api_key="sk-ant-test",
        base_url="https://api.anthropic.test",
        model="claude-test",
        timeout_seconds=5.0,
    )


class _Capture:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float = 0) -> _FakeResponse:
        self.requests.append(request)
        return _FakeResponse(self.payload)


class TestOpenAICompatibleClient:
    def test_request_format_and_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _Capture(
            {"choices": [{"message": {"content": '{"version": 1}'}}]}
        )
        monkeypatch.setattr(urllib.request, "urlopen", capture)
        client = OpenAICompatibleClient(_openai_config())
        result = client.complete("hello", system="sys", response_format="json")
        assert result == '{"version": 1}'

        req = capture.requests[0]
        assert req.full_url == "https://api.openai.test/v1/chat/completions"
        assert req.get_header("Authorization") == "Bearer sk-test"
        assert req.get_header("Content-type") == "application/json"
        body = json.loads(req.data.decode("utf-8"))  # type: ignore[arg-type]
        assert body["model"] == "gpt-test"
        assert body["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        assert body["response_format"] == {"type": "json_object"}

    def test_no_response_format_when_not_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _Capture({"choices": [{"message": {"content": "ok"}}]})
        monkeypatch.setattr(urllib.request, "urlopen", capture)
        client = OpenAICompatibleClient(_openai_config())
        assert client.complete("hi") == "ok"
        body = json.loads(capture.requests[0].data.decode("utf-8"))  # type: ignore[arg-type]
        assert "response_format" not in body
        assert body["messages"] == [{"role": "user", "content": "hi"}]

    def test_mistral_uses_openai_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _Capture({"choices": [{"message": {"content": "bonjour"}}]})
        monkeypatch.setattr(urllib.request, "urlopen", capture)
        config = ProviderConfig(
            provider="mistral",
            api_key="mk-test",
            base_url="https://api.mistral.test/v1",
            model="mistral-test",
        )
        client = OpenAICompatibleClient(config)
        assert client.complete("hi") == "bonjour"
        req = capture.requests[0]
        assert req.full_url == "https://api.mistral.test/v1/chat/completions"
        assert req.get_header("Authorization") == "Bearer mk-test"

    def test_http_error_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(request: urllib.request.Request, timeout: float = 0) -> _FakeResponse:
            raise urllib.error.HTTPError(
                request.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}')
            )

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        client = OpenAICompatibleClient(_openai_config())
        with pytest.raises(AIRuntimeError, match="401"):
            client.complete("hi")

    def test_url_error_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(request: urllib.request.Request, timeout: float = 0) -> _FakeResponse:
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        client = OpenAICompatibleClient(_openai_config())
        with pytest.raises(AIRuntimeError, match="failed"):
            client.complete("hi")

    def test_timeout_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(request: urllib.request.Request, timeout: float = 0) -> _FakeResponse:
            raise TimeoutError("timed out")

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        client = OpenAICompatibleClient(_openai_config())
        with pytest.raises(AIRuntimeError):
            client.complete("hi")

    def test_invalid_json_response_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _Capture.__new__(_Capture)
        capture.payload = None
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request, timeout=0: _RawResponse(b"not json"),
        )
        client = OpenAICompatibleClient(_openai_config())
        with pytest.raises(AIRuntimeError, match="invalid JSON"):
            client.complete("hi")

    def test_missing_choices_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _Capture({"unexpected": True})
        monkeypatch.setattr(urllib.request, "urlopen", capture)
        client = OpenAICompatibleClient(_openai_config())
        with pytest.raises(AIRuntimeError, match="choices"):
            client.complete("hi")

    def test_missing_api_key_raises(self) -> None:
        config = ProviderConfig(provider="openai", api_key=None)
        with pytest.raises(AIRuntimeError):
            OpenAICompatibleClient(config)


class _RawResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _RawResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class TestAnthropicClient:
    def test_request_format_and_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _Capture({"content": [{"type": "text", "text": '{"version": 1}'}]})
        monkeypatch.setattr(urllib.request, "urlopen", capture)
        client = AnthropicClient(_anthropic_config())
        result = client.complete("hello", system="sys", response_format="json")
        assert result == '{"version": 1}'

        req = capture.requests[0]
        assert req.full_url == "https://api.anthropic.test/v1/messages"
        assert req.get_header("X-api-key") == "sk-ant-test"
        assert req.get_header("Anthropic-version") == "2023-06-01"
        body = json.loads(req.data.decode("utf-8"))  # type: ignore[arg-type]
        assert body["model"] == "claude-test"
        assert body["system"] == "sys"
        assert body["max_tokens"] > 0
        expected_content = (
            "hello\n\nRespond with ONLY a single JSON object, no markdown fences."
        )
        assert body["messages"] == [{"role": "user", "content": expected_content}]

    def test_http_error_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(request: urllib.request.Request, timeout: float = 0) -> _FakeResponse:
            raise urllib.error.HTTPError(
                request.full_url, 429, "Too Many Requests", {}, io.BytesIO(b"rate limited")
            )

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        client = AnthropicClient(_anthropic_config())
        with pytest.raises(AIRuntimeError, match="429"):
            client.complete("hi")

    def test_missing_content_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = _Capture({"id": "msg_1"})
        monkeypatch.setattr(urllib.request, "urlopen", capture)
        client = AnthropicClient(_anthropic_config())
        with pytest.raises(AIRuntimeError, match="content"):
            client.complete("hi")
