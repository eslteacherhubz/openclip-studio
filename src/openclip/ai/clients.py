"""Real BYOK provider HTTP clients (stdlib ``urllib`` only, no new deps).

These clients are only exercised against mocked ``urllib.request.urlopen`` in
tests — no test makes a real network call. All failures raise
:class:`~openclip.ai.errors.AIRuntimeError`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from openclip.ai.config import ProviderConfig
from openclip.ai.errors import AIRuntimeError

DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "anthropic": "https://api.anthropic.ai",
}

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "mistral": "mistral-large-latest",
    "anthropic": "claude-sonnet-4-5",
}

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 8192


def _post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> Any:
    """POST ``body`` as JSON and return the decoded response payload."""
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            detail = ""
        raise AIRuntimeError(
            f"Provider HTTP {exc.code} from {url}: {detail or exc.reason}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AIRuntimeError(f"Provider request to {url} failed: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AIRuntimeError(f"Provider returned invalid JSON from {url}: {exc}") from exc


def _require_str(payload: Any, path: str) -> str:
    node: Any = payload
    for part in path.split("."):
        if part.isdigit() and isinstance(node, list):
            index = int(part)
            if index >= len(node):
                raise AIRuntimeError(f"Provider response missing {path!r}")
            node = node[index]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise AIRuntimeError(f"Provider response missing {path!r}")
    if not isinstance(node, str) or not node:
        raise AIRuntimeError(f"Provider response field {path!r} is not a non-empty string")
    return node


class OpenAICompatibleClient:
    """Chat-completions client for OpenAI and Mistral (same API shape)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise AIRuntimeError(f"Provider {config.provider!r} has no API key configured")
        self._config = config
        self._base_url = (config.base_url or DEFAULT_BASE_URLS.get(config.provider, "")).rstrip("/")
        self._model = config.model or DEFAULT_MODELS.get(config.provider, "")
        self._timeout = config.timeout_seconds

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        response_format: str | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {"model": self._model, "messages": messages}
        if response_format == "json":
            body["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        payload = _post_json(f"{self._base_url}/chat/completions", headers, body, self._timeout)
        return _require_str(payload, "choices.0.message.content")


class AnthropicClient:
    """Messages-API client for Anthropic."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise AIRuntimeError(f"Provider {config.provider!r} has no API key configured")
        self._config = config
        self._base_url = (config.base_url or DEFAULT_BASE_URLS.get(config.provider, "")).rstrip("/")
        self._model = config.model or DEFAULT_MODELS.get(config.provider, "")
        self._timeout = config.timeout_seconds

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        response_format: str | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        if response_format == "json":
            # Anthropic has no response_format flag; reinforce in the prompt.
            body["messages"][0]["content"] = (
                prompt + "\n\nRespond with ONLY a single JSON object, no markdown fences."
            )
        headers = {
            "x-api-key": self._config.api_key or "",
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        payload = _post_json(f"{self._base_url}/v1/messages", headers, body, self._timeout)
        return _require_str(payload, "content.0.text")
