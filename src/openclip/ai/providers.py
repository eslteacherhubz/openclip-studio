"""AI provider abstraction.

All providers implement the :class:`AIProvider` protocol: a ``name``, their
``config``, an ``is_available`` check, and a ``complete`` call. Real provider
classes are stubs that raise ``NotImplementedError`` until their milestone;
:class:`~openclip.ai.mock.MockAIProvider` works fully and offline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openclip.ai.config import ProviderConfig


@runtime_checkable
class AIProvider(Protocol):
    """Common interface for every AI backend."""

    @property
    def name(self) -> str: ...

    @property
    def config(self) -> ProviderConfig: ...

    def is_available(self) -> bool:
        """Return True if the provider can serve requests right now."""
        ...

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        response_format: str | None = None,
    ) -> str:
        """Return a completion for ``prompt`` (plain text or JSON string)."""
        ...


class _StubProvider:
    """Base for real providers: configured, but not yet implemented."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return self._config.provider

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def is_available(self) -> bool:
        return self._config.api_key is not None

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        response_format: str | None = None,
    ) -> str:
        raise NotImplementedError(
            f"Real provider integration for {self._config.provider!r} will be added in a "
            "later milestone. Use the 'local_mock' provider for offline testing."
        )


class OpenAIProvider(_StubProvider):
    """OpenAI chat completions (stub)."""


class MistralProvider(_StubProvider):
    """Mistral chat completions (stub)."""


class AnthropicProvider(_StubProvider):
    """Anthropic messages (stub)."""


class ReplicateProvider(_StubProvider):
    """Replicate models (stub)."""


class ElevenLabsProvider(_StubProvider):
    """ElevenLabs audio/voice APIs (stub)."""
