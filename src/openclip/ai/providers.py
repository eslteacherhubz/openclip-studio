"""AI provider abstraction and concrete providers.

All providers implement the :class:`AIProvider` protocol: a ``name``, their
``config``, an ``is_available`` check, and a ``complete`` call. ``local_mock``
works fully offline; real providers (OpenAI, Mistral, Anthropic) are backed by
stdlib-HTTP clients in :mod:`openclip.ai.clients`; Replicate and ElevenLabs
remain stubs until their milestone.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from openclip.ai.clients import AnthropicClient, OpenAICompatibleClient
from openclip.ai.config import ProviderConfig
from openclip.ai.errors import AIRuntimeError
from openclip.ai.mock import MockAIProvider


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


class _HTTPProvider:
    """Base for providers backed by a real HTTP client."""

    client_class: type[OpenAICompatibleClient] | type[AnthropicClient]

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client: OpenAICompatibleClient | AnthropicClient = self.client_class(config)

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
        return self._client.complete(prompt, system=system, response_format=response_format)


class OpenAIProvider(_HTTPProvider):
    """OpenAI chat completions."""

    client_class = OpenAICompatibleClient


class MistralProvider(_HTTPProvider):
    """Mistral chat completions (OpenAI-compatible API)."""

    client_class = OpenAICompatibleClient


class AnthropicProvider(_HTTPProvider):
    """Anthropic messages API."""

    client_class = AnthropicClient


class _StubProvider:
    """Configured provider whose HTTP client is not implemented yet."""

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
        raise AIRuntimeError(
            f"Real provider integration for {self._config.provider!r} will be added in a "
            "later milestone. Use 'openai', 'mistral', 'anthropic' or 'local_mock'."
        )


class ReplicateProvider(_StubProvider):
    """Replicate models (stub)."""


class ElevenLabsProvider(_StubProvider):
    """ElevenLabs audio/voice APIs (stub)."""


_PROVIDER_CLASSES: dict[str, Callable[[ProviderConfig], AIProvider]] = {
    "openai": OpenAIProvider,
    "mistral": MistralProvider,
    "anthropic": AnthropicProvider,
    "replicate": ReplicateProvider,
    "elevenlabs": ElevenLabsProvider,
    "local_mock": MockAIProvider,
}


def create_provider(config: ProviderConfig) -> AIProvider:
    """Instantiate the right provider for a loaded :class:`ProviderConfig`."""
    cls = _PROVIDER_CLASSES.get(config.provider)
    if cls is None:
        raise AIRuntimeError(f"Unknown AI provider {config.provider!r}")
    return cls(config)
