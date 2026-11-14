"""BYOK provider configuration: environment variables and local JSON files.

Resolution order for each field (first match wins):

1. Environment variables (provider-specific, then generic OPENCLIP_AI_*).
2. A local JSON config file (``config_path``), which must never be committed.
3. Built-in defaults.

No network calls are made here; this module only reads configuration.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from openclip.ai.errors import ProviderConfigError

ProviderName = Literal[
    "openai",
    "mistral",
    "anthropic",
    "replicate",
    "elevenlabs",
    "ollama",
    "local_mock",
]

PROVIDER_NAMES: tuple[str, ...] = (
    "openai",
    "mistral",
    "anthropic",
    "replicate",
    "elevenlabs",
    "ollama",
    "local_mock",
)

#: Provider-specific environment variable that carries the API key.
PROVIDER_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
    "elevenlabs": "ELEVENLABS_API_KEY",
}

#: Generic fallback environment variables for any provider.
GENERIC_ENV_VARS: dict[str, str] = {
    "api_key": "OPENCLIP_AI_API_KEY",
    "base_url": "OPENCLIP_AI_BASE_URL",
    "model": "OPENCLIP_AI_MODEL",
}

#: Providers that run locally and need no API key (e.g. ollama on localhost).
KEY_OPTIONAL_PROVIDERS = frozenset({"ollama", "local_mock"})


def key_env_var_for(provider: str) -> str | None:
    """The environment variable that carries this provider's API key, if any."""
    return PROVIDER_KEY_ENV_VARS.get(provider)


DEFAULT_TIMEOUT_SECONDS = 30.0

_MASK = "********"


@dataclass
class ProviderConfig:
    """Configuration for one AI provider.

    Safe to print: ``str``/``repr`` mask the API key.
    """

    provider: str
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    extra: dict[str, object] = field(default_factory=dict)

    def _masked_key(self) -> str:
        if self.api_key is None:
            return "None"
        return _MASK

    def __str__(self) -> str:
        return (
            f"ProviderConfig(provider={self.provider!r}, api_key={self._masked_key()}, "
            f"base_url={self.base_url!r}, model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, extra={self.extra!r})"
        )

    __repr__ = __str__


def _read_config_file(config_path: Path) -> dict[str, object]:
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProviderConfigError(f"Cannot read AI config file {config_path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderConfigError(f"AI config file {config_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProviderConfigError(
            f"AI config file {config_path} must contain a JSON object at the top level"
        )
    return data


def _coerce_timeout(value: object, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderConfigError(f"timeout_seconds in {source} must be a number, got {value!r}")
    timeout = float(value)
    if timeout <= 0:
        raise ProviderConfigError(f"timeout_seconds in {source} must be positive, got {timeout}")
    return timeout


def load_provider_config(
    provider: str,
    env: Mapping[str, str] | None = None,
    config_path: Path | None = None,
) -> ProviderConfig:
    """Load configuration for ``provider`` from env vars and an optional JSON file.

    Environment variables override config-file values. The API key is never
    read from the config file. Raises :class:`ProviderConfigError` if a real
    provider has no API key or the name is unknown. Never touches the network.
    """
    if provider not in PROVIDER_NAMES:
        raise ProviderConfigError(
            f"Unknown AI provider {provider!r}; supported providers: {', '.join(PROVIDER_NAMES)}"
        )

    environ: Mapping[str, str] = env if env is not None else os.environ

    file_values: dict[str, object] = {}
    if config_path is not None and config_path.exists():
        file_values = _read_config_file(config_path)
        file_provider = file_values.get("provider")
        if isinstance(file_provider, str) and file_provider != provider:
            raise ProviderConfigError(
                f"AI config file {config_path} declares provider {file_provider!r}, "
                f"but {provider!r} was requested"
            )

    def pick(field_name: str) -> object | None:
        generic = environ.get(GENERIC_ENV_VARS[field_name])
        if generic:
            return generic
        if field_name in file_values and file_values[field_name] is not None:
            return file_values[field_name]
        return None

    api_key: str | None = None
    if provider not in KEY_OPTIONAL_PROVIDERS:
        key_env = PROVIDER_KEY_ENV_VARS[provider]
        api_key = environ.get(key_env) or environ.get(GENERIC_ENV_VARS["api_key"]) or None
        if not api_key:
            raise ProviderConfigError(
                f"No API key found for provider {provider!r}. Set the {key_env} "
                f"environment variable (or the generic OPENCLIP_AI_API_KEY fallback). "
                "Never commit API keys to the repository."
            )

    base_url_raw = pick("base_url")
    model_raw = pick("model")
    timeout_raw: object = environ.get("OPENCLIP_AI_TIMEOUT_SECONDS")
    if timeout_raw is None:
        timeout_raw = file_values.get("timeout_seconds")
    if timeout_raw is None:
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    else:
        timeout_seconds = _coerce_timeout(timeout_raw, "OPENCLIP_AI_TIMEOUT_SECONDS or config file")

    known_keys = {"provider", "base_url", "model", "timeout_seconds"}
    extra = {k: v for k, v in file_values.items() if k not in known_keys}

    return ProviderConfig(
        provider=provider,
        api_key=api_key,
        base_url=str(base_url_raw) if base_url_raw is not None else None,
        model=str(model_raw) if model_raw is not None else None,
        timeout_seconds=timeout_seconds,
        extra=extra,
    )
