"""Tests for BYOK provider configuration loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openclip.ai import (
    ProviderConfig,
    ProviderConfigError,
    load_provider_config,
)

REAL_PROVIDERS = ["openai", "mistral", "anthropic", "replicate", "elevenlabs"]

KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
    "elevenlabs": "ELEVENLABS_API_KEY",
}


class TestLocalMock:
    def test_loads_without_api_key(self) -> None:
        config = load_provider_config("local_mock", env={})
        assert config.provider == "local_mock"
        assert config.api_key is None

    def test_ollama_is_local_and_needs_no_key(self) -> None:
        from openclip.ai.config import key_env_var_for

        config = load_provider_config("ollama", env={})
        assert config.provider == "ollama"
        assert config.api_key is None
        assert key_env_var_for("ollama") is None
        assert key_env_var_for("openai") == "OPENAI_API_KEY"

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ProviderConfigError):
            load_provider_config("not-a-provider", env={})


class TestMissingKeys:
    @pytest.mark.parametrize("provider", REAL_PROVIDERS)
    def test_missing_key_raises(self, provider: str) -> None:
        with pytest.raises(ProviderConfigError) as excinfo:
            load_provider_config(provider, env={})
        assert KEY_ENV_VARS[provider] in str(excinfo.value)


class TestKeyResolution:
    @pytest.mark.parametrize("provider", REAL_PROVIDERS)
    def test_provider_specific_env_var(self, provider: str) -> None:
        env = {KEY_ENV_VARS[provider]: "sk-test-123"}
        config = load_provider_config(provider, env=env)
        assert config.api_key == "sk-test-123"

    def test_generic_fallback_env_var(self) -> None:
        config = load_provider_config("openai", env={"OPENCLIP_AI_API_KEY": "sk-generic"})
        assert config.api_key == "sk-generic"

    def test_provider_specific_beats_generic(self) -> None:
        env = {"OPENAI_API_KEY": "sk-specific", "OPENCLIP_AI_API_KEY": "sk-generic"}
        config = load_provider_config("openai", env=env)
        assert config.api_key == "sk-specific"


class TestConfigFile:
    def test_config_file_values_are_read(self, tmp_path: Path) -> None:
        path = tmp_path / "ai.json"
        path.write_text(
            json.dumps(
                {
                    "provider": "mistral",
                    "model": "mistral-large-latest",
                    "base_url": "https://api.mistral.ai/v1",
                    "timeout_seconds": 60,
                }
            ),
            encoding="utf-8",
        )
        config = load_provider_config(
            "mistral", env={"MISTRAL_API_KEY": "sk-test"}, config_path=path
        )
        assert config.model == "mistral-large-latest"
        assert config.base_url == "https://api.mistral.ai/v1"
        assert config.timeout_seconds == 60.0
        assert config.api_key == "sk-test"

    def test_env_overrides_config_file(self, tmp_path: Path) -> None:
        path = tmp_path / "ai.json"
        path.write_text(
            json.dumps({"provider": "openai", "model": "gpt-4o-mini", "timeout_seconds": 60}),
            encoding="utf-8",
        )
        env = {"OPENAI_API_KEY": "sk-env", "OPENCLIP_AI_MODEL": "gpt-4o", "OPENCLIP_AI_BASE_URL": "https://example.test/v1"}
        config = load_provider_config("openai", env=env, config_path=path)
        assert config.model == "gpt-4o"
        assert config.base_url == "https://example.test/v1"
        assert config.timeout_seconds == 60.0

    def test_config_file_without_key_still_needs_env_key(self, tmp_path: Path) -> None:
        path = tmp_path / "ai.json"
        path.write_text(
            json.dumps({"provider": "anthropic", "model": "claude-3-5"}), encoding="utf-8"
        )
        with pytest.raises(ProviderConfigError):
            load_provider_config("anthropic", env={}, config_path=path)

    def test_mismatched_provider_in_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "ai.json"
        path.write_text(json.dumps({"provider": "mistral"}), encoding="utf-8")
        with pytest.raises(ProviderConfigError):
            load_provider_config("openai", env={"OPENAI_API_KEY": "sk"}, config_path=path)

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "ai.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ProviderConfigError):
            load_provider_config("openai", env={"OPENAI_API_KEY": "sk"}, config_path=path)

    def test_missing_config_file_is_ignored(self, tmp_path: Path) -> None:
        config = load_provider_config(
            "openai", env={"OPENAI_API_KEY": "sk"}, config_path=tmp_path / "nope.json"
        )
        assert config.api_key == "sk"
        assert config.model is None


class TestSecretMasking:
    def test_str_masks_api_key(self) -> None:
        config = ProviderConfig(provider="openai", api_key="sk-super-secret")
        text = str(config)
        assert "sk-super-secret" not in text
        assert "sk-super-secret" not in repr(config)
        assert "********" in text

    def test_str_without_key(self) -> None:
        config = ProviderConfig(provider="local_mock")
        assert "None" in str(config)
