"""Deterministic, offline mock AI provider for tests and demos.

No API key, no network, no randomness: ``complete`` always returns the same
JSON editing plan for a given prompt.
"""

from __future__ import annotations

import json

from openclip.ai.config import ProviderConfig

MOCK_PROVIDER_NAME = "local_mock"


class MockAIProvider:
    """Offline stand-in for a real LLM provider."""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self._config = config or ProviderConfig(provider=MOCK_PROVIDER_NAME)

    @property
    def name(self) -> str:
        return MOCK_PROVIDER_NAME

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def is_available(self) -> bool:
        return True

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        response_format: str | None = None,
    ) -> str:
        """Return a deterministic JSON editing plan echoing ``prompt``."""
        plan = {
            "version": 1,
            "provider": MOCK_PROVIDER_NAME,
            "prompt": prompt,
            "clips": [
                {
                    "title": "Mock clip",
                    "start_seconds": 0.0,
                    "end_seconds": 10.0,
                    "reason": "Mock reason",
                    "score": 0.8,
                }
            ],
            "cuts": [
                {
                    "start_seconds": 2.0,
                    "end_seconds": 3.0,
                    "reason": "silence",
                    "confidence": 0.9,
                }
            ],
            "captions": [
                {
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "primary_text": "Hello",
                    "support_text": "Hola",
                }
            ],
        }
        return json.dumps(plan, ensure_ascii=False, sort_keys=True)
