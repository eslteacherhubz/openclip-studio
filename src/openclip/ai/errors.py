"""Error hierarchy for the AI provider layer."""

from __future__ import annotations


class AIConfigError(Exception):
    """Base error for AI configuration problems."""


class ProviderConfigError(AIConfigError):
    """A provider is misconfigured (e.g. missing API key)."""


class ProviderNotAvailableError(AIConfigError):
    """A provider is configured but cannot be used right now."""


class AIPlanError(AIConfigError):
    """An AI-generated editing plan is invalid or cannot be parsed."""
