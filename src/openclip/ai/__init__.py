"""AI layer: BYOK provider configuration, offline mock provider, and the
AI Director EDL schema (parse → normalize → review → director bridge).

See docs/AI_INTEGRATION.md and docs/AI_DIRECTOR_EDL.md for usage and
security rules.
"""

from __future__ import annotations

from openclip.ai.clients import AnthropicClient, OpenAICompatibleClient
from openclip.ai.config import PROVIDER_NAMES, ProviderConfig, ProviderName, load_provider_config
from openclip.ai.director_bridge import ai_plan_to_director_decisions
from openclip.ai.edl import (
    AI_PLAN_SCHEMA_VERSION,
    AIAudioEnhancement,
    AIBRoll,
    AICaption,
    AIClip,
    AICut,
    AIDirectorPlan,
    AIExport,
    AISpan,
    AIZoom,
    normalize_ai_plan,
    parse_ai_plan,
    parse_ai_plan_json,
)
from openclip.ai.errors import (
    AIConfigError,
    AIPlanError,
    AIRuntimeError,
    ProviderConfigError,
    ProviderNotAvailableError,
)
from openclip.ai.mock import MockAIProvider
from openclip.ai.pipeline import extract_json_object, generate_plan
from openclip.ai.prompts import build_director_system_prompt, build_director_user_prompt
from openclip.ai.providers import AIProvider, create_provider
from openclip.ai.review import AIReviewItem, ReviewStatus, plan_to_review_items

__all__ = [
    "AI_PLAN_SCHEMA_VERSION",
    "PROVIDER_NAMES",
    "AIAudioEnhancement",
    "AIBRoll",
    "AICaption",
    "AIClip",
    "AIConfigError",
    "AICut",
    "AIDirectorPlan",
    "AIExport",
    "AIPlanError",
    "AIProvider",
    "AIReviewItem",
    "AIRuntimeError",
    "AISpan",
    "AIZoom",
    "AnthropicClient",
    "MockAIProvider",
    "OpenAICompatibleClient",
    "ProviderConfig",
    "ProviderConfigError",
    "ProviderName",
    "ProviderNotAvailableError",
    "ReviewStatus",
    "ai_plan_to_director_decisions",
    "build_director_system_prompt",
    "build_director_user_prompt",
    "create_provider",
    "extract_json_object",
    "generate_plan",
    "load_provider_config",
    "normalize_ai_plan",
    "parse_ai_plan",
    "parse_ai_plan_json",
    "plan_to_review_items",
]
