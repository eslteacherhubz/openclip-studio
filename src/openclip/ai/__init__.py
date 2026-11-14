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
from openclip.ai.render_primitives import (
    DEFAULT_ZOOM_STRENGTH,
    STYLE_TEMPLATES,
    CaptionBlock,
    CaptionStyle,
    ExportConfigArgs,
    audio_chain_for_enhancement,
    build_broll_filter,
    build_caption_filter,
    build_export_config_args,
    build_export_vf,
    build_zoom_vf,
    get_caption_style,
    hex_to_ffmpeg_color,
    to_ffmpeg_color,
)
from openclip.ai.review import AIReviewItem, ReviewStatus, plan_to_review_items
from openclip.ai.transcript_editing import (
    TranscriptSegment,
    decisions_to_json,
    parse_transcript,
    transcript_to_edits,
)

__all__ = [
    "AI_PLAN_SCHEMA_VERSION",
    "DEFAULT_ZOOM_STRENGTH",
    "PROVIDER_NAMES",
    "STYLE_TEMPLATES",
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
    "CaptionBlock",
    "CaptionStyle",
    "ExportConfigArgs",
    "MockAIProvider",
    "OpenAICompatibleClient",
    "ProviderConfig",
    "ProviderConfigError",
    "ProviderName",
    "ProviderNotAvailableError",
    "ReviewStatus",
    "TranscriptSegment",
    "ai_plan_to_director_decisions",
    "audio_chain_for_enhancement",
    "build_broll_filter",
    "build_caption_filter",
    "build_director_system_prompt",
    "build_director_user_prompt",
    "build_export_config_args",
    "build_export_vf",
    "build_zoom_vf",
    "create_provider",
    "decisions_to_json",
    "extract_json_object",
    "generate_plan",
    "get_caption_style",
    "hex_to_ffmpeg_color",
    "load_provider_config",
    "normalize_ai_plan",
    "parse_ai_plan",
    "parse_ai_plan_json",
    "parse_transcript",
    "plan_to_review_items",
    "to_ffmpeg_color",
    "transcript_to_edits",
]
