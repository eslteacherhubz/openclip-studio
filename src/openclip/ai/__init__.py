"""AI layer: BYOK provider configuration, offline mock provider, and the
AI Director EDL schema (parse → normalize → review → director bridge).

See docs/AI_INTEGRATION.md and docs/AI_DIRECTOR_EDL.md for usage and
security rules.
"""

from __future__ import annotations

from openclip.ai.broll_detection import BrollDetector, BrollOpportunity, ReasonType
from openclip.ai.broll_sourcing import BrollSource, BrollSourceResolver, MatchType
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
from openclip.ai.hook_detection import (
    HOOK_TYPES,
    Hook,
    HookDetector,
    HookType,
    hooks_to_director_plan,
)
from openclip.ai.mock import MockAIProvider
from openclip.ai.pipeline import extract_json_object, generate_plan
from openclip.ai.platform_presets import (
    PlatformPreset,
    get_preset,
    list_presets,
    preset_names,
)
from openclip.ai.prompts import build_director_system_prompt, build_director_user_prompt
from openclip.ai.providers import AIProvider, create_provider
from openclip.ai.reframing import (
    DEFAULT_TARGET_ASPECT,
    TrackingPoint,
    anchor_at,
    build_smart_reframe_filter,
    crop_size_for_aspect,
    mock_tracking_data,
)
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
    "DEFAULT_TARGET_ASPECT",
    "DEFAULT_ZOOM_STRENGTH",
    "HOOK_TYPES",
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
    "BrollDetector",
    "BrollOpportunity",
    "BrollSource",
    "BrollSourceResolver",
    "CaptionBlock",
    "CaptionStyle",
    "ExportConfigArgs",
    "Hook",
    "HookDetector",
    "HookType",
    "MatchType",
    "MockAIProvider",
    "OpenAICompatibleClient",
    "PlatformPreset",
    "ProviderConfig",
    "ProviderConfigError",
    "ProviderName",
    "ProviderNotAvailableError",
    "ReasonType",
    "ReviewStatus",
    "TrackingPoint",
    "TranscriptSegment",
    "ai_plan_to_director_decisions",
    "anchor_at",
    "audio_chain_for_enhancement",
    "build_broll_filter",
    "build_caption_filter",
    "build_director_system_prompt",
    "build_director_user_prompt",
    "build_export_config_args",
    "build_export_vf",
    "build_smart_reframe_filter",
    "build_zoom_vf",
    "create_provider",
    "crop_size_for_aspect",
    "decisions_to_json",
    "extract_json_object",
    "generate_plan",
    "get_caption_style",
    "get_preset",
    "hex_to_ffmpeg_color",
    "hooks_to_director_plan",
    "list_presets",
    "load_provider_config",
    "mock_tracking_data",
    "normalize_ai_plan",
    "parse_ai_plan",
    "parse_ai_plan_json",
    "parse_transcript",
    "plan_to_review_items",
    "preset_names",
    "to_ffmpeg_color",
    "transcript_to_edits",
]
