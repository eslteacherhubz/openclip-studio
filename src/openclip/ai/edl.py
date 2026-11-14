"""Typed, validated, JSON-serializable schema for AI Director editing plans.

An :class:`AIDirectorPlan` is the structured output of an AI provider: clips,
cuts, captions, B-roll, zooms, audio enhancement and export settings. Plans
are parsed from JSON with :func:`parse_ai_plan` (strict validation, raises
:class:`~openclip.ai.errors.AIPlanError`) and cleaned with
:func:`normalize_ai_plan` (pure, deterministic) before they are turned into
review items or director decisions. Nothing here touches the network or disk.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any

from openclip.ai.errors import AIPlanError

AI_PLAN_SCHEMA_VERSION = 1

#: Item lists that participate in span normalization (sorting, dedupe, IDs).
_KIND_FIELD = (
    ("clips", "clip"),
    ("cuts", "cut"),
    ("captions", "caption"),
    ("broll", "broll"),
    ("zooms", "zoom"),
)


@dataclass
class AISpan:
    """A half-open timeline range ``[start_seconds, end_seconds)``."""

    start_seconds: float
    end_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {"start_seconds": self.start_seconds, "end_seconds": self.end_seconds}


@dataclass
class AIClip:
    span: AISpan
    id: str = ""
    title: str = ""
    reason: str = ""
    score: float | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "span": self.span.to_dict(),
            "title": self.title,
            "reason": self.reason,
            "score": self.score,
            "extra": dict(self.extra),
        }


@dataclass
class AICut:
    span: AISpan
    id: str = ""
    reason: str = ""
    confidence: float | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "span": self.span.to_dict(),
            "reason": self.reason,
            "confidence": self.confidence,
            "extra": dict(self.extra),
        }


@dataclass
class AICaption:
    span: AISpan
    id: str = ""
    primary_text: str = ""
    support_text: str | None = None
    style: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "span": self.span.to_dict(),
            "primary_text": self.primary_text,
            "support_text": self.support_text,
            "style": self.style,
            "extra": dict(self.extra),
        }


@dataclass
class AIBRoll:
    span: AISpan
    id: str = ""
    query: str = ""
    source: str | None = None
    mode: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "span": self.span.to_dict(),
            "query": self.query,
            "source": self.source,
            "mode": self.mode,
            "extra": dict(self.extra),
        }


@dataclass
class AIZoom:
    span: AISpan
    id: str = ""
    mode: str | None = None
    strength: float | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "span": self.span.to_dict(),
            "mode": self.mode,
            "strength": self.strength,
            "extra": dict(self.extra),
        }


@dataclass
class AIAudioEnhancement:
    enabled: bool = False
    denoise: bool | None = None
    loudness_normalize: bool | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "denoise": self.denoise,
            "loudness_normalize": self.loudness_normalize,
            "extra": dict(self.extra),
        }


@dataclass
class AIExport:
    format: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    vertical: bool | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "vertical": self.vertical,
            "extra": dict(self.extra),
        }


@dataclass
class AIDirectorPlan:
    version: int
    provider: str
    prompt: str
    source_duration_seconds: float | None = None
    clips: list[AIClip] = field(default_factory=list)
    cuts: list[AICut] = field(default_factory=list)
    captions: list[AICaption] = field(default_factory=list)
    broll: list[AIBRoll] = field(default_factory=list)
    zooms: list[AIZoom] = field(default_factory=list)
    audio: AIAudioEnhancement | None = None
    export: AIExport | None = None
    notes: list[str] = field(default_factory=list)
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "provider": self.provider,
            "prompt": self.prompt,
            "source_duration_seconds": self.source_duration_seconds,
            "clips": [c.to_dict() for c in self.clips],
            "cuts": [c.to_dict() for c in self.cuts],
            "captions": [c.to_dict() for c in self.captions],
            "broll": [b.to_dict() for b in self.broll],
            "zooms": [z.to_dict() for z in self.zooms],
            "audio": self.audio.to_dict() if self.audio is not None else None,
            "export": self.export.to_dict() if self.export is not None else None,
            "notes": list(self.notes),
            "extra": dict(self.extra),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------


def _as_dict(data: object, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise AIPlanError(f"{context} must be a JSON object, got {type(data).__name__}")
    return data


def _as_list(data: object, context: str) -> list[Any]:
    if not isinstance(data, list):
        raise AIPlanError(f"{context} must be a JSON array, got {type(data).__name__}")
    return data


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AIPlanError(f"{context} must be a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise AIPlanError(f"{context} must be finite, got {value!r}")
    return result


def _optional_number(data: dict[str, Any], key: str, context: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return _number(value, f"{context}.{key}")


def _range_01(value: object, context: str) -> float:
    number = _number(value, context)
    if not 0.0 <= number <= 1.0:
        raise AIPlanError(f"{context} must be within [0, 1], got {number}")
    return number


def _optional_range_01(data: dict[str, Any], key: str, context: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return _range_01(value, f"{context}.{key}")


def _optional_str(data: dict[str, Any], key: str, context: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AIPlanError(f"{context}.{key} must be a string, got {type(value).__name__}")
    return value


def _parse_span(data: dict[str, Any], context: str) -> tuple[AISpan, dict[str, object]]:
    """Accept either a nested ``span`` object or flat ``start_seconds``/``end_seconds``."""
    extra: dict[str, object] = {}
    if "span" in data:
        span_data = _as_dict(data["span"], f"{context}.span")
        start_raw: object = span_data.get("start_seconds")
        end_raw: object = span_data.get("end_seconds")
        extra = {k: v for k, v in span_data.items() if k not in {"start_seconds", "end_seconds"}}
    else:
        start_raw = data.get("start_seconds")
        end_raw = data.get("end_seconds")
    if start_raw is None or end_raw is None:
        raise AIPlanError(f"{context} is missing start_seconds/end_seconds")
    start = _number(start_raw, f"{context}.start_seconds")
    end = _number(end_raw, f"{context}.end_seconds")
    if start < 0.0:
        raise AIPlanError(f"{context}.start_seconds must be non-negative, got {start}")
    if end < 0.0:
        raise AIPlanError(f"{context}.end_seconds must be non-negative, got {end}")
    if not end > start:
        raise AIPlanError(
            f"{context}.end_seconds ({end}) must be greater than start_seconds ({start})"
        )
    return AISpan(start_seconds=start, end_seconds=end), extra


def _take(data: dict[str, Any], keys: set[str]) -> dict[str, object]:
    return {k: v for k, v in data.items() if k not in keys}


def _merge_extra(
    item: dict[str, Any], span_extra: dict[str, object], rest: dict[str, object]
) -> dict[str, object]:
    incoming = item.get("extra")
    base: dict[str, object] = dict(incoming) if isinstance(incoming, dict) else {}
    return {**base, **span_extra, **rest}


def _parse_clip(data: object, context: str) -> AIClip:
    item = _as_dict(data, context)
    span, span_extra = _parse_span(item, context)
    known = {"span", "start_seconds", "end_seconds", "id", "title", "reason", "score", "extra"}
    return AIClip(
        span=span,
        id=str(item.get("id") or ""),
        title=str(item.get("title") or ""),
        reason=str(item.get("reason") or ""),
        score=_optional_range_01(item, "score", context),
        extra=_merge_extra(item, span_extra, _take(item, known)),
    )


def _parse_cut(data: object, context: str) -> AICut:
    item = _as_dict(data, context)
    span, span_extra = _parse_span(item, context)
    known = {"span", "start_seconds", "end_seconds", "id", "reason", "confidence", "extra"}
    return AICut(
        span=span,
        id=str(item.get("id") or ""),
        reason=str(item.get("reason") or ""),
        confidence=_optional_range_01(item, "confidence", context),
        extra=_merge_extra(item, span_extra, _take(item, known)),
    )


def _parse_caption(data: object, context: str) -> AICaption:
    item = _as_dict(data, context)
    span, span_extra = _parse_span(item, context)
    known = {
        "span",
        "start_seconds",
        "end_seconds",
        "id",
        "primary_text",
        "support_text",
        "style",
        "extra",
    }
    return AICaption(
        span=span,
        id=str(item.get("id") or ""),
        primary_text=str(item.get("primary_text") or ""),
        support_text=_optional_str(item, "support_text", context),
        style=_optional_str(item, "style", context),
        extra=_merge_extra(item, span_extra, _take(item, known)),
    )


def _parse_broll(data: object, context: str) -> AIBRoll:
    item = _as_dict(data, context)
    span, span_extra = _parse_span(item, context)
    known = {"span", "start_seconds", "end_seconds", "id", "query", "source", "mode", "extra"}
    return AIBRoll(
        span=span,
        id=str(item.get("id") or ""),
        query=str(item.get("query") or ""),
        source=_optional_str(item, "source", context),
        mode=_optional_str(item, "mode", context),
        extra=_merge_extra(item, span_extra, _take(item, known)),
    )


def _parse_zoom(data: object, context: str) -> AIZoom:
    item = _as_dict(data, context)
    span, span_extra = _parse_span(item, context)
    known = {"span", "start_seconds", "end_seconds", "id", "mode", "strength", "extra"}
    return AIZoom(
        span=span,
        id=str(item.get("id") or ""),
        mode=_optional_str(item, "mode", context),
        strength=_optional_range_01(item, "strength", context),
        extra=_merge_extra(item, span_extra, _take(item, known)),
    )


def _parse_items(data: dict[str, Any], key: str, parser: Any) -> list[Any]:
    raw = data.get(key)
    if raw is None:
        return []
    return [parser(item, f"{key}[{i}]") for i, item in enumerate(_as_list(raw, key))]


def _parse_audio(data: object, context: str) -> AIAudioEnhancement:
    item = _as_dict(data, context)
    known = {"enabled", "denoise", "loudness_normalize", "extra"}

    def _optional_bool(key: str) -> bool | None:
        value = item.get(key)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise AIPlanError(f"{context}.{key} must be a boolean, got {type(value).__name__}")
        return value

    return AIAudioEnhancement(
        enabled=item.get("enabled") is True,
        denoise=_optional_bool("denoise"),
        loudness_normalize=_optional_bool("loudness_normalize"),
        extra=_merge_extra(item, {}, _take(item, known)),
    )


def _parse_export(data: object, context: str) -> AIExport:
    item = _as_dict(data, context)
    known = {"format", "width", "height", "fps", "vertical", "extra"}
    width_raw = item.get("width")
    height_raw = item.get("height")
    for key, raw in (("width", width_raw), ("height", height_raw)):
        if raw is not None and (isinstance(raw, bool) or not isinstance(raw, int)):
            raise AIPlanError(f"{context}.{key} must be an integer, got {type(raw).__name__}")
    fps = _optional_number(item, "fps", context)
    if fps is not None and fps <= 0:
        raise AIPlanError(f"{context}.fps must be positive, got {fps}")
    vertical = item.get("vertical")
    if vertical is not None and not isinstance(vertical, bool):
        raise AIPlanError(f"{context}.vertical must be a boolean, got {type(vertical).__name__}")

    def _as_opt_int(raw: object) -> int | None:
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else None

    return AIExport(
        format=_optional_str(item, "format", context),
        width=_as_opt_int(width_raw),
        height=_as_opt_int(height_raw),
        fps=fps,
        vertical=vertical if isinstance(vertical, bool) else None,
        extra=_merge_extra(item, {}, _take(item, known)),
    )


_PLAN_KNOWN_KEYS = {
    "version",
    "provider",
    "prompt",
    "source_duration_seconds",
    "clips",
    "cuts",
    "captions",
    "broll",
    "zooms",
    "audio",
    "export",
    "notes",
    "extra",
}


def parse_ai_plan(data: object) -> AIDirectorPlan:
    """Parse and validate a decoded-JSON AI editing plan.

    Raises :class:`AIPlanError` on any invalid input. Tolerates missing
    optional sections and both nested-``span`` and flat-timing item shapes;
    unknown fields are preserved in ``extra`` dictionaries.
    """
    plan = _as_dict(data, "plan")
    if "version" not in plan:
        raise AIPlanError("plan is missing required field 'version'")
    version = plan["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise AIPlanError(f"plan.version must be an integer, got {type(version).__name__}")
    if version != AI_PLAN_SCHEMA_VERSION:
        raise AIPlanError(f"unsupported plan version {version} (expected {AI_PLAN_SCHEMA_VERSION})")
    provider = plan.get("provider")
    if not isinstance(provider, str) or not provider:
        raise AIPlanError("plan is missing required non-empty field 'provider'")
    prompt = plan.get("prompt")
    if not isinstance(prompt, str):
        raise AIPlanError("plan is missing required field 'prompt' (string)")

    duration = _optional_number(plan, "source_duration_seconds", "plan")
    if duration is not None and duration < 0.0:
        raise AIPlanError(f"plan.source_duration_seconds must be non-negative, got {duration}")

    raw_notes = plan.get("notes")
    if raw_notes is None:
        notes: list[str] = []
    else:
        notes = []
        for i, note in enumerate(_as_list(raw_notes, "plan.notes")):
            if not isinstance(note, str):
                raise AIPlanError(f"plan.notes[{i}] must be a string, got {type(note).__name__}")
            notes.append(note)

    audio = _parse_audio(plan["audio"], "plan.audio") if plan.get("audio") is not None else None
    export = (
        _parse_export(plan["export"], "plan.export") if plan.get("export") is not None else None
    )

    return AIDirectorPlan(
        version=version,
        provider=provider,
        prompt=prompt,
        source_duration_seconds=duration,
        clips=_parse_items(plan, "clips", _parse_clip),
        cuts=_parse_items(plan, "cuts", _parse_cut),
        captions=_parse_items(plan, "captions", _parse_caption),
        broll=_parse_items(plan, "broll", _parse_broll),
        zooms=_parse_items(plan, "zooms", _parse_zoom),
        audio=audio,
        export=export,
        notes=notes,
        extra=_merge_extra(plan, {}, _take(plan, _PLAN_KNOWN_KEYS)),
    )


def parse_ai_plan_json(text: str) -> AIDirectorPlan:
    """Parse an AI plan from a JSON string."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AIPlanError(f"plan is not valid JSON: {exc}") from exc
    return parse_ai_plan(data)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize_items(items: list[Any], kind: str, duration: float | None) -> list[Any]:
    """Clamp to source duration, drop exact duplicate spans, sort, assign IDs.

    Documented clamping rule: when ``source_duration_seconds`` is known, items
    whose span starts at or beyond the duration are removed; items that extend
    past the duration are clamped so ``end_seconds == duration``. Overlapping
    (but not identical) spans are kept — merging them is a review decision.
    """
    kept: list[Any] = []
    seen_spans: set[tuple[float, float]] = set()
    for item in items:
        start = item.span.start_seconds
        end = item.span.end_seconds
        if duration is not None:
            if start >= duration:
                continue
            if end > duration:
                end = duration
        key = (start, end)
        if key in seen_spans:
            continue
        seen_spans.add(key)
        kept.append(replace(item, span=AISpan(start_seconds=start, end_seconds=end)))
    kept.sort(key=lambda item: (item.span.start_seconds, item.id))
    taken_ids = {item.id for item in kept if item.id}
    result: list[Any] = []
    for index, item in enumerate(kept):
        if not item.id:
            candidate = f"{kind}-{index:03d}"
            suffix = index
            while candidate in taken_ids:
                suffix += 1
                candidate = f"{kind}-{suffix:03d}"
            taken_ids.add(candidate)
            item = replace(item, id=candidate)
        result.append(item)
    return result


def normalize_ai_plan(plan: AIDirectorPlan) -> AIDirectorPlan:
    """Return a deterministic, cleaned copy of ``plan`` (pure; input unchanged)."""
    duration = plan.source_duration_seconds
    return replace(
        plan,
        clips=_normalize_items(plan.clips, "clip", duration),
        cuts=_normalize_items(plan.cuts, "cut", duration),
        captions=_normalize_items(plan.captions, "caption", duration),
        broll=_normalize_items(plan.broll, "broll", duration),
        zooms=_normalize_items(plan.zooms, "zoom", duration),
    )
