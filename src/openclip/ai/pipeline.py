"""Prompt-to-plan pipeline: provider response → validated, normalized plan.

LLMs often wrap JSON in markdown fences or prose; :func:`extract_json_object`
safely pulls the JSON object out. The result always goes through
:func:`~openclip.ai.edl.parse_ai_plan` and
:func:`~openclip.ai.edl.normalize_ai_plan` before it can be reviewed or
rendered. No network calls happen here — the provider does that.
"""

from __future__ import annotations

import json
import re
from typing import Any

from openclip.ai.edl import (
    AI_PLAN_SCHEMA_VERSION,
    AIDirectorPlan,
    normalize_ai_plan,
    parse_ai_plan,
)
from openclip.ai.errors import AIPlanError
from openclip.ai.prompts import build_director_system_prompt, build_director_user_prompt
from openclip.ai.providers import AIProvider

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> Any:
    """Extract the first JSON object from an LLM response.

    Tries, in order: fenced code blocks (```json ... ```), the whole text, then
    the first balanced ``{...}`` region. Raises :class:`AIPlanError` if no
    valid JSON object can be found.
    """
    for match in _FENCE_RE.finditer(text):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise AIPlanError("Provider response contains no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise AIPlanError(
                        f"Provider response contains a malformed JSON object: {exc}"
                    ) from exc
    raise AIPlanError("Provider response contains an unterminated JSON object")


def generate_plan(
    provider: AIProvider,
    user_prompt: str,
    transcript: str,
    source_duration_seconds: float,
) -> AIDirectorPlan:
    """Run the full prompt-to-plan pipeline and return a normalized plan.

    Steps: build the director system/user prompts, call
    ``provider.complete(..., response_format="json")``, extract the JSON from
    the response, fill in known metadata when the model omitted it, then
    parse (strict validation) and normalize (pure cleanup). Raises
    :class:`AIPlanError` for unparseable/invalid plans and
    :class:`~openclip.ai.errors.AIRuntimeError` for provider failures.
    """
    system = build_director_system_prompt()
    user = build_director_user_prompt(user_prompt, transcript, source_duration_seconds)
    raw = provider.complete(user, system=system, response_format="json")
    data = extract_json_object(raw)
    if isinstance(data, dict):
        data.setdefault("version", AI_PLAN_SCHEMA_VERSION)
        data.setdefault("provider", provider.name)
        data.setdefault("prompt", user_prompt)
        if data.get("source_duration_seconds") is None:
            data["source_duration_seconds"] = source_duration_seconds
    plan = parse_ai_plan(data)
    return normalize_ai_plan(plan)
