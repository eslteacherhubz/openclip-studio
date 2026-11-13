# AI Integration (BYOK)

OpenClip Studio's AI layer lets users bring their own API keys (BYOK) so that
AI-assisted editing features — proposed cuts, clips, captions, zooms, B-roll
and export settings — can run against OpenAI, Mistral, Anthropic, Replicate or
ElevenLabs, while rendering stays local via FFmpeg.

This document covers the first foundation of that layer: secure provider
configuration and an offline mock provider used by tests.

## Purpose of the AI layer

- Accept a video plus a natural-language prompt and ask an LLM for an editing
  plan (clips, cuts, captions, overlays, export settings).
- Keep every API key under the user's control (BYOK); nothing is sent anywhere
  unless the user configures a provider and invokes an AI feature.
- Keep the app testable: all automated tests use the `local_mock` provider and
  never make network calls or require real API keys.

## Supported providers

| Provider     | API-key environment variable | Notes                          |
| ------------ | ---------------------------- | ------------------------------ |
| `openai`     | `OPENAI_API_KEY`             | chat completions (stub)        |
| `mistral`    | `MISTRAL_API_KEY`            | chat completions (stub)        |
| `anthropic`  | `ANTHROPIC_API_KEY`          | messages API (stub)            |
| `replicate`  | `REPLICATE_API_TOKEN`        | hosted models (stub)           |
| `elevenlabs` | `ELEVENLABS_API_KEY`         | voice/audio APIs (stub)        |
| `local_mock` | none required                | deterministic offline mock     |

Real provider classes currently raise `NotImplementedError` with a clear
message; they land in a later milestone. `local_mock` is fully functional.

## Environment variables

Provider-specific key variables are listed above. Generic fallbacks apply to
any provider:

- `OPENCLIP_AI_API_KEY` — fallback API key
- `OPENCLIP_AI_BASE_URL` — override the provider base URL
- `OPENCLIP_AI_MODEL` — override the model name
- `OPENCLIP_AI_TIMEOUT_SECONDS` — request timeout in seconds (default `30`)

Environment variables override values from the local config file.

## Local config file

`load_provider_config(provider, env=None, config_path=None)` also reads a local
JSON file (default locations used by the app: `.openclip/ai.json` or
`.openclip/ai.local.json`):

```json
{
  "provider": "mistral",
  "model": "mistral-large-latest",
  "base_url": "https://api.mistral.ai/v1",
  "timeout_seconds": 60
}
```

- The config file must **not** contain the API key; keys come only from
  environment variables.
- If the file declares a different `provider` than requested, loading fails.
- Missing key for a real provider raises `ProviderConfigError` with a helpful
  message. `local_mock` never requires a key.

## Mock provider usage

```python
from openclip.ai import MockAIProvider, load_provider_config

config = load_provider_config("local_mock")
provider = MockAIProvider(config)
plan_json = provider.complete("Make a 30-second highlight reel")
```

`complete()` returns a deterministic JSON string (an editing plan with
`version`, `provider`, `prompt`, `clips`, `cuts`, `captions`) — stable enough
for snapshot-style tests. It performs no network I/O and needs no key.

## Security rules

- **Never commit secrets.** API keys, tokens and credentials must never appear
  in code, logs, tests or commits. `.openclip/ai.json` and
  `.openclip/ai.local.json` are git-ignored; only environment variables carry
  keys.
- `ProviderConfig.__str__`/`__repr__` mask the API key (`********`), so configs
  are safe to log.
- The config loader never makes network calls.
- Tests must not require real keys or network access.

## AI Director EDL: parse → normalize → review

Every AI proposal (mock today, real providers later) is a JSON document that
must pass through the typed EDL schema in `src/openclip/ai/edl.py` **before**
anything is reviewed or rendered. See `docs/AI_DIRECTOR_EDL.md` for the full
schema.

```python
import json
from openclip.ai import (
    MockAIProvider, load_provider_config,
    parse_ai_plan, normalize_ai_plan, plan_to_review_items,
    ai_plan_to_director_decisions,
)

provider = MockAIProvider(load_provider_config("local_mock"))
raw = provider.complete("Make a 30-second highlight reel")

plan = parse_ai_plan(json.loads(raw))      # strict validation, raises AIPlanError
plan = normalize_ai_plan(plan)             # pure: sort, dedupe, clamp, assign IDs
items = plan_to_review_items(plan)         # every proposal becomes PENDING review item
decisions = ai_plan_to_director_decisions(plan)  # cut/caption → pending director Decisions
```

- `parse_ai_plan(data)` validates required fields (`version`, `provider`,
  `prompt`), timing sanity (`end > start`, finite, non-negative), and ranges
  (`score`/`confidence`/`strength` in `[0, 1]`); unknown fields are kept in
  `extra`. Invalid input raises `AIPlanError`.
- `normalize_ai_plan(plan)` is pure and deterministic: sorts items by start
  time, removes exact duplicate spans, clamps items to
  `source_duration_seconds` (dropping items fully beyond it), and generates
  deterministic IDs for items that lack one.
- `plan_to_review_items(plan)` turns clips/cuts/captions/broll/zooms into
  `AIReviewItem`s, all `ReviewStatus.PENDING` — nothing is auto-accepted.
- `ai_plan_to_director_decisions(plan)` bridges cuts/captions into the
  existing Production Director `Decision` records (status `pending`, reason
  preserved in `rationale`); other kinds are review-only stubs for now.

**Rule: real provider output must be normalized through this schema before
rendering.** No model output reaches FFmpeg except as reviewed, accepted
decisions. Local-first rendering via FFmpeg remains unchanged.
