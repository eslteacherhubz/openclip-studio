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
    MockAIProvider,
    load_provider_config,
    parse_ai_plan,
    normalize_ai_plan,
    plan_to_review_items,
    ai_plan_to_director_decisions,
)

provider = MockAIProvider(load_provider_config("local_mock"))
raw = provider.complete("Make a 30-second highlight reel")

plan = parse_ai_plan(json.loads(raw))  # strict validation, raises AIPlanError
plan = normalize_ai_plan(plan)  # pure: sort, dedupe, clamp, assign IDs
items = plan_to_review_items(plan)  # every proposal becomes PENDING review item
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

## Real providers and the prompt-to-plan pipeline

`src/openclip/ai/clients.py` implements BYOK HTTP clients using only the
standard library (`urllib.request` + `json` — no new dependencies):

- `OpenAICompatibleClient` — works for **OpenAI** (`/v1/chat/completions`,
  `Authorization: Bearer …`) and **Mistral** (same API shape).
- `AnthropicClient` — Anthropic messages API (`/v1/messages`, `x-api-key` +
  `anthropic-version` headers, top-level `system` field).

Both take `base_url`, `api_key`, `model` and `timeout_seconds` from the
`ProviderConfig`, and raise `AIRuntimeError` on HTTP errors, timeouts,
invalid JSON responses or missing fields. `create_provider(config)` in
`src/openclip/ai/providers.py` instantiates the right provider class
(`local_mock` → `MockAIProvider`). Replicate and ElevenLabs remain stubs
until their milestone. Tests mock `urllib.request.urlopen` — **no test ever
makes a real network call**.

`generate_plan(provider, user_prompt, transcript, source_duration_seconds)`
in `src/openclip/ai/pipeline.py` runs the full pipeline:

1. `build_director_system_prompt()` (`src/openclip/ai/prompts.py`) commands
   the LLM to act as the "OpenClip AI Director" and output **only** JSON
   matching the `AIDirectorPlan` schema (a full example is embedded), with
   all timestamps within `source_duration_seconds`.
2. The user prompt combines the request, transcript and duration.
3. `provider.complete(..., response_format="json")` is called; OpenAI-style
   clients set `response_format: {"type": "json_object"}`.
4. `extract_json_object()` safely pulls the JSON out of the response —
   handling ` ```json … ``` ` fences, plain fences, raw JSON, and JSON
   embedded in prose — raising `AIPlanError` otherwise.
5. Missing `version`/`provider`/`prompt`/`source_duration_seconds` are filled
   from known values, then `parse_ai_plan` (strict) and `normalize_ai_plan`
   (pure) produce the final `AIDirectorPlan`.

## Director extension (safe)

`src/openclip/director/models.py` `DECISION_KINDS` now also includes the AI
kinds `clip`, `broll`, `zoom`, `audio_enhance`, `export_config`, and
`ai_plan_to_director_decisions()` maps every plan section to pending
`Decision`s. `check_invariants` accepts AI-origin overlays without a script
line (`params["origin"] == "ai"`).

The executor (`src/openclip/director/executor.py`) **safely skips** accepted
decisions of AI kinds that have no render primitive yet (`clip`, `broll`,
`export_config`) with a logged warning — while `cut`/`caption`/
`callout`/`section_title` rendering is 100% untouched, and the legacy
unimplemented kinds (`reframe`, …) still raise `NotImplementedError` as
before. No AI decision is ever auto-accepted.

## Render primitives (Phase 4)

`src/openclip/ai/render_primitives.py` turns accepted AI decisions into
deterministic FFmpeg filters, wired into the executor's final pass:

- **`zoom`** → `build_zoom_vf(span, w, h, strength, fps)` builds a `zoompan`
  punch-in: zoom factor `1/(1-strength)` (strength 0.5 → 2x, default 0.2 →
  1.25x) active only for the output frames inside the span (mapped to the
  export timeline through accepted cuts via `map_span`); output resolution is
  unchanged and frames outside the span are untouched. (FFmpeg's `crop`
  filter does not support timeline `enable`, hence `zoompan`.)
- **`audio_enhance`** → `audio_chain_for_enhancement(enabled, denoise,
  loudness_normalize)` maps the decision flags onto the existing
  `engine/enhance.py` presets (`afftdn` denoise, `loudnorm` normalization) and
  replaces the default audio chain when an accepted decision exists.

`render_export()` gained an optional `video_filters` parameter (default
`None` — existing callers are byte-identical).

## End-to-end CLI: `openclip ai-edit`

```bash
openclip ai-edit --media lesson.mp4 --prompt "Make a short highlight reel" \
    --transcript transcript.txt -o out.mp4                  # review JSON to stdout, no render
openclip ai-edit --media lesson.mp4 --prompt "..." -o out.mp4 \
    --provider local_mock --auto-accept                     # render the MP4
```

Flow: probe media → read transcript (optional) → `generate_plan()` →
review items + director decisions. Without `--auto-accept` the command
prints the plan plus PENDING review items as JSON and exits **without
rendering**; with `--auto-accept` every proposal becomes an accepted
`Decision` and `render_edl` renders the MP4 locally via FFmpeg. Real
providers need their API key in the environment (BYOK); tests use
`local_mock` only.

## B-roll and export config rendering (Phase 5)

- **`broll`** — `build_broll_filter(decision, base_resolution)` in
  `src/openclip/ai/render_primitives.py` builds the exact filter_complex
  segment: the asset (input N) is `scale`d to fit the base resolution with
  aspect ratio preserved, `pad`ded to full frame, and `overlay`ed with
  `enable='between(t,start,end)'` in export-timeline time — above the base
  video, under burned-in captions (which are applied earlier in the main
  chain). Image assets are looped for the span; video assets are read as-is.
  A missing/invalid `asset` path drops the decision with a logged warning —
  the render never crashes on it.
- **`export_config`** — `build_export_config_args(decision)` maps
  `width`/`height`/`fps`/`vertical` to a final-pass filter and output args:
  `vertical: true` center-crops to 9:16 (`crop=w=trunc(ih*9/16/2)*2:h=ih`)
  before `scale=WxH`; otherwise a plain `scale=WxH`; `fps` becomes `-r`.
- `render_export()` gained optional `broll` (list of `BrollOverlay`) and
  `out_fps` parameters — both default to `None`, so existing callers are
  byte-identical. Only `clip` remains in the executor's safe-skip list.

## GUI review integration

The PySide6 GUI (`src/openclip/gui/main_window.py`) has an **AI Director
proposals** panel: enter a prompt → **AI Plan…** runs `generate_plan()` on a
background worker (`AIPlanWorker`, provider `local_mock` offline / BYOK env
keys for real providers) → proposals appear as an Accept/Reject table (all
PENDING, checked = accept) → **Accept && Render AI…** marks checked rows
accepted, the rest rejected, and renders via `AIRenderWorker` →
`render_edl`. Nothing renders without explicit review. The panel has a
**provider selector** (`local_mock`, `openai`, `mistral`, `anthropic`,
`replicate`, `elevenlabs`, `ollama`) and a **password-masked BYOK API-key
field**; the key is passed in-memory to `AIPlanWorker` as an env override
for that call only — never logged or persisted (`provider_env_overrides` in
`src/openclip/gui/workers.py`). `ollama` runs locally
(`http://localhost:11434/v1`, OpenAI-compatible) and needs no key, like
`local_mock`.

## Transcript-based editing (edit video by editing text)

`src/openclip/ai/transcript_editing.py` implements
`transcript_to_edits(original, edited) -> list[Decision]`: blocks present in
the original timestamped transcript but missing from the edited one become
pending `cut` decisions at their original media timestamps. Matching is exact
on normalized text, then fuzzy (rapidfuzz ratio ≥ 90) so slightly reworded
blocks still count as kept; additions never create cuts; blocks without valid
timestamps are skipped with a logged warning; an empty original/edited
transcript logs a warning and never crashes.

CLI:

```bash
openclip ai-edit-transcript --media lesson.mp4 \
    --original original.json --transcript edited.json -o out.mp4 [--dry-run]
```

The edited transcript is the review: generated cuts are auto-accepted and
rendered. `--dry-run` prints the EDL JSON without rendering. If the edit
would delete every timestamped block (nothing left to render), the command
exits with code 2 and a clear message instead of crashing.
