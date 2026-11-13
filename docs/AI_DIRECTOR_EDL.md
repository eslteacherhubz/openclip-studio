# AI Director EDL — schema for AI-proposed editing plans

Status: schema + parser + normalizer + review bridge landed
(`src/openclip/ai/edl.py`, `src/openclip/ai/review.py`,
`src/openclip/ai/director_bridge.py`), tested. No provider yet emits real
plans; `local_mock` output conforms to this schema.

## What it is and what problem it solves

An **AI Director EDL** (Edit Decision List) is the typed, validated,
JSON-serializable form of an AI editing proposal: clips to keep, cuts to
remove, captions, B-roll inserts, zooms, audio enhancement and export
settings, all derived from a natural-language prompt.

The problem it solves: raw LLM output is free-form text. Before anything can
be reviewed or rendered, it must become a **structured, validated, inspectable
plan** — otherwise an AI proposal could contain impossible timings
(`end <= start`, negative or infinite times), out-of-range scores, or unsafe
instructions that a renderer would act on silently. This schema is the gate:
every AI proposal (mock today, real providers later) must pass
`parse_ai_plan` → `normalize_ai_plan` → review **before** any rendering.

## JSON structure

Top level (`AIDirectorPlan`):

| Field | Required | Type | Notes |
|---|---|---|---|
| `version` | yes | int | must equal `1` (`AI_PLAN_SCHEMA_VERSION`) |
| `provider` | yes | non-empty string | e.g. `local_mock`, `openai` |
| `prompt` | yes | string | the original natural-language prompt |
| `source_duration_seconds` | no | finite number ≥ 0 | used for clamping during normalization |
| `clips` | no | array of clip | see below |
| `cuts` | no | array of cut | |
| `captions` | no | array of caption | |
| `broll` | no | array of b-roll | |
| `zooms` | no | array of zoom | |
| `audio` | no | object | audio enhancement settings |
| `export` | no | object | export settings |
| `notes` | no | array of string | free-text AI notes for the reviewer |
| `extra` | no | object | unknown fields are preserved here |

Every timed item (`clips`, `cuts`, `captions`, `broll`, `zooms`) carries a
span, accepted either as a nested object or flat keys (both parse):

```json
{"span": {"start_seconds": 0.0, "end_seconds": 10.0}}
{"start_seconds": 0.0, "end_seconds": 10.0}
```

### Item fields

- `AISpan`: `start_seconds`, `end_seconds` (floats).
- `AIClip`: `id?`, `title`, `span`, `reason`, `score?`, `extra?`
- `AICut`: `id?`, `span`, `reason`, `confidence?`, `extra?`
- `AICaption`: `id?`, `span`, `primary_text`, `support_text?`, `style?`, `extra?`
- `AIBRoll`: `id?`, `span`, `query`, `source?`, `mode?`, `extra?`
- `AIZoom`: `id?`, `span`, `mode?`, `strength?`, `extra?`
- `AIAudioEnhancement`: `enabled`, `denoise?`, `loudness_normalize?`, `extra?`
- `AIExport`: `format?`, `width?`, `height?`, `fps?`, `vertical?`, `extra?`

`extra` on any item preserves unknown fields from the source JSON.

## Validation rules (`parse_ai_plan`)

All violations raise `AIPlanError` with a message naming the offending path:

- input must be a JSON object; `version`, `provider`, `prompt` are required
  (`version` must be the supported schema version);
- all times must be finite numbers, **non-negative**, and
  `end_seconds > start_seconds` (zero-length spans are rejected);
- `score`, `confidence`, `strength` must lie in `[0, 1]` when present;
- `source_duration_seconds` must be finite and ≥ 0;
- `export.width`/`height` must be integers, `fps` a positive number,
  `vertical` a boolean;
- lists must be arrays, `notes` entries strings, item fields correctly typed;
- unknown fields never fail parsing — they are preserved in `extra`.

The parser is tolerant of minor shape variations (nested `span` vs flat
timing keys) but fails clearly on unsafe or ambiguous timing data. It never
touches the network or disk.

## Normalization (`normalize_ai_plan`)

Pure and deterministic (no side effects, no IO; the input plan is not
mutated). For each of clips/cuts/captions/broll/zooms:

1. **Duration rule** — if `source_duration_seconds` is known, items starting
   at/after the duration are **removed**; items extending past it are
   **clamped** to `end_seconds == source_duration_seconds`.
2. **Exact duplicates** — items with an identical span within the same list
   are deduplicated (first occurrence wins). Overlapping but non-identical
   spans are **kept**; merging them is a human review decision, not an
   automatic one.
3. **Sorting** — items are sorted by `(start_seconds, id)`.
4. **ID generation** — items without an `id` get a deterministic
   `"<kind>-<NNN>"` id from their sorted position.

## Review status model (`src/openclip/ai/review.py`)

`plan_to_review_items(plan)` converts every clip, cut, caption, B-roll and
zoom into an `AIReviewItem`:

```python
class ReviewStatus(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
```

Each item has `id`, `kind` (`clip`/`cut`/`caption`/`broll`/`zoom`), `title`,
`reason`, `start_seconds`, `end_seconds`, `status` (always starts
`PENDING`), and a `payload` dict that round-trips the original plan item.
Output is deterministic (sorted by start, kind, id). Nothing is ever
auto-accepted.

## Relation to the existing director module

`src/openclip/director/` already defines the Production Director EDL
(`Decision`, `EditDecisionList`, statuses pending/accepted/rejected, review
gating before render). `src/openclip/ai/director_bridge.py` adapts AI plans
into that world:

- `ai_plan_to_director_decisions(plan) -> list[Decision]` maps AI `cut` items
  to `kind="cut"` and `caption` items to `kind="caption"` decisions, with
  `status="pending"`, the AI reason preserved in `rationale`, and AI metadata
  in `params` (`origin: "ai"`, provider, confidence, …). Rendering behavior
  is unchanged: the engine still executes only human-accepted decisions.
- **Stub contract:** `clip`, `broll`, `zoom`, `audio` and `export` have no
  v1 `DECISION_KINDS` equivalent, so the bridge intentionally skips them for
  now; they remain reviewable via `plan_to_review_items`. Mapping them to new
  decision kinds is a future milestone.

## Example payload

See `tests/fixtures/ai/mock_plan.json` — a complete realistic plan
(version, provider, prompt, source duration, clips, cuts, captions, broll,
zooms, audio, export, notes). Parse it with:

```python
import json
from openclip.ai import parse_ai_plan, normalize_ai_plan, plan_to_review_items

plan = normalize_ai_plan(parse_ai_plan(json.loads(open("tests/fixtures/ai/mock_plan.json").read())))
items = plan_to_review_items(plan)
```

## Current limitations

- Schema version is fixed at `1`; future fields must land in `extra` or a
  versioned schema bump.
- Normalization does not merge overlapping spans and does not resolve
  conflicts (e.g. two overlays at once) — that is review/GUI work.
- Only cuts and captions bridge into `director.Decision`; other kinds are
  review items only.
- Real providers still raise `NotImplementedError`; only `local_mock` emits
  conforming plans today.
- No rendering path consumes AI plans yet — nothing is executed without
  human acceptance.
