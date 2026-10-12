# PRODUCTION_DIRECTOR — architecture for automated educational video assembly

Status: architecture defined + v1 skeleton landed (`src/openclip/director/`,
tested). This is the implementation-ready plan for the next development
slice. It replaces "vibes-based automation" with **structured, inspectable
editing decisions executed by deterministic render components**.

## 1. Problem

The owner needs minimal-manual-edit production of commercial-quality
bilingual lesson videos. The current engine can cut and caption, but every
"smart" behaviour beyond that (callouts, diagrams, colour, framing,
backgrounds) would either stay manual or be delegated to an opaque
generative model. Neither is acceptable:

- Manual does not scale to a teacher's weekly output.
- Uncontrolled generative video effects are non-deterministic, unverifiable,
  can alter educational meaning, and cannot be reviewed meaningfully.

## 2. Core concept: a decision list, not a black box

```
script + recording
   │
   ▼
ANALYZE        existing engine: alignment, silences, transcript, (opt) gaze stats
   │
   ▼
DIRECTOR PLAN  deterministic rule planner (+ optional schema-validated LLM advisor)
   │           output: EditDecisionList (EDL) — versionable JSON
   ▼
REVIEW         human accepts / rejects / edits each decision (GUI list or JSON diff)
   │
   ▼
EXECUTE        deterministic renderer: engine primitives execute accepted decisions
   │
   ▼
EXPORT MP4     existing H.264/AAC pipeline
```

**The Production Director is a planner, not a renderer.** It converts a
teaching script plus recording analysis into an `EditDecisionList`; the
deterministic engine executes it. Same inputs → same EDL → same video.

## 3. Non-negotiable invariants (enforced in code and tests)

1. **Meaning preservation.** The director never reorders, re-voices, or
   synthesizes spoken content. Removals are limited to what the script-
   divergence/silence/filler analysis already identified. Additions are
   overlays only (captions, callouts, titles, diagram slots).
2. **Inspectability.** Every decision is a typed record with: stable `id`,
   `kind`, `source_line` (the script sentence that justifies it),
   `source_span` (timeline range), human-readable `rationale`, `params`,
   and a review `status` (pending/accepted/rejected).
3. **Review before export.** Export executes only `accepted` decisions;
   `pending` blocks export unless the user explicitly accepts all.
4. **Determinism.** Same project state → byte-identical EDL (no timestamps,
   no dict-order dependence, no randomness).
5. **Reusable compositions over generative effects.** Each visual treatment
   is a parameterized, deterministic composition (a function from params +
   aligned timing → FFmpeg filter/ASS output). No free-form generation.

## 4. Decision catalogue (v1 → v2)

| Kind | v | Source | Executes as |
|---|---|---|---|
| `cut` | v1 | existing cut plan (script divergence, silences, fillers) | segment render + concat (exists) |
| `caption` | v1 | aligned script lines | animated bilingual ASS (exists) |
| `callout` | v1 | script markers `Keyword: term` / `Example:` | ASS overlay event, layer 2, safe area (new, landed) |
| `section_title` | v1 | script markers `Section: ...` / `# ...` | ASS title event (new, landed) |
| `diagram_slot` | v2 | script marker `Diagram: assets/x.png` | ffmpeg overlay of a reviewed asset during the aligned line |
| `color_preset` | v2 | shot statistics | ffmpeg `eq`/`colorbalance`/`curves` chain + bt709 tags |
| `reframe` | v2 | face-track analysis | smoothed crop window (presenter positioning) |
| `background` | v3 | person segmentation | overlay onto solid/gradient/image |
| `emphasis` | v2 | teacher-marked terms | karaoke-highlight of target vocabulary in captions |

Script markers are a documented convention (deterministic, diff-able in git,
no hidden state): a line `Keyword: photosynthesis` in the teaching script is
itself the review artifact.

### Why markers and not an LLM?
Markers make the teacher's intent explicit and version-controlled. An LLM
advisor (optional, later, default off) may *propose* decisions, but they
enter the EDL as ordinary records with rationale + source line, pass the
same review, and are schema-validated before planning. No model sits in the
render path.

## 5. Safe-area and overlay rules (v1, enforced)

- Overlays render in the lower 25% or upper 20% of the frame by default
  (documented margins), never centred on the presenter's face region.
- Every composition has a maximum concurrent-occupancy rule (v1: one
  callout OR one title at a time; conflicts resolved deterministically by
  earliest `source_line`).
- All timings derive from the aligned script line ± documented padding, so
  overlays never float free of the narration.

## 6. Execution mapping (deterministic components)

| EDL kind | FFmpeg/libass primitive |
|---|---|
| cut | per-segment re-encode + concat demuxer (`engine/render.py`) |
| caption / callout / section_title / emphasis | ASS events (libass `ass` filter) — layered, animated, deterministic |
| diagram_slot | `overlay` filter with `enable='between(t,in,out)'` |
| color_preset | `-vf` chain + `-colorspace bt709 -color_primaries bt709 -color_trc bt709` |
| reframe | animated `crop` (linear-interpolated, clamped) + fixed output scale |
| background | masked `overlay` (segmentation mask as alpha) |

Every primitive is parameterized, testable, and reversible (re-run produces
identical output). The executor refuses kinds it cannot execute (explicit
`NotImplementedError` with the missing component named) — no silent
degradation.

## 7. Landed in this correction pass (v1 skeleton)

- `director/models.py` — typed decision records + `EditDecisionList` with
  JSON round-trip, determinism (stable ordering + no timestamps), and the
  invariant checks above.
- `director/planner.py` — `RuleBasedPlanner`: script markers + existing
  alignment/cut plan → EDL. Pure function of project state. Markers are
  detected by the engine (`engine/script.py`) and skipped by the aligner, so
  they never consume ASR words; each attaches to the next aligned spoken line.
- `director/executor.py` — executes accepted `cut`/`caption`/`callout`/
  `section_title` decisions through the engine renderer:
  - every overlay span is mapped through `map_span` against the normalized
    accepted cuts, so overlays land at **export-timeline** timestamps;
    overlays fully inside cuts are dropped;
  - captions render only for **accepted** `caption` decisions, and only
    when `RenderOptions.burn_captions` is set (callouts/titles are governed
    by their own decision status);
  - unimplemented kinds (diagram_slot, color_preset, reframe, background,
    emphasis) fail with an explicit `NotImplementedError` — no silent
    degradation.
- `engine/render.py` — public `normalize_cut_spans` (merge overlaps/
  duplicates/adjacency) used by `map_span`, `render_project`, and the
  executor, so caption/overlay mapping and the cut renderer share one
  normalized timeline; public `build_export_ass`/`render_export` path.
- `engine/captions.py` — split into `build_ass_header` (+extra styles) and
  `caption_events` so the director composes valid ASS documents (including
  overlay-only documents) without duplicating style logic.
- Tests: determinism, round-trip, meaning-preservation, review gating
  (accepted/rejected/pending/mixed), export-timeline mapping (ASS timestamps),
  and a **pixel-verified video-render regression** that probes exported
  frames for overlays placed before/during/after cuts plus a fully removed
  overlay and raw-timestamp sentinels (negative-controlled against the bug).

## 8. Next slices (in recommended order)

1. **Colour/lighting** (`color_preset` v2): face-region histogram →
   exposure/WB params; bt709 tagging; "classroom" presets. No new deps.
2. **Auto-framing** (`reframe` v2): face-track → smoothed crop; reuses gaze
   detector work; no new deps.
3. **Diagram slots** (`diagram_slot` v2): overlay filter + asset review in
   GUI; timing from aligned lines.
4. **GUI review view for the EDL** (list + timeline lanes + per-decision
   accept/reject/params) — replaces the cut-only table.
5. **Indonesian tuning** (filler list, language defaults, real-audio
   validation) + **encoder hardening** (HW encode option, two-pass loudnorm).
6. **Background replacement** (v3, needs segmentation dependency + CPU
   budget measurement on target hardware).
7. Optional LLM advisor (schema-validated, off by default, plan-time only).
