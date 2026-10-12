# PRODUCTION_GAP_ASSESSMENT — implemented vs the owner's core requirements

Date: 2026-10-11 (correction pass). Evidence pointers refer to code and test
files on `correction/caption-mapping-and-packaging`. Statuses:
**Implemented** (working, verified by automated tests), **Partial** (real
functionality, documented limits), **Missing** (not started).

The honest headline: OpenClip Studio today is a **working bilingual
cut/caption/audio editor with research-grade eye-contact correction**. Five
of the owner's ten requirements are Missing or largely missing; animated
subtitles must not be mistaken for an educational motion-graphics system.

| # | Requirement | Status | Summary |
|---|---|---|---|
| 1 | Natural eye-contact correction | **Partial** | Working algorithm, rig-validated; not proven natural on real footage |
| 2 | Script-directed editing | **Implemented (core)** | Alignment→cuts→captions pipeline; no retake grouping/semantic planning |
| 3 | Dynamic presenter positioning | **Missing** | Nothing reframes/repositions the presenter |
| 4 | Professional background replacement | **Missing** | No segmentation, no compositing |
| 5 | Lighting and colour improvement | **Missing** | Audio has presets; video has none |
| 6 | Contextual educational animations | **Partial (captions only)** | Animated captions ≠ motion graphics |
| 7 | Intelligent timing of examples/diagrams | **Missing** | No overlay engine or timing planner |
| 8 | Bilingual English–Indonesian | **Partial** | Architecture is language-agnostic; no Indonesian-specific work |
| 9 | Commercial-quality MP4 output | **Partial** | Correct baseline (H.264/AAC/R128); missing color tags, HW encode, normalization |
| 10 | Minimal manual editing | **Partial** | One-command `auto` exists; no templates, batch, or timeline UX |

## Detailed assessments

### 1. Natural eye-contact correction — Partial (research-grade)
- Exists: warp-based iris redirection (`openclip/gaze/`), morphological iris
  detector, temporal smoothing, rig evaluation harness. Measured on the
  synthetic rig: 65.8% mean offset reduction, 99 dB PSNR outside the eye,
  147 fps CPU (`docs/EYE_CONTACT_RESEARCH.md`, `pytest -m gaze`).
- Ships in the Windows package and is smoke-verified there (this correction
  pass; frozen gaze smoke in CI).
- Not done: **no validation on real footage** (no camera in the dev sandbox;
  the owner checklist has not been run); **no blink gating** (blinks can be
  warped — visible artifact); frontal Haar face detection only (profiles,
  glasses, motion blur unknown); not integrated into the main edit pipeline
  or GUI (separate `gaze-apply` step); "natural" is a subjective claim that
  requires real-hardware testing.
- To close: owner runs the §5 checklist → tune on real footage → blink
  gating → MediaPipe FaceMesh backend → GUI integration.

### 2. Script-directed editing — Implemented (core)
- Exists: bilingual script parsing (`engine/script.py`), word-level ASR
  alignment with faster-whisper + fuzzy matcher (`engine/align.py`),
  cut planning from script divergence (off-script spans, fillers, silences
  with human-readable reasons, `engine/cutplan.py`), caption timing from the
  aligned script (`engine/captions.py`), review + export (`engine/render.py`,
  GUI cut table). Verified end-to-end in CI on both OSes, frozen app included.
- Not done: retake grouping (the same sentence spoken twice is not yet
  detected as "keep the best take"); per-script-line keep/reject UX (review
  is time-based); semantic planning (e.g., "skip the intro", "emphasize the
  target word") — that is the Production Director slice (see
  `docs/PRODUCTION_DIRECTOR.md`).

### 3. Dynamic presenter positioning — Missing
- No auto-framing, repositioning, pan/zoom, or multi-cam of any kind.
- Deterministic implementation path (next slice): face-track the presenter →
  smoothed crop window (EMA on center + clamped velocity) → ffmpeg
  crop/scale with fixed output size; decision surfaced as a reviewable
  `ReframeDecision`. Cost: one extra decode pass; CPU-feasible at 1080p.

### 4. Professional background replacement — Missing
- No person segmentation or compositing anywhere in the codebase.
- Deterministic path: per-frame person mask (MediaPipe selfie segmentation
  or rembg, both local, Apache-2.0/MIT) → ffmpeg overlay onto a
  solid/gradient/image background with edge feathering. Needs real-footage
  tuning; CPU cost is the main risk (one inference per frame).

### 5. Lighting and colour improvement — Missing
- Only audio has presets (`engine/enhance.py`); video passes through
  untouched.
- Deterministic path: `ColorPresetDecision` → ffmpeg `eq`/`colorbalance`/
  `curves` chains driven by face-region histogram statistics (auto-exposure +
  white balance), plus fixed "classroom" look presets. Cheap to build;
  parameters need real-footage tuning.

### 6. Contextual educational animations — Partial (captions only)
- Exists: animated bilingual captions (fade/pop/slide/karaoke) with
  alignment-driven timing, verified burned-in by PSNR test.
- **This is not a motion-graphics system.** There are no keyword callouts,
  concept highlights, lower-thirds, arrows/annotations, step reveals, or
  b-roll inserts. The Production Director architecture defines these as
  reusable compositions (`docs/PRODUCTION_DIRECTOR.md`); the first
  composition (KeywordCallout) lands with the director skeleton.

### 7. Intelligent timing of examples and diagrams — Missing
- No overlay engine, no asset slots, no planner that schedules visual aids
  from script content.
- Deterministic path: script markers (`Example:`, `Diagram: assets/x.png`)
  → `DiagramSlotDecision` with in/out times derived from the aligned line →
  ffmpeg overlay execution. Designed in the director doc; not implemented.

### 8. Bilingual English–Indonesian — Partial
- Exists: language-agnostic bilingual caption architecture (primary +
  support events, per-language fonts/sizes/colors), project JSON, alignment
  of the primary (spoken) line. Tests prove CJK rendering; Latin script
  (Indonesian) is simpler.
- Not done for Indonesian specifically: filler-word list for the cut planner
  is English-only ("eee", "emm" candidates — note "ya"/"jadi" are real words
  and must NOT be auto-cut; needs a curated list); GUI language defaults and
  docs; whisper `--language id` validation on real Indonesian audio; no
  Indonesian test fixture.

### 9. Commercial-quality MP4 output — Partial
- Exists: H.264 High + AAC-LC, yuv420p, `+faststart`, EBU R128 loudness
  normalization, CRF quality control, ffprobe-verified output in CI.
- Not done: explicit colour-space/transfer tags (bt709) for cross-player
  consistency; hardware-encoder option (NVidia QSV) for speed; input
  normalization stage (mixed resolutions/fps → single conform); two-pass
  loudnorm; chapters/metadata; 1080p throughput measured on the target
  hardware (never benchmarked there).

### 10. Minimal manual editing — Partial
- Exists: `openclip auto` = one command from project to export; GUI flow is
  import → paste script → Analyze → accept-all → Export; frozen Windows
  package with bundled features.
- Not done: batch/folder processing; reusable teacher profiles/templates;
  timeline-style review UX; undo; the review step is currently a gate the
  teacher must click through (by design, but it is manual).

## Cross-cutting gaps
- **Real-hardware performance is unmeasured** (dev box is a throttled ARM
  sandbox). 1080p render throughput, GUI playback, and gaze fps on the
  owner's PC are unknowns gated by `docs/OWNER_VALIDATION.md`.
- Single-camera talking-head assumption baked into the engine; multi-track
  editing is out of scope by design (see ROADMAP "Out of scope").
- Whisper accuracy on Indonesian-accented English / code-switching is
  untested.

## Recommended closure order (maps to the Production Director plan)
1. Colour/lighting presets + bt709 tags (cheap, high visible value).
2. Auto-framing (presenter positioning) — deterministic, no new deps.
3. Director compositions: KeywordCallout → DiagramSlot → SectionTitle
   (reuses overlay plumbing from #2's crop math).
4. Background replacement (needs segmentation dep + performance work).
5. Eye-contact: owner validation → blink gating → GUI integration.
6. Indonesian tuning: filler list, language defaults, real-audio validation.
7. Encoder hardening: hardware encode, two-pass loudnorm, metadata.
