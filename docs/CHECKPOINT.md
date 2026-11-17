# CHECKPOINT — exact, resumable progress state

Last updated: 2026-11-16 — Phase 10 Objective 4 (Beauty Engine Integration) complete
and CI-verified; Priority 3 (video preview player) complete and CI-verified.

## Current state

The feature chain `feature/ai-face-analysis-tracking` is 43 commits ahead of `main`
(merge base `d71452f` Stage 0 bootstrap). `main` carries only 2 CI-infra commits
the feature chain lacks (`3317ad9` register workflow, `a608f2a` fix workflow
syntax). All Phase 8–10 work plus Priority 3 are landed on this branch, not yet
merged to `main`. See docs/MERGE_PLAN.md for the merge analysis.

### Phases complete (this branch)

| Phase | Objective | Where | State |
|---|---|---|---|
| 8 | Smart reframing — subject-tracking 9:16 crop | `src/openclip/ai/reframing.py`, `gui/reframe_preview.py` | landed (f53ad67) |
| 9 | Shared face-analysis tracking layer | `src/openclip/ai/face_analysis.py` (Mock + stubbed MediaPipe providers, CachedFaceAnalysis, smoothing) | landed (d04abb7) |
| 10 Obj 1 | Beauty engine core — frequency-separation skin smoothing | `src/openclip/ai/beauty.py` (`smooth_skin`) | landed (a9cc0b3) |
| 10 Obj 2 | Beauty engine — micro-sculpt + spot-heal | `src/openclip/ai/beauty.py` (`micro_sculpt`, `spot_heal`) | landed (in 46ff86f) |
| 10 Obj 3 | BeautyConfig + real-provider pipeline | `src/openclip/ai/beauty.py` (`BeautyConfig`, `CachedFaceAnalysis` plumbing) | landed (f8ae61f, 46ff86f) |
| 10 Obj 4 | **Beauty Engine Integration** — render pipeline + GUI + CLI | `engine/render.py` (`RenderOptions.beauty`), `director/executor.py` (pre-pass), `cli.py` (`--beauty`/`--beauty-smooth/jaw/neck`), `gui/main_window.py` + `gui/workers.py` (checkbox + sliders), `gui/preview.py` | landed (46ff86f, ed2b3c1, 2d4baeb) |
| Priority 3 | Real video preview player (CORE) | `src/openclip/gui/preview.py` (`VideoPreviewWidget`: load/seek/play-pause, cv2-guarded placeholder, release-on-close) | landed (2d4baeb) |

Byte-identical guarantee: when `RenderOptions.beauty is None` the render path is
completely unchanged — `director/executor.py` short-circuits and passes the
original `media` through untouched; no existing render logic was modified.

### PR / branch map (#7–#14)

| PR | Branch | Content | State |
|---|---|---|---|
| #7 | feature/ai-transcript-clip-byok | transcript-based editing, ai-edit-transcript CLI, BYOK GUI selector | landed on chain |
| #8 | feature/ai-provider-foundation | BYOK provider foundation + mock provider | landed on chain |
| #9 | feature/ai-render-primitives-cli | render primitives (zoom/audio) + end-to-end CLI | landed on chain |
| #10 | feature/ai-gui-transcript-captions | transcript editing panel + customizable caption styles + GUI review | landed on chain |
| #11 | feature/ai-edl-schema | AI director EDL schema + review bridge | landed on chain |
| #12 | feature/ai-face-analysis-tracking | smart reframing + shared face-analysis + beauty engine + preview | **this branch (HEAD 2d4baeb)** |
| #13 | feature/ai-broll-export-gui | B-roll / export-config rendering + GUI review integration | not yet merged |
| #14 | feature/ai-customization-intelligence | customization intelligence layer | not yet merged |

Chain order (bottom → top): #7 → #8 → #9 → #10 → #11 → #12 → #13 → #14.

## Verification record

- Local (dev sandbox): **377 passed**, ruff clean, `mypy --strict` clean (47 source files).
  - `pytest -q -m gui`: 14 passed (offscreen PySide6)
  - `pytest -q -m gaze`: 28 passed (cv2 present; real-clip decode/seek/play-pause)
- CI: green for HEAD `2d4baeb` — run 43865333081 (lint, mypy, test 3.11+3.12,
  gui offscreen, gaze, test-windows, package-windows).
- Prior commits `7b38f6d`, `e519310`, `ed2b3c1`, `46ff86f` also green.

## Environment (dev sandbox)

- Linux aarch64 (Debian 12), 8 vCPU (shared/throttled), 7.5 GB RAM, no GPU, no camera.
- Python 3.11.2 (venv `.venv/`), FFmpeg 5.1.9 (apt) + imageio-ffmpeg wheel,
  opencv-python 4.14.0 (pinned <5), PySide6 6.8.0.2 + system Qt libs
  (libegl1 libgl1 libxkbcommon0 libglib2.0-0 libdbus-1-3 libfontconfig1 libpulse0).
- `gh` authenticated as `eslteacherhubz` (repo + workflow scopes).
- Windows not available here: covered by CI (`windows-latest` engine tests +
  packaging smoke).

## Blocked

- Nothing technical. Awaiting owner review/merge of the stacked chain #7→#14.
  See docs/MERGE_PLAN.md for the recommended merge order and conflict analysis.

## Next steps (in order)

1. Owner: review + merge PRs #7→#14 in chain order (see docs/MERGE_PLAN.md).
2. After merge: Phase 11 — real MediaPipe provider behind
   `MediaPipeFaceAnalysisProvider` (currently a stub) + real-footage validation
   test (beauty_off byte-identical on real footage; beauty_on visual parity).
3. Phase 12: hook / viral-moment detection.
4. Phase 13: contextual B-roll sourcing.
5. Phase 14: platform export presets.
6. Phase 15: quality/perf (incl. high-quality beauty intermediate).
7. Phase 16: beta hardening.

## Resume protocol for a new session

1. Read this file top to bottom.
2. `git status` + `git log --oneline -10` + `gh pr list` + `gh run list`.
3. Continue from the first unfinished item in "Next steps".
4. Obey AGENTS.md. Update this file before stopping.
