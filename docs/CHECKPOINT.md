# CHECKPOINT — exact, resumable progress state

Last updated: 2026-10-09 (Stage 1 complete on branch; PR pending)

## Environment (dev sandbox where this work runs)

- Linux aarch64 (Debian 12), 8 vCPU, 7.5 GB RAM, no GPU, no camera.
- **CPU availability is highly variable** (shared box): identical ffmpeg runs
  measured 7s–290s. Test media was sized down (320x180@15fps) accordingly;
  treat slow renders as environment noise, verify with `time`.
- Python 3.11.2 (venv at `.venv/`), FFmpeg 5.1.9 (apt) + imageio-ffmpeg wheel
  binary (7.0.2), espeak-ng 1.51 (for ASR test fixtures), internet available.
- faster-whisper installed in venv; whisper `tiny` model downloaded to local
  cache (~/.cache) — ASR tests run fully offline after that.
- `gh` authenticated as `eslteacherhubz` (repo + workflow scopes).
- Windows **not** available here; Windows correctness covered by CI
  (`windows-latest`) and, for real validation, by the owner.

## Done

- **Stage 0** (main, commit d71452f): governing docs, LICENSE (GPL-3.0),
  editor verification research, package scaffold.
- **Stage 1** (branch `feature/engine-core`, commits up to "Stage 1: engine
  core with alignment, cuts, captions, enhancement, render, CLI"):
  - `engine/ffmpeg.py`: ffprobe JSON + `ffmpeg -i` fallback probing,
    silencedetect parsing, safe filter-path escaping, ffmpeg locator
    (env → PATH → imageio-ffmpeg wheel).
  - `engine/synth.py`: deterministic synthetic bilingual lesson (contains a
    flub + long silence for cut planner validation) + mock word hypotheses.
  - `engine/script.py`, `engine/align.py`: bilingual script parsing;
    monotonic fuzzy aligner (rapidfuzz `token_sort_ratio` — `token_set_ratio`
    is subset-lenient and misbehaves), faster-whisper backend, even fallback.
  - `engine/cutplan.py`: silence/filler/off-script cuts with reasons,
    tiny-island absorption, kept-interval computation.
  - `engine/captions.py`: animated bilingual ASS (fade/pop/slide/karaoke).
  - `engine/enhance.py`: FFmpeg audio presets (teaching/denoise/normalize).
  - `engine/render.py`: segment-render + concat + caption/audio final pass,
    source→export time mapping for captions.
  - `engine/project.py`, `engine/pipeline.py`, `cli.py` (probe/synth/new/
    align/plan/render/auto), `engine/jsonio.py` typed JSON boundary.
  - Tests: 19 engine tests green (unit + real-render integration + PSNR
    caption-burn check + CLI E2E); 2 ASR tests green locally (real whisper on
    espeak speech). mypy --strict and ruff clean.

## In flight

- Stage 1 PR `feature/engine-core` → `main` (about to open).

## Blocked

- Nothing. Eye-contact (Stage 3) has known sandbox limits (no camera/GPU);
  plan is synthetic-rig validation + owner checklist — not blocked per
  master prompt rules.

## Next steps (in order)

1. Open PR for `feature/engine-core`; wait for owner merge (do NOT self-merge).
2. Stage 2: branch `feature/desktop-app` — PySide6 GUI (project window, script
   editor, cut review, caption settings, export w/ progress, offscreen-safe
   tests). Rebase on main if Stage 1 merged meanwhile.
3. Stage 3: branch `feature/eye-contact` — synthetic eye rig, OpenCV
   detector, warp-based redirection, experiments →
   `docs/EYE_CONTACT_RESEARCH.md`, `openclip gaze-*` CLI behind `gaze` extra.
4. Stage 4: branch `feature/ci-packaging` — Actions matrix, mypy gate on,
   headless E2E job, PyInstaller Windows artifact.

## Key technical notes for resuming sessions

- `probe()` must read ffprobe JSON key `codec_type` (not `type`).
- Whisper fixture: espeak-ng makes speech offline; sine tones will NOT work
  as ASR input.
- The synth lesson is 320x180@15fps CRF24 (perf: this box). Do not "fix" the
  small size — tests assert it.
- Longest silence in synth plan is the 1.8s tail, not the 1.6s mid gap.
- `run_ffmpeg` prepends `-hide_banner -nostdin`; keep that (stdin hangs).

## Resume protocol for a new session

1. Read this file top to bottom.
2. `git status` + `git log --oneline -10` + `gh pr list` to see actual state.
3. Continue from the first unfinished item in "Next steps".
4. Obey AGENTS.md. Update this file before stopping.
