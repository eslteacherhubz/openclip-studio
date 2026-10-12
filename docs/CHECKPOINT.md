# CHECKPOINT — exact, resumable progress state

Last updated: 2026-10-09 (Stage 3 complete on branch; PRs #1 #2 pending, #3 about to open)

## Environment (dev sandbox where this work runs)

- Linux aarch64 (Debian 12), 8 vCPU (shared/throttled — identical ffmpeg runs
  measured 7s–290s), 7.5 GB RAM, no GPU, no camera.
- Python 3.11.2 (venv `.venv/`), FFmpeg 5.1.9 (apt) + imageio-ffmpeg wheel,
  espeak-ng 1.51 (ASR test fixtures), opencv-python 4.14 (pinned <5: OpenCV
  5.0 removed CascadeClassifier), PySide6 6.8.0.2 + system Qt libs,
  faster-whisper (tiny model cached) — internet available.
- GUI tests run offscreen (`QT_QPA_PLATFORM=offscreen`); QMediaPlayer has no
  backend in this minimal sandbox (warnings only; works on real desktops).
- `gh` authenticated as `eslteacherhubz` (repo + workflow scopes).
- Windows **not** available here; covered by CI (`windows-latest`, Stage 4)
  and by owner validation on the real machine.

## Done

- **Stage 0** (main d71452f): governing docs, LICENSE GPL-3.0, editor
  verification research (docs/research/EDITOR_VERIFICATION.md), scaffold.
- **Stage 1** (PR #1 `feature/engine-core`, CI green): headless engine —
  ffprobe(+fallback) probing, silencedetect, deterministic synthetic lesson
  (planted flub + long silence), bilingual script parsing, monotonic fuzzy
  aligner (token_sort_ratio) + faster-whisper + even fallback, cut planner
  (silence/filler/off-script, reasons, tiny-island absorption), animated
  bilingual ASS (fade/pop/slide/karaoke), FFmpeg audio presets, render
  (segment+concat+final pass, source→export caption time mapping), project
  JSON, CLI probe/synth/new/align/plan/render/auto.
- **CI plumbing**: workflow registered on main (infra-only 3317ad9, decision
  D10 — GitHub only registers workflows from the default branch). Pitfalls
  fixed: unquoted colon in step names breaks Actions YAML; validate with
  actionlint (binary at /tmp/opencode/actionlint).
- **Stage 2** (PR #2 `feature/desktop-app`, stacked on #1): PySide6 GUI —
  script editor, cut-review table (keep/cut checkboxes, double-click seek),
  caption/audio/transcriber settings, threaded workers + progress bar,
  preview player, `openclip-gui`/`openclip gui` entry points, 4 offscreen
  tests + CI `gui` job.
- **Stage 3** (branch `feature/eye-contact`, PR next): gaze research package
  (`gaze` extra) — synthetic eye rig with ground truth, morphological dark-
  blob iris detector (aperture bbox search; close→erode→dilate + aspect
  filters; E1: 1.36px mean err, 8.3% miss at extremes), warp-based iris
  redirection with tight cosine falloff (pivotal experiment E5a: reach 2.3
  → 15.8% reduction [whole sclera slides]; reach 1.35 → 92.8%; E2 grid:
  14.6%→6.4% = 65.8% mean reduction; PSNR outside eye 99dB; E4: 147 fps
  CPU), TemporalSmoother, video apply/analyze with audio remux, CLI
  gaze-analyze/gaze-apply, 7 gaze tests, full experiment record in
  docs/EYE_CONTACT_RESEARCH.md incl. owner validation checklist. No
  real-footage claims made anywhere.
- All gates green at each stage: ruff, mypy --strict (24 files), full pytest
  suite (33 tests: engine unit/integration, ASR (skippable), GUI, gaze).

## In flight

- PR #1 (Stage 1), PR #2 (Stage 2) — awaiting owner review/merge.
- Stage 3 PR about to open (base main, stacked).

## Blocked

- Nothing. Real-footage eye-contact validation is deferred to the owner by
  design (checklist in docs/EYE_CONTACT_RESEARCH.md §5), not blocked.

## Next steps (in order)

1. Open PR for `feature/eye-contact`; dispatch a CI run for it
   (`gh workflow run ci.yml --ref feature/eye-contact`) since PR-event CI
   has been unreliable in this repo.
2. Stage 4: branch `feature/ci-packaging` — expand CI matrix to
   windows-latest (pytest incl. gui import test), add PyInstaller one-folder
   Windows build job with SHA-256 artifact, final dependency-ledger pass,
   keep mypy strict gate.
3. Stage 5: hardening — README quickstart with the 5-step GUI path, final
   CHECKPOINT, owner validation checklist consolidated; tag v0.1.0-rc1 only
   if owner merges and all gates pass.

## Key technical notes for resuming sessions

- `probe()` reads ffprobe JSON key `codec_type` (not `type`).
- ASR fixtures need espeak-ng speech; sine tones produce no words.
- Synth lesson is 320x180@15fps CRF24 by design (throttled box); tests
  assert the size.
- Aligner: rapidfuzz `token_sort_ratio` (token_set_ratio is subset-lenient).
- `run_ffmpeg` prepends `-hide_banner -nostdin` (stdin hangs otherwise).
- OpenCV pinned <5 (5.0 removed CascadeClassifier); cv2.remap needs
  float32 maps; `connectedComponentsWithStats` connectivity is keyword-only
  in the stubs (positional lands on `labels`).
- Gaze detector: search = aperture bbox rectangle (NOT the bright-aperture
  mask — that would exclude the dark pupil); iris radius ≈ blob extent ×
  0.55 (extent is a diameter).
- Actions YAML: no unquoted `word: word` in step names; actionlint before
  push; PR-event CI unreliable here — use workflow_dispatch for branches.

## Resume protocol for a new session

1. Read this file top to bottom.
2. `git status` + `git log --oneline -10` + `gh pr list` to see actual state.
3. Continue from the first unfinished item in "Next steps".
4. Obey AGENTS.md. Update this file before stopping.
