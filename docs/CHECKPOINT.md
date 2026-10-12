# CHECKPOINT — exact, resumable progress state

Last updated: 2026-10-09 (Stage 2 complete on branch; PRs #1 and #2 pending owner merge)

## Environment (dev sandbox where this work runs)

- Linux aarch64 (Debian 12), 8 vCPU, 7.5 GB RAM, no GPU, no camera.
- **CPU availability is highly variable** (shared box): identical ffmpeg runs
  measured 7s–290s. Test media was sized down (320x180@15fps) accordingly;
  treat slow renders as environment noise, verify with `time`.
- Python 3.11.2 (venv at `.venv/`), FFmpeg 5.1.9 (apt) + imageio-ffmpeg wheel
  binary (7.0.2), espeak-ng 1.51 (for ASR test fixtures), internet available.
- faster-whisper installed in venv; whisper `tiny` model downloaded to local
  cache (~/.cache) — ASR tests run fully offline after that.
- PySide6 6.8.0.2 installed in venv + system Qt libs (libegl1 libgl1
  libxkbcommon0 libglib2.0-0 libdbus-1-3 libfontconfig1) — GUI tests run
  offscreen (`QT_QPA_PLATFORM=offscreen`). QMediaPlayer has no backend in
  this minimal sandbox (warnings only; playback works on real desktops).
- `gh` authenticated as `eslteacherhubz` (repo + workflow scopes).
- Windows **not** available here; Windows correctness covered by CI
  (`windows-latest`) and, for real validation, by the owner.

## Done

- **Stage 0** (main, d71452f): governing docs, LICENSE (GPL-3.0),
  editor verification research, package scaffold.
- **Stage 1** (PR #1, `feature/engine-core`, green CI, awaiting owner merge):
  complete headless engine — probing, synth lessons, bilingual script parsing,
  fuzzy alignment + faster-whisper backend + even fallback, intelligent cut
  planner (silence/filler/off-script with reasons), animated bilingual ASS
  captions, FFmpeg audio enhancement presets, MP4 render (segment+concat+
  final pass with source→export time mapping), project JSON, CLI
  (probe/synth/new/align/plan/render/auto/gui).
- **CI plumbing**: workflow registered on main (infra-only commit 3317ad9,
  decision D10: GitHub only registers workflows from the default branch, so
  PR checks were impossible while ci.yml lived only on a feature branch).
  Known pitfall fixed: unquoted colons in step names break Actions YAML —
  validate with `actionlint` (binary cached at /tmp/opencode/actionlint).
- **Stage 2** (branch `feature/desktop-app`, PR about to open): PySide6
  MainWindow (script editor with bilingual placeholder help, cut-review
  table with keep/cut checkboxes + double-click-to-seek, caption animation/
  karaoke/burn toggles, audio preset, CRF, language + transcriber pickers,
  video preview + play slider), threaded Analysis/Render workers with
  progress bar, `openclip gui` + `openclip-gui` entry points, 4 offscreen
  GUI tests. mypy --strict + ruff clean; all 25 tests green locally.

## In flight

- PR #1 (Stage 1) and PR #2 (Stage 2, stacked on #1) — awaiting owner.

## Blocked

- Nothing. Eye-contact (Stage 3) sandbox limits are planned for
  (synthetic-rig validation + owner checklist), not blocking.

## Next steps (in order)

1. Open PR for `feature/desktop-app` (base main; contains Stage 1 commits
   until #1 merges — standard stacking).
2. Stage 3: branch `feature/eye-contact` off `feature/desktop-app` —
   synthetic eye rig, OpenCV detector backend, warp-based gaze redirection,
   experiments with numbers → `docs/EYE_CONTACT_RESEARCH.md`, CLI
   `openclip gaze-*` behind `gaze` extra (numpy+opencv).
3. Stage 4: branch `feature/ci-packaging` — windows-latest matrix job
   (pytest incl. gui imports), mypy gate comment removed (strict stays),
   PyInstaller one-folder Windows artifact + SHA-256, dependency ledger
   final pass.
4. Stage 5: hardening — README quickstart polish, final CHECKPOINT, owner
   validation checklist.

## Key technical notes for resuming sessions

- `probe()` must read ffprobe JSON key `codec_type` (not `type`).
- Whisper fixture: espeak-ng makes speech offline; sine tones will NOT work
  as ASR input.
- The synth lesson is 320x180@15fps CRF24 (perf: this box). Do not "fix" the
  small size — tests assert it.
- Longest silence in synth plan is the 1.8s tail, not the 1.6s mid gap.
- `run_ffmpeg` prepends `-hide_banner -nostdin`; keep that (stdin hangs).
- Aligner uses rapidfuzz `token_sort_ratio`; `token_set_ratio` is
  subset-lenient and matches wrong windows (verified experimentally).
- GUI: keep engine imports out of `openclip/gui` at *module import time* only
  if PySide6 missing — lazy import in `_cmd_gui` already handles it.
- GitHub Actions YAML: never put an unquoted `word: word` colon inside a
  step name. Run `actionlint` before pushing workflow changes.

## Resume protocol for a new session

1. Read this file top to bottom.
2. `git status` + `git log --oneline -10` + `gh pr list` to see actual state.
3. Continue from the first unfinished item in "Next steps".
4. Obey AGENTS.md. Update this file before stopping.
