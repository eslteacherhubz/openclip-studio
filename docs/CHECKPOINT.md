# CHECKPOINT — exact, resumable progress state

Last updated: 2026-10-09, end of Stage 5 (all planned stages executed; awaiting owner review)

## Environment (dev sandbox where this work ran)

- Linux aarch64 (Debian 12), 8 vCPU (shared/throttled — identical ffmpeg runs
  measured 7s–290s), 7.5 GB RAM, no GPU, no camera.
- Python 3.11.2 (venv `.venv/`), FFmpeg 5.1.9 (apt) + imageio-ffmpeg wheel,
  espeak-ng 1.51 (ASR test fixtures), opencv-python 4.14 (pinned <5: OpenCV
  5.0 removed CascadeClassifier), PySide6 6.8.0.2 + system Qt libs
  (libegl1 libgl1 libxkbcommon0 libglib2.0-0 libdbus-1-3 libfontconfig1
  libpulse0 — same set CI installs), faster-whisper (tiny model cached),
  pyinstaller 6.22 (spec validated locally).
- GUI tests run offscreen (`QT_QPA_PLATFORM=offscreen`); QMediaPlayer has no
  backend in this minimal sandbox (warnings only; works on real desktops).
- `gh` authenticated as `eslteacherhubz` (repo + workflow scopes).
- Windows **not** available here: covered by CI (`windows-latest` engine
  tests + packaging smoke) and deferred to owner validation
  (docs/OWNER_VALIDATION.md).

## Done — full 72-hour plan executed

| Stage | Where | State |
|---|---|---|
| 0 Bootstrap (docs, license, editor verification) | `main` d71452f + CI infra 3317ad9/a608f2a | merged |
| 1 Engine core (alignment, cuts, captions, audio, render, CLI) | PR #1 `feature/engine-core` | green CI, awaiting owner |
| 2 Desktop app (PySide6, cut review, threaded export) | PR #2 `feature/desktop-app` | green CI, awaiting owner |
| 3 Eye-contact research (rig, detector, warp, eval harness) | PR #3 `feature/eye-contact` | green CI, awaiting owner |
| 4 CI matrix + Windows packaging | PR #4 `feature/ci-packaging` | final dispatch in flight (first run had a mypy-job gap: PySide6 stubs; fixed b6f8497) |
| 5 Hardening + handoff | branch `feature/hardening` (this + docs/OWNER_VALIDATION.md) | PR to open after #4's run is green |

Key measured results (all reproducible via `pytest -m gaze`):
- Gaze detection E1: 1.36 px mean error; correction E2: 65.8% mean offset
  reduction (75.6% @ ±5%, 80.0% @ ±15%, 34% @ ±25% — warp-method limit,
  documented); E3: 99 dB PSNR outside eye; E4: 147 fps CPU; pivotal
  parameter study E5: displacement-field reach 2.3× → 15.8% reduction
  (sclera slides with iris) vs 1.35× → 92.8% — see
  docs/EYE_CONTACT_RESEARCH.md. Synthetic rig only; NO real-footage claims.
- Engine: 33-test suite (unit + real-render integration + PSNR
  caption-burn proof + CLI E2E + real whisper-on-espeak ASR (skips in CI)
  + 4 offscreen GUI tests + 7 gaze tests). mypy --strict clean (24 files),
  ruff clean. PyInstaller spec validated on Linux (frozen CLI: --version,
  synth, probe; frozen GUI launches offscreen; imageio-ffmpeg binary is
  collected into the bundle).

## In flight

- PR #4 final CI dispatch (run includes windows package build + smoke).
- feature/hardening: owner-validation doc + final docs polish; PR #5 opens
  after that run is green.

## Blocked

- Nothing. Real-hardware/real-footage validation is deliberately deferred to
  the owner (docs/OWNER_VALIDATION.md) — per master prompt, no
  production-readiness claims were made.

## Next steps for the owner (in order)

1. Review PRs #1 → #2 → #3 → #4 → #5 (stacked; merge in order). All have
   green (or for #4/#5: expected-green) CI. Do NOT trust claims — spot-check:
   `pip install -e ".[dev]" && pytest -v` on any machine.
2. Download the `OpenClipStudio-windows-x64` artifact from the last green
   `ci` run on `main` after merging, and run docs/OWNER_VALIDATION.md on the
   real Windows machine (10–20 min of recording + reporting).
3. Paste gaze numbers/verdicts back into docs/EYE_CONTACT_RESEARCH.md §5/§6.
4. If validation is satisfactory, tag `v0.1.0-rc1` on `main`.
5. Natural next features (not in the 72h bound): blink gating for gaze
   (needs real footage), MediaPipe FaceMesh backend, retake grouping in the
   cut planner, GUI timeline waveform, i18n of the UI itself.

## Key technical notes for resuming sessions

- `probe()` reads ffprobe JSON key `codec_type` (not `type`); falls back to
  parsing `ffmpeg -i` (works in the frozen bundle without ffprobe).
- ASR fixtures need espeak-ng speech; sine tones produce no words.
- Synth lesson is 320x180@15fps CRF24 by design (throttled box); tests
  assert the size. Longest silence is the 1.8s tail, not the 1.6s mid gap.
- Aligner: rapidfuzz `token_sort_ratio` (token_set_ratio is subset-lenient
  and matches wrong windows).
- `run_ffmpeg` prepends `-hide_banner -nostdin` (stdin hangs otherwise).
- OpenCV pinned <5; cv2.remap needs float32 maps;
  `connectedComponentsWithStats` connectivity is keyword-only in stubs.
- Gaze detector: search = aperture bbox rectangle (NOT the bright-aperture
  mask — that excludes the dark pupil); iris radius ≈ blob extent × 0.55.
- Actions YAML: no unquoted `word: word` in step names; actionlint
  (/tmp/opencode/actionlint) before pushing; PR-event CI is unreliable in
  this repo — use `gh workflow run ci.yml --ref <branch>` for branch runs.
- mypy CI job needs `[dev,gui]` (PySide6 ships the stubs mypy checks against).

## Resume protocol for a new session

1. Read this file top to bottom.
2. `git status` + `git log --oneline -10` + `gh pr list` + `gh run list`.
3. Continue from the first unfinished item in "Next steps for the owner".
4. Obey AGENTS.md. Update this file before stopping.
