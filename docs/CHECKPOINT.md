# CHECKPOINT — exact, resumable progress state

Last updated: 2026-10-11 — Correction pass 1 complete on branch `correction/caption-mapping-and-packaging` (PR open); five stage PRs remain open awaiting owner review.

## Environment (dev sandbox where this work ran)

- Linux aarch64 (Debian 12), 8 vCPU (shared/throttled — identical ffmpeg runs
  measured 7s–290s), 7.5 GB RAM, no GPU, no camera.
- Python 3.11.2 (venv `.venv/`), FFmpeg 5.1.9 (apt) + imageio-ffmpeg wheel,
  espeak-ng 1.51 (ASR test fixtures), opencv-python 4.14 (pinned <5: OpenCV
  5.0 removed CascadeClassifier), PySide6 6.8.0.2 + system Qt libs
  (libegl1 libgl1 libxkbcommon0 libglib2.0-0 libdbus-1-3 libfontconfig1
  libpulse0 — same set CI installs), faster-whisper (tiny model cached),
  pyinstaller 6.22 (spec validated locally, incl. asr+gaze frozen features).
- `gh` authenticated as `eslteacherhubz` (repo + workflow scopes).
- Windows **not** available here: covered by CI (`windows-latest` engine
  tests + packaging smoke incl. frozen export E2E, frozen gaze, frozen
  whisper via SAPI TTS, GUI alive-check) and by owner validation
  (docs/OWNER_VALIDATION.md).

## Done — all planned stages + correction pass 1

| Stage | Where | State |
|---|---|---|
| 0 Bootstrap (docs, license, editor verification) | `main` d71452f + CI infra 3317ad9/a608f2a | merged |
| 1 Engine core (alignment, cuts, captions, audio, render, CLI) | PR #1 `feature/engine-core` | green CI, awaiting owner |
| 2 Desktop app (PySide6, cut review, threaded export) | PR #2 `feature/desktop-app` | green CI, awaiting owner |
| 3 Eye-contact research (rig, detector, warp, eval harness) | PR #3 `feature/eye-contact` | green CI, awaiting owner |
| 4 CI matrix + Windows packaging | PR #4 `feature/ci-packaging` | green CI (9/9 jobs; Windows artifact built, zip + sha256) |
| 5 Hardening + handoff | PR #5 `feature/hardening` | green CI (9/9 jobs) |
| **Correction 1** | `correction/caption-mapping-and-packaging` | this pass — see below |

### Correction pass 1 (this branch)

1. **P1 — caption/timeline mapping fixed** (`engine/render.py::map_span`):
   interior cuts now compress spans correctly (caption 0–10s with cut 3–6s →
   output 0–7s); the old code only shifted by cuts *before* the span start.
   23 regression tests (`tests/test_caption_mapping.py`) cover multiple
   cuts, boundary overlaps, fully-removed spans, unsorted cut lists, and
   duration/monotonicity invariants.
2. **P2 — Windows package audited and completed**: the artifact previously
   did **not** contain whisper (asr) or gaze (opencv) — the packaging env
   installed only `[gui,dev]`. Now bundles `[gui,asr,gaze]` + package data
   (faster-whisper silero VAD ONNX, OpenCV Haar cascades — both were frozen
   -runtime failures found by the audit). Verified inside the frozen app on
   Linux: whisper align with first-use model download (3/3 lines, conf
   0.92–1.00), `gaze-analyze`, full synth→edit→export→probe, probing with
   NO ffmpeg/ffprobe on PATH (bundled binary + fallback), GUI alive-check.
   The Windows packaging job now asserts the same battery (SAPI-TTS speech
   for frozen whisper). Docs updated to claim exactly what is verified
   (README feature matrix, ACCEPTANCE).
3. **P3 — production gap assessment** (`docs/PRODUCTION_GAP_ASSESSMENT.md`):
   per-requirement implemented/partial/missing with evidence. Headline:
   script-directed editing core is implemented; eye-contact is
   research-grade partial; presenter positioning, backgrounds,
   lighting/colour, diagram timing are missing; captions ≠ motion
   graphics; Indonesian-specific tuning not started; MP4 baseline solid
   but unmeasured on target hardware.
4. **P4 — Production Director** (`docs/PRODUCTION_DIRECTOR.md` +
   `src/openclip/director/`): deterministic EDL planner (script markers →
   typed decisions with source line/span/rationale), review gating
   (pending blocks export), meaning-preservation invariants enforced in
   `check_invariants`, executor that composes captions+callout/title ASS
   and refuses unimplemented kinds explicitly (diagram_slot, color_preset,
   reframe, background, emphasis → named NotImplemented). Behavior-
   preserving render refactor: public `build_export_ass` + `render_export`
   (all previous render tests still green). 15 director tests.

Test suite now 70 tests; mypy --strict clean (28 files); ruff clean.

## In flight

- PR from `correction/caption-mapping-and-packaging` (CI dispatched; the
  Windows packaging job grows by ~2–4 min for the new extras + smoke tests).

## Blocked

- Nothing technical. Awaiting owner review/merge of PRs #1–#5 (stacked;
  they conflict with main only until merged in order — CI infra commits on
  main duplicate branch history; merging #1 first clears the path).

## Next steps (in order)

1. Owner: review + merge PRs #1→#5, then the correction PR.
2. Owner: run docs/OWNER_VALIDATION.md on the real Windows machine
   (artifact now includes whisper + gaze; whisper needs one model download).
3. Next implementation slice per docs/PRODUCTION_DIRECTOR.md §8:
   colour/lighting presets → auto-framing → diagram slots → EDL review UI.
4. After real-machine validation + gap closure begins: revisit
   `docs/PRODUCTION_GAP_ASSESSMENT.md` statuses.

## Key technical notes for resuming sessions

- `probe()` reads ffprobe JSON key `codec_type` (not `type`); falls back to
  parsing `ffmpeg -i` (verified in the frozen bundle with empty PATH).
- `map_span`: endpoints shift by cuts strictly before them; spans clip
  against cuts first, then shift. Regression tests are the contract.
- PyInstaller: optional feature data must be added via `_optional_datas`
  in `packaging/openclip.spec` (faster-whisper assets, cv2/data) — package
  data is NOT auto-collected even when the module is.
- Frozen whisper needs HF model download on first use (internet once);
  error path in `WhisperTranscriber.transcribe` explains this.
- Synth lesson is 320x180@15fps CRF24 by design; tests assert the size.
- Aligner: rapidfuzz `token_sort_ratio`; marker lines are never spoken —
  the director planner attaches them to the next aligned spoken line.
- Director EDL: deterministic JSON (sorted keys, ordered decisions); cuts
  may only come from the reviewed cut plan; pending decisions block export.
- Actions YAML: no unquoted colons in step names; actionlint before push;
  PR-event CI unreliable in this repo — use `gh workflow run ci.yml --ref`.
- mypy CI job needs `[dev,gui,gaze]` (PySide6/numpy/cv2 stubs).

## Resume protocol for a new session

1. Read this file top to bottom.
2. `git status` + `git log --oneline -10` + `gh pr list` + `gh run list`.
3. Continue from the first unfinished item in "Next steps".
4. Obey AGENTS.md. Update this file before stopping.
