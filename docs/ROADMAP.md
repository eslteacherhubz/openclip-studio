# ROADMAP — bounded 72-hour plan

Stages are executed in order. Each stage ends with: tests green, `docs/CHECKPOINT.md`
updated, branch pushed, PR opened (except Stage 0, which bootstraps `main`).
Durations are planning aids, not deadlines; the bound is total effort.

| Stage | Window | Deliverable |
|-------|---------|-------------|
| 0 | h 0–4 | Repo bootstrap: governing docs, license, scaffolding, editor verification research, CI skeleton, `main` pushed. |
| 1 | h 4–24 | **Engine core** (`feature/engine-core`): ffprobe wrapper, media model, synthetic media generator, script alignment (with + without ASR), intelligent cut planner, timeline model, bilingual animated ASS captions, audio enhancement, MP4 render, CLI. Full pytest suite incl. integration renders. |
| 2 | h 24–40 | **Desktop app** (`feature/desktop-app`): PySide6 GUI usable end-to-end on Windows (import → script → align → review cuts → captions → export), offscreen-safe for CI, engine unchanged. |
| 3 | h 16–56 (parallel track) | **Eye-contact research** (`feature/eye-contact`): synthetic-eye rig, detector backends, warp-based gaze redirection, evaluation metrics, experiments documented in `docs/EYE_CONTACT_RESEARCH.md`, `openclip gaze-*` CLI behind a flag. |
| 4 | h 40–64 | **CI + packaging** (`feature/ci-packaging`): Actions matrix (ubuntu+windows), ruff+mypy(strict)+pytest gates, headless E2E job, PyInstaller Windows build artifact + checksum, dependency ledger finalized. |
| 5 | h 64–72 | **Hardening + handoff**: README user guide, CHECKPOINT final state, owner validation checklist (what to run on the real Windows machine), tag `v0.1.0-rc1` if all gates pass. |

## Stage rules

- A stage is done only when its `docs/ACCEPTANCE.md` section is satisfied.
- If a stage is blocked, record evidence and move to the next unblocked stage
  (eye-contact is explicitly allowed to be blocked by hardware/footage).
- Never let a branch drift: rebase onto `main` before opening its PR.
- Keep `docs/CHECKPOINT.md` current at every pause point.

## Stage 1 detail (engine core)

Modules under `src/openclip/engine/`:

- `media.py` — ffprobe JSON wrapper → `MediaInfo` (streams, fps, duration, audio params).
- `synth.py` — synthetic lesson footage/audio via FFmpeg lavfi for tests and demos.
- `script.py` — `LessonScript` model: ordered sentences, each with text in primary
  language + optional support-language text (bilingual pairs), punctuation-aware tokenization.
- `align.py` — script↔timeline alignment: ASR backends (faster-whisper when the
  `asr` extra is installed) with fuzzy word alignment (rapidfuzz); fallback even/linear
  alignment for manual mode.
- `cutplan.py` — cut planner: silence gaps, filler words, off-script spans, retake
  detection (repeated script sentences), min-kept-duration guardrails; every decision
  carries a machine + human-readable reason.
- `timeline.py` — timeline/caption model and JSON (de)serialization.
- `captions.py` — ASS subtitle generation: bilingual styles, animated (fade/slide/pop),
  karaoke highlight option.
- `enhance.py` — audio chains: `afftdn` (denoise), `loudnorm` (EBU R128), highpass,
  de-ess via `deesser` when available, `agate`; preset system.
- `render.py` — cut renderer (per-segment re-encode + concat demuxer), caption burn-in
  (`ass` filter), audio enhancement, MP4 mux (`libx264`, `yuv420p`, `aac`), progress callbacks.
- `project.py` — project file model + load/save.
- `cli.py` — `openclip` command: `probe`, `synth`, `render`, `align`, `cutplan`.

## Stage 3 detail (eye-contact research)

- Synthetic eye rig (parameterized photometric eyes with known ground truth).
- Detector backends: OpenCV heuristics (face/eye Haar, iris localization) default;
  no hard dependency on heavyweight models in base install.
- Warp-based gaze redirection: displacement field on eye patches, temporal
  smoothing, feathered blending.
- Metrics: iris-center error px→% of eye width, artifact PSNR outside eye region,
  processing fps on available hardware (documented).
- Experiments with numbers in `docs/EYE_CONTACT_RESEARCH.md`; on-target
  validation checklist for the owner (real webcam footage) — **not** claimed here.

## Out of scope (explicitly)

- GPU inference, cloud services, paid assets.
- Multi-camera editing, 4K proxies, hardware-accelerated encode (documented how to add).
- Real-footage validation claims (owner must run on target hardware).
