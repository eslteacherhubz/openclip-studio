# MASTER PROMPT — Eterna OpenClip Studio (governing assignment)

This document is the source of truth for the assignment that governs this
repository. It was the brief under which all planning and implementation in
this repo was executed. If any doc, commit, or decision conflicts with this
file, this file wins until the owner changes it.

## Assignment

Repository: `eslteacherhubz/openclip-studio` ("Eterna OpenClip Studio").

Build, as an autonomous lead-development effort, a genuinely usable
**open-source Windows desktop application** for editing **bilingual
teaching videos** (ESL teacher talking-head lessons), within a **bounded
72-hour plan**.

Priorities, in order:

1. Verify a suitable existing open-source Windows video editor.
2. Investigate and experimentally test realistic eye-contact correction on
   the target hardware.
3. Implement bilingual teaching-video editing with script alignment,
   intelligent cuts, visual composition, animations, audio enhancement and
   MP4 export.
4. Build a genuinely usable Windows desktop application.
5. Implement meaningful GitHub CI, Windows packaging, and end-to-end tests.
6. Preserve all progress in Git commits and a detailed checkpoint file.

## Hard constraints

- **No money.** No paid APIs, services, or assets. Free/open-source only.
- **No private footage to third parties.** All analysis is local. No network
  calls in the editing pipeline. ASR and gaze models must run offline once
  installed.
- **No secrets** in code, logs, or commits.
- **Licenses**: project code is GPL-3.0-or-later. Dependencies must be
  GPL-3-compatible (MIT, BSD, Apache-2.0, LGPL, ISC, MPL-2.0, Zlib).
  Record every dependency in `docs/ACCEPTANCE.md` § dependency ledger.
- **No production-readiness claims** without real end-to-end testing on the
  target hardware (the owner's Windows machine) and explicit owner
  validation.
- **Never merge to `main` automatically.** Feature branches → PRs → owner
  review. (Stage 0 bootstrap on `main` is the sole exception, as the empty
  repo needs a root before branching.)

## Process rules

- Work in isolated feature branches named `feature/<stage>-<slug>`.
- Keep commits small; imperative commit messages.
- Every stage must leave PRs green (lint + tests).
- If eye-contact correction is blocked on available hardware, document the
  evidence (in `docs/EYE_CONTACT_RESEARCH.md` and `docs/CHECKPOINT.md`) and
  continue with the other editing capabilities.
- Continue through the roadmap for as long as the execution environment
  permits; when interrupted, preserve exact progress in
  `docs/CHECKPOINT.md` so a new session can resume without archaeology.

## Decisions log (autonomous engineering decisions)

Each decision below was made autonomously and can be revisited by the owner.

| # | Decision | Rationale |
|---|----------|------------|
| D1 | **Don't fork an existing editor; build a focused Python application.** Verified editors (Shotcut, Kdenlive, OpenShot; see `docs/research/EDITOR_VERIFICATION.md`) are excellent general-purpose NLEs but none ships the teaching-specific pipeline (script alignment → intelligent cuts → bilingual animated captions → audio enhancement). Forking a C++/Qt NLE (Shotcut/Kdenlive/MLT) to embed that pipeline costs far more than a focused tool, and OpenShot's Python libopenshot bindings are not packaging-friendly on Windows. FFmpeg remains the render engine (as in all of them), so output quality matches. |
| D2 | **Architecture: layered Python package.** `engine` (no GUI deps, CLI-first, headless-testable) ← `gaze` (optional extra) ← `gui` (optional extra). CI exercises everything headless through the engine; GUI is a shell over the same engine calls. |
| D3 | **Render engine: FFmpeg subprocess** (segment render + concat demuxer for cuts; filter graphs for enhancement; ASS subtitles for animated bilingual captions). Same primitives Shotcut/Kdenlive/LosslessCut use; no binary blobs in git; `imageio-ffmpeg` provides fallback binaries. |
| D4 | **ASR: faster-whisper (MIT), optional extra `asr`.** Local, offline after model download, word-level timestamps, CPU-adequate for lesson-length audio. The pipeline degrades gracefully (manual/even alignment) when the extra is absent, so the base install stays tiny and CI never downloads models. |
| D5 | **Gaze correction: warp-based redirection, research-first.** No GPU and no camera in the dev sandbox; the target is the owner's Windows PC with a webcam (CPU-only). Implemented behind a flag with a synthetic-eye experimental rig + evaluation harness; all experiments recorded in `docs/EYE_CONTACT_RESEARCH.md`. On-target validation explicitly deferred to the owner (documented command to run). |
| D6 | **GUI: PySide6** (LGPL-3; also available GPL). Cross-platform (dev on Linux, ship on Windows), QtMultimedia/FFmpeg-backed preview, first-class accessibility, packaging via PyInstaller. |
| D7 | **Windows packaging: PyInstaller** one-folder build + zip artifact from GitHub Actions `windows-latest`; no code-signing (costs money) — SmartScreen warning documented for users. |
| D8 | **CI: GitHub Actions** matrix (ubuntu-latest, windows-latest) × (3.11, 3.12): ruff, mypy (soft→enforced at Stage 4), pytest with FFmpeg-synthesized media, and a headless end-to-end render job that asserts on the exported MP4 via ffprobe. |
| D9 | **Stage 0 bootstrap committed directly to `main`** (docs + scaffolding) because branching from a zero-commit repo leaves nothing to branch from; all subsequent stages go through PRs. |

## Bounded plan

See `docs/ROADMAP.md` for the 72-hour stage breakdown and
`docs/ACCEPTANCE.md` for per-stage acceptance criteria.
