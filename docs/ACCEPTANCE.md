# ACCEPTANCE — per-stage criteria

A stage is done only when every box here for that stage is checked and
verifiable from the repo (commit, CI run, artifact, or doc). Update the
checkboxes in the same PR that completes the stage.

## Stage 0 — bootstrap

- [x] Governing docs exist: MASTER_OPENCODE_PROMPT.md, ROADMAP.md, ACCEPTANCE.md, CHECKPOINT.md, AGENTS.md, README.md
- [x] LICENSE is GPL-3.0-or-later (full text in repo)
- [x] Editor verification research committed with evidence (docs/research/EDITOR_VERIFICATION.md)
- [x] Python package scaffold installs (`pip install -e .[dev]`) and smoke test passes
- [x] `main` pushed to origin

## Stage 1 — engine core

- [x] `openclip probe` returns ffprobe-derived MediaInfo on a synthetic file
- [x] `openclip synth` generates a synthetic bilingual lesson (video+audio) usable by all later stages
- [x] Script alignment works with mock ASR (unit) and, when the `asr` extra is installed, with faster-whisper (integration, skippable in CI)
- [x] Cut planner produces silenced/filler/off-script cuts with reasons; unit tests cover each decision type
- [x] Caption generation produces valid ASS (libass/ffmpeg parses it); bilingual styles present; animation tags present
- [x] Audio enhancement chain applies without error on synthetic audio
- [x] End-to-end: synthetic lesson → script+cutplan → render → ffprobe asserts: h264/aac streams, yuv420p, duration ≈ planned kept-duration (±2 video frames + 0.25 s)
- [x] All tests green locally; ruff clean; mypy --strict clean
- [ ] PR opened from `feature/engine-core` → merged by owner

## Stage 2 — desktop app

- [x] PySide6 app launches offscreen (CI: `QT_QPA_PLATFORM=offscreen`) and headful (dev sandbox)
- [x] User can: create/open project, import media, paste bilingual script, run alignment (or manual), review each proposed cut (accept/reject), configure captions, export MP4 — all without touching a terminal
- [x] GUI never blocks the main thread on rendering; progress surfaced to the user
- [x] GUI smoke tests green on ubuntu CI (`gui` job); import-time test on windows CI lands in Stage 4
- [ ] PR opened from `feature/desktop-app` → merged by owner

## Stage 3 — eye-contact research

- [x] Synthetic eye rig generates parameterized eyes with ground-truth gaze offsets
- [x] At least two experiments documented in docs/EYE_CONTACT_RESEARCH.md with method, parameters, numbers, verdict (E1–E5, incl. reach-factor and offset-limit parameter studies)
- [x] Warp-based correction reduces iris-center error on the rig by ≥60% at ≤25% eye-width offsets without dropping PSNR outside the eye region below 35 dB (measured: 65.8% mean reduction, 99 dB — rig-validated; real-footage claim explicitly deferred)
- [x] `openclip gaze-analyze` and `openclip gaze-apply` work behind the `gaze` extra + flag; degrade cleanly when opencv missing
- [x] On-target (owner Windows PC) validation checklist written; no production claim made
- [ ] PR opened from `feature/eye-contact` → merged by owner

## Stage 4 — CI + packaging

- [x] Actions on push/PR: ruff, mypy (strict on src/, enforced from this stage), pytest — ubuntu (3.11+3.12) + windows (3.11+3.12) matrix
- [x] Headless E2E job (synthetic → render → ffprobe asserts) runs in CI on both OSes with no network beyond pip install (ASR/gaze/gui extras auto-skip or install via extras; no model downloads in CI)
- [x] `windows-latest` job builds a PyInstaller one-folder artifact (GUI + headless CLI, bundled ffmpeg from imageio-ffmpeg) + SHA-256 checksum, uploaded to the run (spec mechanics validated on Linux in the sandbox; the Windows job produces the real artifact)
- [x] Correction pass: the frozen bundle now **contains and is smoke-verified** for whisper (faster-whisper + ctranslate2 + VAD asset + first-use model download) and gaze (opencv + cascade data) — validated in the frozen app on Linux and asserted by the Windows packaging job (export E2E, gaze smoke, SAPI-speech whisper align, GUI alive-check). Previously these extras were missing from the artifact; the "self-contained" claim now covers exactly what is verified.
- [x] Dependency ledger (below) complete and license-checked
- [ ] PR opened from `feature/ci-packaging` → merged by owner (CI green on the branch before handoff)

## Stage 5 — hardening + handoff

- [x] README user quickstart (GUI + CLI) with a 5-step path (screenshots deferred: no display in dev sandbox)
- [x] CHECKPOINT.md reflects final state and exact next steps for the owner
- [x] Owner validation checklist: what to install, run, and report on the real target machine (docs/OWNER_VALIDATION.md)
- [ ] If all gates pass: tag `v0.1.0-rc1` (still not a production-readiness claim) — reserved for the owner after real-machine validation

## Correction pass 1 — caption mapping, packaging truth, production gap plan

- [x] `map_span` fixed for interior cuts (caption 0–10s with cut 3–6s → 0–7s); regression suite `tests/test_caption_mapping.py` (23 cases: multiple cuts, boundary overlaps, fully-removed spans, unsorted input, duration/monotonicity invariants)
- [x] Windows package audited: whisper + gaze were **absent** from the artifact (packaging env lacked the extras); now bundled (`[gui,asr,gaze]`) with package data (silero VAD ONNX, Haar cascades) and verified inside the frozen app: whisper align with first-use model download, gaze-analyze, full synth→edit→export→probe pipeline, ffprobe-less probing, GUI launch — all on Linux locally; asserted by the expanded Windows packaging job in CI
- [x] Production gap assessment delivered (docs/PRODUCTION_GAP_ASSESSMENT.md): precise implemented/partial/missing per owner requirement
- [x] Production Director architecture + v1 skeleton (docs/PRODUCTION_DIRECTOR.md, `src/openclip/director/`): deterministic EDL planner, typed inspectable decisions, review gating, meaning-preservation invariants, executor with explicit unimplemented kinds; 15 tests
- [ ] PR opened from `correction/caption-mapping-and-packaging` → merged by owner

## Correction pass 2 — overlay timing, review enforcement, cut normalization

- [x] Director overlays use export-timeline timestamps via `map_span` against normalized accepted cuts; overlays fully inside cuts are dropped (`director/executor.py`)
- [x] Rejected/pending `caption` decisions never render; `RenderOptions.burn_captions` honored through the director path (suppresses captions only; callout/title decisions governed by their own status); engine `render_project` behavior unchanged
- [x] `normalize_cut_spans` (merge overlap/duplicate/adjacent) shared by `map_span`, `render_project`, and the executor; `cuts [3,7]+[5,9]` in a 0–10 s caption → 4 s output (tested); renderer kept-intervals and mapping agree (cross-checked)
- [x] Marker lines detected in `engine/script.py` and skipped by the aligner (they no longer steal ASR words from the next spoken line — defect found by the review-enforcement tests)
- [x] Pixel-verified video-render regression (`tests/test_director_render_timing.py`): overlays probed in exported frames before/during/after cuts, fully-removed overlay absent, raw-timestamp sentinels empty; negative-controlled against the bug (289/285 px leak detected with raw spans, <10 px after fix)
- [x] Full suite + mypy --strict + ruff green; Windows packaging CI exercised via the dispatched full run
- [ ] Merged by owner (PR #6, final SHA recorded in CHECKPOINT)

## Dependency ledger

Every runtime/test dependency, license, and where it is used. Keep in sync with `pyproject.toml`.

| Dependency | Version | License | Used for | Notes |
|---|---|---|---|---|
| imageio-ffmpeg | >=0.4.9 | Apache-2.0 (wheel) / FFmpeg binaries: GPL/LGPL builds | guaranteed ffmpeg binary fallback | subprocess only, never imported for frames |
| rapidfuzz | >=3 | MIT | fuzzy word alignment | |
| numpy | >=1.26 | BSD-3-Clause | gaze math | extra `gaze` |
| opencv-python | >=4.8,<5 | Apache-2.0 | face/eye detection, warping | extra `gaze`; OpenCV 5.0 removed CascadeClassifier — pin <5 |
| faster-whisper | >=1.0 | MIT | local ASR | extra `asr`; models downloaded once, offline thereafter |
| ctranslate2 | (via faster-whisper) | MIT | ASR runtime | |
| PySide6 | >=6.6 | LGPL-3.0 (also GPL-3) | GUI | extra `gui`; LGPL use via unmodified pip wheel |
| pytest | >=8 | MIT | tests | dev |
| ruff | >=0.4 | MIT | lint | dev |
| mypy | >=1.10 | MIT | types | dev |
| pyinstaller | >=6.0 | GPL-2.0 (bootloader exception) | Windows packaging | dev/build tool; used as documented in Stage 4 CI |

All licenses GPL-3-compatible. FFmpeg is invoked as a subprocess (not linked),
so FFmpeg's GPL/LGPL status does not restrict this codebase.
