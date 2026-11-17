# ROADMAP — phased development plan (replaces the obsolete 72-hour plan)

Stages 0–10 are complete on the feature chain `feature/ai-face-analysis-tracking`
(HEAD `2d4baeb`; CI green). The plan below tracks what remains, in recommended
execution order. Each phase ends with: tests green, `docs/CHECKPOINT.md`
updated, branch pushed, PR opened.

| Phase | Focus | Deliverable | State |
|-------|-------|-------------|-------|
| 0 | Bootstrap | governing docs, license, scaffolding, CI skeleton, `main` | done (main) |
| 1 | Engine core | alignment, cut planning, bilingual captions, audio enhancement, MP4 render, CLI | done (#1) |
| 2 | Desktop app | PySide6 GUI end-to-end, offscreen-safe | done (#2) |
| 3 | Eye-contact research | synthetic rig, detector backends, warp-based correction, CLI | done (#3) |
| 4 | CI + packaging | Actions matrix, ruff/mypy/pytest gates, PyInstaller Windows artifact | done (#4) |
| 5 | Hardening + handoff | README, CHECKPOINT, owner validation | done (#5) |
| Corr 1 | Caption mapping, packaging truth, gap plan | `map_span` fix, whisper+gaze in bundle, gap assessment | done (#6 corr) |
| Corr 2 | Overlay timing, review enforcement, cut normalization | export-timeline overlays, pending-gating, `normalize_cut_spans` | done (#6 corr) |
| 8 | Smart reframing | subject-tracking 9:16 crop, reframe preview | done (#12 chain) |
| 9 | Shared face-analysis tracking | Mock + stubbed MediaPipe providers, CachedFaceAnalysis, smoothing | done (#12 chain) |
| 10 | Beauty engine | skin smoothing, micro-sculpt, spot-heal, `BeautyConfig` | done (#12 chain) |
| 10 Obj 4 | Beauty integration | render pre-pass + GUI checkbox/sliders + CLI `--beauty` | done (#12 chain) |
| Priority 3 | Video preview player (core) | `VideoPreviewWidget`: load/seek/play-pause, cv2-guarded placeholder | done (#12 chain) |
| 11 | **Real MediaPipe provider** | wire `MediaPipeFaceAnalysisProvider` to real FaceMesh; real-footage validation test (beauty_off byte-identical; beauty_on visual parity) | **next** |
| 12 | Hook / viral-moment detection | detect engagement hooks and viral moments from aligned script + audio stats | planned |
| 13 | Contextual B-roll sourcing | match B-roll assets to script lines/topics; overlay scheduling | planned |
| 14 | Platform export presets | 9:16/1:1/4:5 presets + platform-specific encoding (TikTok/Reels/Shorts) | planned |
| 15 | Quality/perf | high-quality beauty intermediate, encoder perf, two-pass loudnorm | planned |
| 16 | Beta hardening | real-machine validation, bug bash, `v0.1.0-rc1` tag (owner decision) | planned |

## Phase rules

- A phase is done only when its `docs/ACCEPTANCE.md` section is satisfied.
- `docs/CHECKPOINT.md` must be current at every pause point.
- Never let a branch drift: rebase onto `main` (or merge `main` in) before
  opening its PR — see docs/MERGE_PLAN.md for the recommended chain order.

## Out of scope (explicitly)

- GPU inference, cloud services, paid assets, paid model APIs as the default
  path (BYOK provider keys are optional; `local_mock` is the default).
- Multi-camera editing, 4K proxies, hardware-accelerated encode
  (documented how to add; not claimed).
- Real-footage production claims (owner must run validation on target hardware;
  Phase 11 includes the real-footage test scaffold).
- Beauty-on output is intentionally NOT byte-identical to source (that is the
  point of retouching); beauty-off remains byte-identical — sacred invariant.
