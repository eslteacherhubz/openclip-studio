# MERGE_PLAN.md — ANALYSIS ONLY

This file is analysis. **Do not execute any rebase or merge.** It documents the
relationship between `main` and the stacked feature chain (#7–#14) so the owner
can merge deliberately.

## 1. Branch state (observed from git)

Merge base of `main` and `feature/ai-face-analysis-tracking`:
`d71452f` (Stage 0: bootstrap repo with governing docs, license, package scaffold).

| Direction | Count | What |
|-----------|-------|------|
| `main` ahead of feature chain | **2** | `3317ad9` CI: register workflow on main; `a608f2a` CI: fix workflow syntax on main. Both touch only `.github/workflows/ci.yml`. |
| Feature chain ahead of `main` | **43** | All of stages 1–5, correction passes 1–2, Phases 8–10, Priority 3 preview, on branches #7–#12 (#13/#14 not yet branched from the tip). |

So the feature chain is **not** "9 behind main" — it is 43 commits ahead. `main`
is essentially Stage 0 plus two CI-infra commits. The feature chain already
contains everything main lacks plus all the October correction-pass changes.

## 2. "October correction pass" — main vs feature chain

The task references an "October correction pass" that changed caption mapping,
export-timeline overlays, and review enforcement. **Observed reality: those
changes are already on the feature chain, not on main.** `main` has none of
them.

| Correction | Commit (feature chain) | On `main`? |
|------------|------------------------|------------|
| Caption timeline mapping (`map_span` interior cuts) | `64642cc` Fix caption timeline mapping for interior cuts | No |
| Export-timeline overlay mapping + review enforcement + cut normalization | `ad93911` Correction 2: export-timeline overlay mapping, review enforcement, cut normalization | No |
| Production gap assessment + Production Director v1 | `2ec90a2` Correction pass: production gap assessment, Production Director architecture | No |
| Whisper+gaze in Windows package | `ab6df00` Bundle and verify whisper+gaze in the Windows package | No |

Conclusion: there is **no diverged correction-pass logic on `main` to reconcile**.
The feature chain is a strict superset of `main` modulo `main`'s 2 CI-infra
commits. Merging the chain into `main` brings main up to date; the only
conflict surface is `.github/workflows/ci.yml`.

## 3. Predicted conflict files

| File | Why it may conflict | Risk |
|------|---------------------|------|
| `.github/workflows/ci.yml` | `main` added the workflow (3317ad9) + fixed syntax (a608f2a); the feature chain rewrote it extensively (Stage 4 CI matrix, mypy strict gate, gaze job, PyInstaller packaging, ruff/mypy pins, gui/gaze marker exclusions — commits `b8b7016`…`2173f45`) | **Low** — both sides are YAML syntax fixes to the same file; resolve by taking the feature chain's version (it is a superset that also runs on windows and installs extras). |
| `src/openclip/engine/render.py` | Touched on feature chain only (`RenderOptions.beauty`, `normalize_cut_spans`, `build_export_ass`) | None — main has no changes here. |
| `src/openclip/director/executor.py` | Touched on feature chain only (beauty pre-pass, overlay mapping) | None — main has no changes here. |
| `src/openclip/cli.py` | Touched on feature chain only (`--beauty` flags) | None — main has no changes here. |
| `src/openclip/gui/main_window.py` | Touched on feature chain only (beauty checkbox/sliders, preview widget) | None — main has no changes here. |

No source-code conflicts are expected because `main` diverged only in CI
infra. The feature chain touched every source file that matters, exclusively.

## 4. Recommended merge order

Chain dependency order (bottom → top), must be preserved:

```
#7 feature/ai-transcript-clip-byok      (transcript editing, ai-edit-transcript CLI)
#8 feature/ai-provider-foundation       (BYOK provider foundation + mock)
#9 feature/ai-render-primitives-cli     (render primitives zoom/audio + CLI)
#10 feature/ai-gui-transcript-captions  (transcript panel + caption styles + GUI)
#11 feature/ai-edl-schema               (EDL schema + review bridge)
#12 feature/ai-face-analysis-tracking   (smart reframing + face-analysis + beauty + preview) <- current tip
#13 feature/ai-broll-export-gui         (not yet branched from tip)
#14 feature/ai-customization-intelligence (not yet branched from tip)
```

Recommended procedure:

1. **Merge `main` into the chain tip first** (#12 `feature/ai-face-analysis-tracking`):
   - `git checkout feature/ai-face-analysis-tracking && git merge main`
   - Resolve the single `ci.yml` conflict by taking the feature-chain version.
   - This brings `main`'s 2 CI commits into the chain so every downstream PR builds on a CI-aware base.

2. **Merge the chain into `main` in order #7→#14**:
   - For each PR branch in order: `git checkout main && git merge <branch>` (or merge each PR into main via GitHub PR).
   - Because each branch builds on the previous, merge #7 first, then #8 on top of the now-larger main, etc.
   - After each merge, run CI (lint + mypy strict + pytest 377 + gui + gaze) before proceeding to the next.

3. **#13 and #14**: not yet created. When created, branch them from the post-#12 tip (or from main after #12 is merged) and repeat step 2.

## 5. Rebase-onto-main vs merge-main-into-chain

| Approach | Fit for this stacked chain | Recommendation |
|----------|---------------------------|----------------|
| Rebase each of #7–#14 onto `main` | Poor — 8 separate rebases, each risks re-resolving the same `ci.yml` conflict, and rebasing a stacked chain requires rebasing every downstream branch after each rebase. High friction, high error surface. | Avoid. |
| Merge `main` into the chain tip (#12), then merge chain → main | Good — one `ci.yml` resolution at the tip; the whole chain then flows into `main` as a superset. | **Preferred for the bulk merge.** |
| Merge `main` into each PR #7→#14 in order, then merge each into main | Best if the owner reviews each PR individually (8 PRs). Each PR brings main's CI commits forward incrementally; `ci.yml` conflict resolved once per PR but trivially (take feature version). | **Preferred for per-PR review.** |

Bottom line: **merge-main-into-chain** (either at the tip once, or incrementally per PR) is the low-friction path. Rebase-onto-main is discouraged for a stacked 8-PR chain.

## 6. Risks

- **CI skew**: `main`'s CI job is minimal (ruff + pytest ubuntu); the feature chain expects mypy strict + gui + gaze + windows jobs. If #7–#14 are merged one at a time before `main`'s ci.yml is updated, intermediate PRs may run the minimal job and miss regressions. Mitigate by merging `main`→chain first (step 1) so ci.yml is current before any PR merge.
- **Test count drift**: the chain targets 377+ passing tests. Any merge that drops `gui`/`gaze` extras from the CI install would silently skip those tests. Verify the merged ci.yml installs `[dev,gui,gaze]` for the mypy/gui/gaze jobs.
- **No source conflicts expected**: since `main` has no source changes, the merge is mechanically safe; the only manual resolution is `ci.yml`.

## 7. Definition of done for the merge

- All 8 PRs (#7–#14) merged into `main` in order.
- `main` CI green: lint, mypy `--strict` (47+ source files), pytest 377+, gui offscreen, gaze, test-windows, package-windows.
- `docs/CHECKPOINT.md` updated to record the merged SHAs.
- `main` is then the superset; `feature/ai-face-analysis-tracking` (and #7–#14) can be deleted.
