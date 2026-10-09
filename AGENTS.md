# AGENTS.md — Engineering rules for all agents (humans and AI) on this repo

## Source of truth

1. `docs/MASTER_OPENCODE_PROMPT.md` — the governing assignment and constraints.
2. `docs/ROADMAP.md` — the bounded 72-hour plan and stage order.
3. `docs/ACCEPTANCE.md` — per-stage acceptance criteria; do not mark done without them.
4. `docs/CHECKPOINT.md` — exact, resumable progress state. **Update it before you stop working.**

## Workflow

- Work in isolated feature branches: `feature/<stage>-<slug>`.
- Push milestones; open PRs; **never merge to `main` automatically** — the owner reviews and merges.
- Keep commits small and described in the imperative ("Add cut planner tests").
- Every stage must leave `main`-bound PRs green (lint + tests) before being considered reviewable.

## Non-negotiable constraints

- **No money.** No paid APIs, services, or assets.
- **No private footage to third parties.** All analysis local. No network calls in the editing pipeline. ASR and gaze models must run offline once installed.
- **No secrets** in code, logs, or commits. CI uses repository secrets only.
- **Licenses**: project code is GPL-3.0-or-later. Only add dependencies whose licenses are GPL-3-compatible (MIT, BSD, Apache-2.0, LGPL, ISC, MPL-2.0, Zlib). Record every new dependency in `docs/ACCEPTANCE.md` § dependency ledger. If a dependency is GPL-incompatible, stop and ask the owner.
- **No production-readiness claims** without real end-to-end testing on target hardware and explicit owner validation.

## Engineering standards

- Python 3.11+. Style: `ruff` (line length 100), types annotated, `mypy --strict` on `src/` (soft until Stage 4, then enforced in CI).
- Package layout: `src/openclip/`. Engine code (`src/openclip/engine/`) must not import GUI or gaze modules — keep layers clean: `engine` ← `gaze` ← `gui`.
- Heavy/optional deps go in extras (`gui`, `gaze`, `asr`), never the base install; base install must be pip-installable on a clean machine with only stdlib + tiny pure deps.
- All media I/O goes through FFmpeg subprocess (ffprobe JSON, filter graphs) — no binary blobs in git.
- Tests: `pytest`, synthetic media generated on the fly via FFmpeg lavfi (see `engine/synth.py`); no fixtures checked in. Integration tests must run in CI without network or models.
- CLI first, GUI second: every engine capability must be reachable headlessly (that is how CI exercises it).
- Hardware reality: development/validation happens on the sandbox available at the time (document it in CHECKPOINT). Windows-specific behavior must be covered by CI on `windows-latest`. Never claim target-hardware results that were not actually run there.

## Eye-contact research rules

- Document every experiment (method, data, parameters, numbers, verdict) in `docs/EYE_CONTACT_RESEARCH.md`.
- Synthetic rigs are legitimate for algorithm development but are never sufficient for a "works on real footage" claim — state that explicitly in docs.
- If blocked (no footage/hardware), implement behind a flag, prove what can be proven, and continue the rest of the roadmap.

## Checkpoint discipline

`docs/CHECKPOINT.md` must always contain: what is done (with commit hashes), what is in flight (branch + PR), what is blocked and why, and the exact next step to resume. A new session must be able to continue from it without archaeology.
