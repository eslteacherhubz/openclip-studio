# CHECKPOINT — exact, resumable progress state

Last updated: 2026-10-09 (Stage 0 in progress)

## Environment (dev sandbox where this work runs)

- Linux aarch64 (Debian 12), 8 vCPU, 7.5 GB RAM, no GPU, no camera.
- Python 3.11.2, FFmpeg 5.1.9 (apt). Internet available (PyPI, GitHub).
- `gh` authenticated as `eslteacherhubz` (repo + workflow scopes).
- Windows **not** available here; Windows correctness is covered by CI
  (`windows-latest`) and, for real validation, by the owner.

## Done

- Repository audit: repo was completely empty (0 commits, 0 refs local and
  remote). Bootstrap required before branching. No prior progress to resume.
- Environment tooling verified: python, pip, git, gh, ffmpeg (installed via apt),
  internet access.
- Editor verification research completed with evidence:
  `docs/research/EDITOR_VERIFICATION.md`.
- Stage 0 docs written (this file, MASTER, ROADMAP, ACCEPTANCE, AGENTS, README,
  LICENSE).

## In flight

- Stage 0 bootstrap on `main` (scaffolding + initial commit + push). This is
  the one allowed direct-to-main commit (empty repo needs a root; recorded as
  decision D9 in MASTER_OPENCODE_PROMPT.md).

## Blocked

- Nothing yet.

## Next steps (in order)

1. Finish Stage 0: package scaffold (`pyproject.toml`, `src/openclip/`, smoke
   test), commit, push `main`.
2. Branch `feature/engine-core` → implement Stage 1 per ROADMAP → tests → PR.
3. Branch `feature/desktop-app` → Stage 2 → PR.
4. Branch `feature/eye-contact` → Stage 3 → PR.
5. Branch `feature/ci-packaging` → Stage 4 → PR.

## Resume protocol for a new session

1. Read this file top to bottom.
2. `git status` + `git log --oneline -10` + `gh pr list` to see actual state.
3. Continue from the first unfinished item in "Next steps".
4. Obey AGENTS.md. Update this file before stopping.
