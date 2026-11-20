# ml_systems_v1

Read-only snapshot of the project as it stood on 2026-11-20, before the root was rebuilt around a single data pipeline.

This tree is reference material. Do not import it from the new root, do not point the new pre-commit hook at it, and do not treat it as the live codebase. History is intact because the move was `git mv`.

What is here:

- `crypto/` — the full crypto system, including `crypto/archive/pipeline_v1`, `crypto/src/pipeline_v2` (the package that was copied out and renamed), live books under `crypto/runs/`, and the old CDAP audit
- `games/`, `harness/`, `forex/`
- `.claude/` and `CLAUDE.md`

The live pipeline is `crypto/src/pipeline` at the repository root, outside this folder.
