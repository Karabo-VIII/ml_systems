# `_shared` -- class-agnostic agent helpers

Genuinely class-agnostic building blocks shared across A1 / A2 / A1H (generic
policy heads, reward utilities, etc.). **KPI:** N/A -- this is a utility module,
not an agent; it owns no held-out compound number.

**Status this phase: EMPTY SCAFFOLD.** The `src/agent/` cluster moved into
`a1_wm_consuming/` is one tightly-coupled module graph wired by bare same-dir
imports rooted on its A1-specific `config.py` (e.g. `policy.py`/`rewards.py` do
`from config import <A1-specific constants>`, and `ppo.py` does
`from policy import ...`). Extracting helpers here would require rewriting that
bare-import contract = a refactor, not Phase-0 plumbing. Such extraction is
deferred to a later phase; nothing is moved here in Phase 0.
