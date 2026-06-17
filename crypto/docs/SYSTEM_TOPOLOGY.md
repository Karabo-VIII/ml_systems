# System Topology — the agent/meta framework: where it lives, how it interacts, how it evolves (2026-06-05)

> The canonical map of the autonomous-agent system + its weak-point register + the AUTHORITATIVE control-surface
> hierarchy + the change-cadence model. Written so any instance (or the user) can see the whole machine at once,
> find the chokepoints, and know which control source wins when two disagree. Companion to
> [AUTONOMY_FRAMEWORK.md](AUTONOMY_FRAMEWORK.md) (the design) and [`OVERSEER.md`](../.claude/skills/_common/OVERSEER.md) (the role).

## 1. The map — components by LAYER x LIFETIME x REPRODUCIBILITY

| Layer | Components | Runs / loaded | Tracked? |
|---|---|---|---|
| **L0 Constitution** (knowledge) | `CLAUDE.md`, `STATE.md`, `memory/*` | per session (CLAUDE.md always) | tracked |
| **L1 Protocols** (prose discipline) | `OVERSEER.md`, `AUTONOMOUS_RUNNER.md`, `.claude/skills/*`, `docs/AUTONOMY_FRAMEWORK.md` | on demand | tracked |
| **L2 Mechanical hooks** | `autonomy_loop.py` (Stop), `permission_gate.py` (PreToolUse), `autonomous_mode_check.py` + `user_prompt_router.py` (UserPromptSubmit) | every turn / tool call, by the harness | tracked |
| **L3 Gates** | `check_invariants.py` (CDAP), `mandatory_gate.py`; manifests `_invariants.yaml`, `mandatory_gates.yaml` | commit-time / on-demand | tracked |
| **L4 Wiring** | **`.claude/settings.json`** (wires every hook + permission baseline) | session start | **LOCAL (gitignored)** |
| **L5 Live state** | `frontier.json` (the plan), `permission_policy.json` (runtime perms), `autonomous_mode.json` (arming+envelope) | per run | mixed: autonomous_mode.json tracked; frontier/policy/AUTONOMY_ON LOCAL |
| **L6 Execution** | the Opus instance + sub-agents/workflows; `autonomy_driver.py` (cross-session) | ephemeral | driver tracked |

## 2. Control flow — one turn through the machine
```
user/loop prompt
  -> [L2 UserPromptSubmit] user_prompt_router + autonomous_mode_check  => inject OVERSEER role + mode banner
  -> [L6] model reasons; for each tool call:
        -> [L2 PreToolUse] permission_gate reads L5 permission_policy.json => allow / deny / normal-flow
  -> model writes work (src/docs/...) and/or meta (L0/L1/L2/L3); commits
        -> [L3] git pre-commit -> CDAP (check_invariants) [+ mandatory_gate when wired]
  -> model finishes
  -> [L2 Stop] autonomy_loop reads L5 (arming + frontier) => block(next node) | allow_stop
Cross-session: [L6] autonomy_driver spawns fresh `claude -p` per node; state persists in L5 + git.
```

## 3. Weak-point register (the chokepoints — what the meta must watch)
| # | Weak point | Where | Risk |
|---|---|---|---|
| W1 | **Wiring is LOCAL** — `.claude/settings.json` (gitignored) wires every hook | L4 | a fresh clone has all parts, zero wiring -> **the system is INERT until hand-rebuilt**. #1 reproducibility gap. |
| W2 | **Self-modification has no barrier** — a hook can be edited while it governs the session | L2 | a bad self-edit to `autonomy_loop.py` -> hook errors -> fail-open -> **the loop silently STOPS**. (permission_gate has a fail-safe; the Stop hook's fail-open = silent halt.) |
| W3 | **Dual arming** — `AUTONOMY_ON` (frontier) vs `autonomous_mode.json` (envelope) | L5 | two sources of "am I autonomous?"; already caused a banner that said OFF while the loop ran. |
| W4 | **Dual permissions** — settings `deny`/`defaultMode` vs gate `permission_policy.json` | L4/L5 | overlap + conflict (settings-deny blocked an edit the gate would have allowed). |
| W5 | **Prose vs mechanism gap** — OVERSEER.md promises dispatch/drift-control nothing enforces | L1 vs L2 | the meta behaves as disciplined as the model chooses; not mechanically guaranteed. |
| W6 | **No concurrency control** — shared L5 state + git, no lease/lock | L5/L6 | parallel instances race on `frontier.json` / `memory/` / commits. |

## 4. The AUTHORITATIVE control-surface hierarchy (harmonization DECISIONS — which source WINS)
The fix for W3/W4 is NOT to merge everything (that adds machinery) but to declare ONE authority per concern and make the others derive from / defer to it.

- **Arming (W3) -> AUTHORITY: `runs/autonomy/AUTONOMY_ON` (CORRECTED 2026-06-06; see §9).** It is the single arming source, and it lives OUTSIDE `.claude/` so writing it never triggers the IDE/config-dir prompt that `.claude/autonomous_mode.json` does (esp. when that file is open in the editor). `launch_autonomy.py` writes it + records the window in the frontier's `stop_conditions`; the Stop hook (`autonomy_loop.py`) arms on it when `autonomous_mode.json` is absent. `.claude/autonomous_mode.json` is now **DEPRECATED for arming** — if present with `autonomous:false` it acts ONLY as a manual kill-switch override (it wins, so a stale switch can't keep a disarmed run alive). ONE place to look: "am I autonomous?" -> does `runs/autonomy/AUTONOMY_ON` exist? (The pre-2026-06-06 authority was the reverse; it inverted because the `.claude/` write prompted the user — the prompt-free path is now authoritative.)
- **Permissions (W4) -> AUTHORITY: `permission_policy.json` (the gate).** It is the runtime permission source (hot-reloadable). `settings.json` permissions are **bootstrap-only**: `defaultMode` + the hardcoded-protected denies (settings self-edit) that the harness enforces pre-gate. Everything else lives in the gate policy. ONE place to change perms at runtime -> `permission_policy.json`.
- **Rules are SEPARATE CONCERNS, not redundant sources (clarifies, doesn't merge):**
  - `_invariants.yaml` = **code-correctness** invariants (CDAP). "Is the code right?"
  - `mandatory_gates.yaml` = **gate-enforcement** ("which gates can never be skipped").
  - `permission_policy.json` = **tool-permission** ("what may the agent do right now").
  These three answer different questions; documenting that they don't overlap removes the *perception* of contention without adding machinery.

## 5. Change cadence — cadence ~ 1 / blast-radius (how to evolve in real-time, correctly)
| Tier | What | Change when | Guard before live |
|---|---|---|---|
| **Dials** | `frontier.json`, `permission_policy.json` | per-loop, in-flight | reversible; low blast-radius; just edit |
| **Mechanisms** | L2 hooks, L1 skills, L3 gates, the driver | per-task / checkpoint | **MUST pass the meta-change gate** (py_compile + the component's selftest) BEFORE trusted-live (closes W2) |
| **Constitution** | `CLAUDE.md`, core `_invariants.yaml` | per-epoch, deliberate | reviewed; binds ALL instances; never per-loop |
**Rule:** per-loop you touch only Dials + clear bug-fixes. Mechanisms change at checkpoints, gated. The Constitution changes rarely and deliberately. A meta-change is not trusted until it passes its own gate — that is what makes "evolve as we go" *correct* instead of *thrashing*.

## 6. Parallel-instance coordination (the W6 protocol — design)
For N instances/skills on different parts of one problem:
1. **Lease the frontier** — a worker acquires a lease on the node(s) it owns (`runs/autonomy/leases/<node>.lease` with holder+timestamp; stale-after-TTL reclaim). No two workers take the same node.
2. **Append-only memory** — workers append to `memory/` (never rewrite shared files); the overseer merges.
3. **Worktree-per-worker** — code-mutating workers run in isolated git worktrees; the overseer merges/arbitrates.
4. **Overseer = single arbiter** — only the overseer writes the canonical `frontier.json` status + merges worktrees + resolves conflicts. Workers propose; the overseer disposes.
(Built minimally as a lease primitive; full multi-worker orchestration is the Workflow tool's job.)

## 7. Reproducibility contract (the W1 fix)
A clone bootstraps the ENTIRE system from tracked artifacts via `scripts/autonomy/bootstrap.py`:
wires the hooks into `settings.json` (from a tracked `settings.hooks.template.json`), seeds `permission_policy.json`
from its template, and creates an empty `frontier.json`. After bootstrap, `python scripts/mandatory_gate.py` and
`python .claude/hooks/test_permission_gate.py` confirm the wiring. NOTHING load-bearing stays only in a local file
without a tracked template + an installer that recreates it.

## 8. -k Falsifier baseline (harmonization must NOT add net machinery)
Start-of-window control-surface count (the contention metric): **arming sources = 2** (AUTONOMY_ON, envelope),
**runtime permission sources = 2** (settings deny, gate policy). Target after harmonization: **arming = 1**
(authoritative autonomous_mode.json), **runtime permission = 1** (authoritative gate policy). New files added by
harmonization (bootstrap, lease, meta-gate, this doc) must be justified as REDUCING contention or closing a named
weak point (W1/W2/W6), not as new control surfaces. If a change raises the authoritative-surface count, it is REVERTED.

## 9. 2026-06-06 additions (harmonised — keeps §1 map + §4 authority current)
Components built/changed this date, in the §1 layer×lifetime×authority schema:
- **`PersistentCliBrain`** (`scripts/autonomy/metaop/brain.py`, `--backend persistent`) — Tier-1 brain; carries a warm
  `claude -p --resume` session across nodes (no per-node cold-start = the main latency/quota saving); rebirth on
  context-limit; graceful CliBrain fallback. RWYB-verified (2-node continuity smoke). Closes register #14.
- **Governance fences (W3/W4 enforcement, defense-in-depth):** loops physically CANNOT commit/push or write control
  surfaces. Two layers because metaop has two execution paths: `metaop/tools.py` HARD_DENY/HARD_FILE_DENY (the
  `Worker.run_shell`/`write_file` path) AND `permission_gate.py` `METAOP_LOOP` env-marker (the `claude -p` delegated
  path — the layer that actually governs the solutioning loops). The OVERSEER (no marker) commits freely.
- **ARMING AUTHORITY CORRECTED (supersedes §8's target):** the single arming authority is now **`runs/autonomy/
  AUTONOMY_ON`** (prompt-free, outside `.claude/`). `.claude/autonomous_mode.json` is DEPRECATED for arming — writing
  it prompts (it is a `.claude/` config write, and is often open in the IDE). `launch_autonomy.py` arms via
  AUTONOMY_ON + removes a stale `autonomous_mode.json`. autonomy_loop.py honors AUTONOMY_ON when autonomous_mode.json
  is absent. (100% no-prompt lever = the user launching with `--dangerously-skip-permissions`.) Arming surfaces = 1.
- **Loop-2 = 60s DUAL-VIEW meta** (project audit + running-loop tasks) on the watcher heartbeat; loop-3's 3h project
  audit folded into it. **`watcher.py`** gained STALL-detection (alive-but-hung loop → wake overseer) + self-respawn.
- **`DIRECTIVES_REGISTER.md`** (`.claude/skills/_common/`) — the canonical record of ALL standing user mandates
  (read-forward at each `/orc` cycle). **`/orc`** (renamed from `orchestrator`) is the default operating-model skill.
- **`skill_diagnostics.py`** widened (G1) to scan skill SUB-files (closes the "0 ERROR != clean" blind spot).
