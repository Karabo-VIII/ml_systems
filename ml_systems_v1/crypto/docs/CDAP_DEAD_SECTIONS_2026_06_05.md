# CDAP dead-sections — hygiene record + follow-ups (2026-06-05)

The 2026-06-04 reset archived large parts of the strategy/oracle layer, leaving several CDAP invariant sections
pointing at deleted files → **silent no-ops** (a "critical" rule that guards 0 files passes vacuously, giving
false confidence). Surfaced by an audit scout 2026-06-05. This is the honest record of what was FIXED vs what
remains, so the cleanup is finishable and nothing stays silently dead.

## Fixed this session (adjacent-problem pass)
- **`simulator::mtm_only_no_double_count`** (was 3 deleted `src/analysis/*` files) → **re-pointed** to
  `src/wealth_bot/**` + `src/strat/**` (now scans 32 live files for the `pnl_bar += ret_from_entry` antipattern).
  Live harness uses different accounting → MtM also covered by the new apparatus gate + reconciliation probe.
- **NEW `strat_apparatus` (CDAP Layer 9, `check_strat_apparatus.py`)** → runs `src/strat/selftest_all.py` every
  commit; exit-2-HALT if the measurement apparatus loses two-sided power. Closes the preflight regression-gap.

## Remaining dead sections — RETIRE or re-point AFTER design review (do NOT rush)
These guard an **archived architecture**; a naive re-point risks a *wrong* guard (false confidence or false-block),
which is worse than a dead one. Each needs a deliberate decision, not a final-minutes edit:
- `walk_forward::purge_gap_present` — 2 of 3 files deleted (`src/strategy/ml/...`, `src/analysis/walk_forward.py`);
  `src/anti_fragile.py` survives. Decide whether `src/strat`'s robustness path (block-bootstrap, not classic
  walk-forward-with-purge) needs this guard, and re-point or scope accordingly.
- `leakage / layer_isolation::strategy_no_direct_{chimera,panel}_read` — point at deleted `src/strategy/gen{3,4,5}/`.
  The new `src/strat` reads via `chimera_loader`; confirm its access pattern, then re-point to `src/strat/**` or
  retire if the loader contract makes direct-read impossible by construction.
- `cost_model`, `strat_99_invariants`, `flag_batch_invariants`, `pipeline_a_plus_closure`, ... — reference the
  archived oracle/gen5 subsystem; most are **retire candidates** (the subsystem no longer exists). Confirm no live
  equivalent before deleting the section.

## 2026-06-07 update (G-E) — the SILENT danger is now CLOSED; per-guard re-point/retire still deferred
The original danger was a CRITICAL section guarding 0 files passing **silently** (false confidence). That is now
**neutralized**: G-F (commit 1bb0f48) wired `scripts/mandatory_gate.py` INTO CDAP (`check_invariants.run_audit`),
so the 18 dead guards surface as a tracked **WARN on every commit** — no longer silent. They cannot give false
confidence anymore (an operator sees them each commit).
**Still deferred (deliberately):** the per-guard RE-POINT-vs-RETIRE decision. Per the "do NOT rush" principle above
AND because the `src/strat` layer is under active rebuild (a wrong re-point at a moving target = false confidence or
false-block on a real-capital guard, which is worse than a tracked-dead one), this is correctly done **with the
strat-layer design review**, not as an isolated CDAP edit. Status: silent-no-op risk CLOSED; cleanup tracked.

## Principle (now enforced)
A CRITICAL section that guards 0 files should be impossible: either it scans live files, or it is retired.
**DONE 2026-06-05:** `scripts/mandatory_gate.py` now has a dead-critical-guard detector (check #4) that flags any
`severity:critical` rule whose `files` resolve to 0 — it currently reports **18** such guards (the full backlog to
retire/re-point). So this class can no longer recur silently; run `python scripts/mandatory_gate.py` to see the
live list. (Most are the archived gen3/4/5 + oracle + strat_99 + flag_batch sections; retire after confirming no
live equivalent.)
