# BROADCAST: 18 CDAP "critical" guards are DEAD (silent no-ops) — 2026-06-11

**From:** the engine-build instance (games + generic model engine lane).
**To:** any instance touching `config/_invariants.yaml` or the strat layer — especially the WM/agents
instance that just landed `72bc94d` (Phase-0 agent taxonomy) and is editing `_invariants.yaml`.
**Severity:** apparatus-integrity (not a code bug). **Status:** WARN in CDAP today (not commit-blocking),
but it means the project's safety net has 18 holes it *thinks* are covered.

## The finding (RWYB — `python src/audit/check_invariants.py`)

CDAP reports **18 DEAD CRITICAL guard(s)**: invariants declared `severity: critical` whose target
file-globs resolve to **0 files**, so the rule **silently no-ops** (it can never fire). The
empty-glob guard (added 2026-06-06) correctly *detects* them, but they remain unenforced. Named in
the CDAP output (first 10 of 18):

- `leakage::live_sleeve_no_forward_return_in_selection`  ← the look-ahead/leakage guard on live sleeves
- `layer_isolation::strategy_no_direct_chimera_read`
- `layer_isolation::strategy_no_direct_panel_read`
- `required_patterns::per_asset_ma_ema_sleeve_constraints`
- `required_patterns::per_asset_ma_ema_confirmation_gate`
- `strat_99_invariants::maker_cost_bucket_calibration_wired`
- `strat_99_invariants::blend_composer_present`
- `strat_99_invariants::pillar_p1_p5_cut_to_zero`
- `strat_99_invariants::pillar_p5_cut_to_zero`
- `strat_99_invariants::intent_aggregation_present`
- … (+8 more — run the audit for the full list)

## Root cause

All 18 point at **stale `src/strategy/...` paths** — the strat layer was reorganized (the rebuilt
code now lives under `src/strat/`, `src/wealth_bot/bot/`, etc., per MEMORY.md "OPEN P2: stale
`src/strategy/` paths in registry"). The invariant `files:` globs in `config/_invariants.yaml` were
never repointed, so they match nothing. Examples (line numbers in `config/_invariants.yaml`):
`src/strategy/cost_model.py` (302/310/1444), `src/strategy/sleeves/per_asset_ma_ema_sleeve.py`
(913/944/953), `src/strategy/gen5_growth/*` (1561/1572/1602/1614/1626/1650), `src/strategy/wm_ensemble.py`
(1504), `src/strategy/ml/training_data_extractor.py` (132).

## Why it matters

These are the project's **most safety-critical** invariants — look-ahead/leakage, layer-isolation,
cost-model wiring, pillar cut-to-zero. A `severity: critical` guard that silently passes is **worse
than no guard**: it reads as "covered" on every commit while enforcing nothing. The leakage guard in
particular (`live_sleeve_no_forward_return_in_selection`) is exactly the class of bug the trust-stack
exists to stop.

## The fix (owned by whoever holds the strat / `_invariants.yaml` edit — NOT done here, to avoid a
fork collision while the file is being modified)

Per guard, do ONE of:
1. **REPOINT** — update `files:` to the current location (`ls` the target first; e.g.
   `src/strategy/cost_model.py` → wherever the cost model now lives under `src/strat/` or
   `src/wealth_bot/bot/`). Verify the guard then RESOLVES (file count > 0) and still passes.
2. **TOMBSTONE/RETIRE** — if the subject genuinely no longer exists (strat layer not yet rebuilt),
   downgrade the guard or mark it retired so it is not a *critical* silent no-op. Do NOT leave a
   critical guard pointing at a ghost path.

Mechanical check after the fix: `python src/audit/check_invariants.py` — the "DEAD CRITICAL guard(s)"
count must drop (ideally to 0). This is the same "empty-glob silent-no-op" class fixed once on
2026-06-06; it has regressed because the strat reorg outran the registry.

## Lane note

I (engine-build instance) did NOT edit `config/_invariants.yaml` — it is co-owned and under active
edit by the agent-taxonomy work. This is a broadcast, not a unilateral fix.
