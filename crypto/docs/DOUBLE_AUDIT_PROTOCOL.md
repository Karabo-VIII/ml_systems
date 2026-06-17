# RED TEAM Audit Protocol — mandate (post 2026-04-28)

**Stance: adversarial.** Every change is presumed broken until proven otherwise. The reviewer's job is to ATTACK the diff, find ways it could fail in production, look for hidden coupling — not to charitably validate that it "looks fine." Production-grade code requires red-team, not green-team, review.

Origin: 2026-04-28 strategy-layer audit caught **15 bugs** that had passed initial review (1 CRITICAL = strategy layer unimportable, 1 HIGH = walk-forward purge gap=0 in xsec ranker, 13 others). The CRITICAL bug landed because the gen3 consolidation moved engines without verifying that all consumers (notably `strat_profiles.py:33-50`'s flat `from engine_X` imports) still resolve. We caught it ONLY because the user explicitly asked for an end-to-end audit.

**To prevent this recurrence, all non-trivial code changes are now subject to a TWO-STAGE audit:**

## Stage 1 — Audit-as-you-build (during development)

Apply at every step of a multi-file refactor or feature build. Mandatory for changes touching ≥3 files OR any file that has ≥3 external importers (the "blast radius" threshold).

**Per-step checklist** (≤30 seconds each):

1. **Caller search**: after editing function `X` or moving file `Y`, run:
   ```powershell
   # Function/method change
   grep -rn "from <module> import <X>\|<X>(" src/ scripts/

   # File move
   grep -rn "from <old.module>\|import <old.module>" src/ scripts/
   ```
   Update every match.

2. **Compile-check the change**:
   ```powershell
   python -m py_compile <changed_file>
   ```

3. **Smoke-import the module** (cheaper than a full pipeline run):
   ```powershell
   python -c "import <module>; print('ok')"
   ```
   For larger refactors:
   ```powershell
   python scripts/validate_all_models.py --quick     # 18/18 PASS sanity
   python src/pipeline/run_pipeline.py --status      # T0..T7 health
   ```

4. **Commit the diff small**: never bundle a refactor into a feature commit. The CRITICAL strat_profiles bug landed because the consolidation move was bundled into a 27-file commit, hiding the broken imports under "moved files don't really change behavior."

## Stage 2 — Audit-on-completion (before commit)

Apply ONCE at the end of any multi-file change set, before `git commit`. This is the catch-all for emergent issues from interactions between Stage-1-clean individual changes.

**Per-completion checklist**:

1. **Re-import the touched modules**:
   ```powershell
   python -c "
   import sys; sys.path.insert(0, 'src/strategy')
   import strat_profiles as sp                    # canary for strat-layer health
   print(f'PROFILES={len(sp.PROFILES)}, MODULES={len(sp.MODULES)}')
   "
   ```

2. **Run validate_all_models.py** if any model file was touched:
   ```powershell
   python scripts/validate_all_models.py --quick
   ```
   Required result: 18 PASS / 0 FAIL / 2 FROZEN.

3. **Run the gap-finder**: open `docs/GAPS.md`, scan the OPEN section. If the change SHOULD have closed a gap, mark it ✅ CLOSED with the date. If the change SURFACED a new gap, file a new G-XXX-NNN entry. *Never silently pass through known gaps.*

4. **Run the cross-version invariant audit** for any settings.py change:
   ```powershell
   # See CLAUDE.md "Cross-Version Training Invariants" table
   grep -rE "^TRAIN_RATIO|^VAL_RATIO|^WM_BATCH_SIZE|^DIRECT_RETURN_WEIGHT|^TWOHOT_FOCAL_GAMMA|^target_prefix" src/wm/v*/v*_training/settings.py
   ```

5. **Spawn a Sonnet auditor agent** if the change is high-stakes (>10 files changed, OR strategy/training/cost-model code, OR pipeline DAG):
   ```
   Agent({
     description: "Pre-commit audit",
     subagent_type: "Explore",
     prompt: "Audit the diff of branch <X>: read git diff HEAD, look for
              <project-specific anti-patterns: walk-forward gap, MtM
              double-count, look-ahead bias, sign errors, hardcoded magic
              numbers, untested error paths>. Report under 200 words
              ranked by severity."
   })
   ```
   Agent reports must be VERIFIED by the human before treating as truth — agents can hallucinate (see `STRAT_LAYER_AUDIT_2026_04_28.md` where 1 of 5 agent CRITICAL findings was overstated).

6. **Reference gap IDs in commit messages**: every commit that closes a gap must `Closes G-XXX-NNN` in the message body. This is grep-able later.

## Project-specific anti-patterns to scan for

These have caused real bugs in this project — Stage 2 should explicitly check each:

| Pattern | Where it bit us | Stage-2 grep |
|---|---|---|
| Walk-forward purge gap = 0 | xsec ranker (G-AUDIT-002) | `grep "df\[.timestamp.\] >= TRAIN_END_MS"` — should have `- _PURGE_MS` |
| MtM double-count | `short_term_speculator_v2` (fixed 2026-04-22) | `grep "ret_from_entry" src/analysis/*.py` — must NOT add to `pnl_bar` |
| Look-ahead in regime/feature | BOCPD warm-up (G-AUDIT-011) | `grep "np\.\(nanmean\|nanstd\|mean\|std\)" engines/` — verify scoped to past data |
| Cost-model bypass | gen3 engines (G-AUDIT-010) | `grep "0\.001\|0\.002\|fee = " src/strategy/gen3_engine_stack/` — flag hardcoded fees |
| Stale imports after move | strat_profiles (G-AUDIT-001) | `grep "from engine_\|^import engine_" src/` — verify path resolves |
| Same-day publication race | ETF / panel features (G-AUDIT-008) | `grep "ffill()" src/strategy/gen4_frontier_alpha/` — should chain `.shift(1)` |
| Inline gitignore comments | `.gitignore` (G-PIPE-005) | `grep "/    #" .gitignore` — must be `# comment\nrule/` |
| Output-path drift | bar_fabric / panels (G-PIPE-001/002) | `grep "data/frontier/\|data/bars/" src/` — must use `_layout.bars_dir()` etc. |
| Two-orchestrator capital silo | paper_trader_v2 vs gen5_growth (G-STR-009) | per `ORCHESTRATOR_RECONCILE_2026_04_28.md` — design pending |
| Non-causal regime (full-history std) | BOCPD (G-AUDIT-011) | warmup-window scoped now |
| Capture-output buffering on long stages | bar_fabric / panels heartbeat | `grep "capture_output=True" src/pipeline/build_*.py` — must be False for streaming |

## When to skip the audit

- Doc-only changes (`*.md`).
- Adding a new test that doesn't change tested code.
- Comment/typo edits.

Anything else: **both stages run.**

## Failure mode handling

If Stage 2 surfaces a finding:
- 🟡 LOW/MEDIUM: file a new GAPS entry (G-XXX-NNN) and address in a follow-up commit. Don't block the current commit.
- 🟠 HIGH: address in the same commit if ≤30min; otherwise file as 🟢 SCOPED with a timeline.
- 🔴 CRITICAL: **HALT THE COMMIT.** Fix in-place. The CRITICAL bug landing is exactly what this protocol prevents.

## How this gets enforced

- The protocol is documented in `CLAUDE.md` ("Code Change Verification" section, points 13-14 added).
- New commits that close audit gaps reference the G-AUDIT-NNN ID.
- Periodic (weekly) audit pass: spawn a Sonnet agent against the strategy layer and pipeline; expect zero new CRITICAL findings.

## Why "double" not just "single"

A single end-of-development audit catches issues but only after the developer has already lost cache-warm context on early changes. By that time, fixes are slower and more error-prone. Stage-1 audits catch issues while context is hot; Stage-2 catches emergent issues from interactions. The cost is ~5 minutes per multi-file change — well below the cost of a CRITICAL bug landing in main.

---

## Contract-Driven Audit Protocol (CDAP) — extension 2026-04-28

The Stage-1 + Stage-2 audits caught file-local issues but missed cross-file
contract violations (universe-propagation skew, cross-version constant
drift, MtM double-count regressions, walk-forward purge gaps, output-path
drift). Common thread: **contract under-specification + static-audit
blindness to runtime violations.**

CDAP closes that gap with three orthogonal axes.

### Axis 1 — Every component declares a `__contract__` dict

Top-of-file, before any imports:

```python
__contract__ = {
    "kind": "pipeline_stage" | "strategy" | "model" | "simulator" | "splitter" | ...,
    "inputs": {
        "args":        ["--universe {u10|u50|u100}", "--workers"],
        "upstream":    ["data/processed/.../*.parquet"],
        "config_keys": ["data.start_date"],
    },
    "outputs": {
        "files":       "data/.../<sym>_*.parquet",
        "columns":     [...],
        "value_ranges": {"col": [lo, hi]},
    },
    "invariants": {
        "atomic_write":               True,
        "column_name_verify":         True,
        "purge_gap_bars_min":         400,
        "asset_set_eq":               "downstream:<other_kind>",
    },
    "rationale": "...",
}
```

Loaded by `src/audit/contract_loader.py` via AST (no execution).
Cross-contract validation in `validate_contracts()`.

### Axis 2 — Audit runs at THREE checkpoints

| Checkpoint | When | Tool |
|---|---|---|
| Static (pre-commit) | every diff | `python src/audit/check_invariants.py` (auto via pre-commit hook) |
| Runtime smoke | before merge | minimal end-to-end exercise of changed code path; e.g. `python src/pipeline/run_pipeline.py --dry-run --universe u10 --tiers all` |
| Production drift | weekly | re-check invariants vs prod state (p_fill calibration vs default, realized DD vs declared DD halt, live IC vs declared IC) |

### Axis 3 — Global invariants registry

`config/_invariants.yaml` declares cross-cutting invariants that don't
belong in any single file:
- Cross-version constants (DIRECT_RETURN_WEIGHT, BIN_MIN, BIN_MAX, ...)
- Walk-forward purge gap minimums
- Cost-model defaults vs calibration warnings
- Simulator MtM-no-double-count regression guard
- DAG ordering (chimera_legacy after fetch_only, chimera_v51 after both)
- CLI universe support across pipeline scripts
- Atomic-write contract for silver/gold producers

Validated by `src/audit/check_invariants.py`; exit 2 = halt commit.

### Pre-commit hook

Install once per checkout:
```
python src/audit/install_hook.py
```

Idempotent. Bypass once (discouraged) via `SKIP_CDAP=1 git commit -m "..."`
with explanation in commit message.

### Bootstrap status

- **Phase 1 (registry + checker + hook)**: ✅ shipped
- **Phase 2 (contracts on top-leverage files)**: ⚠️ ~10 files annotated;
  back-fill remaining ~50+ on next touch
- **Phase 3 (this section)**: ✅ shipped

### Historical audit gaps caught by CDAP

| Past failure | CDAP axis that catches it |
|---|---|
| Hawkes hardcoded u10 | Axis 1 input contract + Axis 2 static (downstream consumer mismatch) |
| chimera_legacy ran late | Axis 1 output contract + Axis 3 DAG invariant |
| MtM double-count (5x inflation) | Axis 1 simulator invariant + Axis 3 regression guard |
| DIRECT_RETURN_WEIGHT drift V3/V4/V6 | Axis 3 cross-version constant |
| WM_BATCH_SIZE drift V2/V7 (caught **today**) | Axis 3 cross-version constant |
| p_fill 0.80 vs 0.30 calibration | Axis 2 production drift check |
| Walk-forward purge gap = 0 | Axis 3 walk-forward invariant |
| Strategy imports broken (G-AUDIT-001) | Axis 1 input contract + Axis 2 static py_compile every consumer |
| TEMPORAL_CTX_DROP batch-level | Axis 3 cross-version invariant |
| Output-path drift (G-PIPE-001/002) | Axis 1 output contract |

100% catch rate on this list once Axis 1 + Axis 3 are populated.
