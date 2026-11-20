# Reinvention audit -- where we hand-rolled vs adopt OSS (2026-06-09)

Triggered by the user stepping back: "are there frameworks for what we're building, or did we reinvent?"
Verdict up front: **mostly NOT reinvented** -- most custom code is justified domain IP or cases the
framework doesn't fit. Two genuine corrections were made; one initial decision (vectorbt) was REVERSED by
its own feasibility probe.

## What was CORRECTED (committed)

| Correction | Why it was real | Commit |
|---|---|---|
| **pandas_ta -> indicator registry** | RSI/MACD/Bollinger were `NotImplementedError` stubs in `INDICATOR_REGISTRY` while pandas_ta (already a dep) provides every family. Now real `Indicator`s in `src/oracle/indicators_ta.py`; all 4 families run through the oracle/adaptive/compare engines. Test gate 10/10. | (pandas_ta commit) |
| **O(D^2) perf fix (NOT vectorbt)** | The "fine-cadence sweep takes hours" wall was an algorithmic bug: `oracle()` runs per decision-day D and recomputed `_sma/_ema/_crosses` on the full slice every (date,config). Memoized per-config crosses once -> per-D slice. Bit-exact (0 diffs), 28.9x at 60 dates. | (perf commit) |

## What was REJECTED (with evidence)

- **vectorbt port -- REVERSED.** My initial call was to port the backtest/sweep to vectorbt. The Opus
  feasibility probe (`runs/oracle/_vbt_probe.py`, reconciled bit-exact vs our engine) found: (1) vectorbt
  CANNOT express the rolling-validity driver (it's a per-decision-day validity score, not a
  signal->portfolio equity curve); (2) the perf wall is NOT what vectorbt accelerates -- vbt speeds up
  per-config total-return (already ~85ms in our numpy) and pays a ~3.1s JIT tax; the real cost is the
  O(D^2) re-mining, which vbt does not touch. A port would add a dependency + a new look-ahead surface to
  re-audit, to speed up a part that's already fast. **The probe refuted the decision before any code
  changed -- the gate working as intended.** (vbt 0.28.1 DOES import cleanly with our numpy<2.3/numba pin
  -- the rejection is on fit, not install.)
- **qlib -- rejected.** Equity/factor-investing oriented, heavy, poor crypto fit, high lock-in.

## What is JUSTIFIED CUSTOM (do NOT port -- domain IP / verified gates / no OSS equiv)

- Robustness gates: `src/strat/{battery,firewall,candidate_gate,benchmark}.py`, `synthetic_positive_control`
  (two-sided soundness) -- domain-specific, audited; no OSS analogue (the membership-matched null is novel).
- `src/wm/v0/v0_baseline/dsr_pbo.py` DSR + PBO-CSCV -- published formulas, not in scipy/arch/statsmodels
  (PBO only in the paid mlfinlab).
- `src/strat/setup_harness.py` SetupHarness + leak_guard, `src/anti_fragile.py` WalkForwardSplitter
  (50/20/20/10 + 400-bar purge -- sklearn TimeSeriesSplit has no purge gap, load-bearing G-AUDIT-002).
- `src/pipeline/{chimera_loader,bar_fabric,parquet_io,dispatch,cli}.py` -- proprietary bar format + the
  atomic-write/exit-2/canonical-CLI contracts; thin stdlib wrappers, not framework-replaceable.
- The oracle-decomposition FRAMING (capture-rate not IC, setup-across-a-move) -- the project's core IP.

## Deliberate trade-offs (NOT corrected -- correcting would be wrong)

- `litellm`/`mem0`/`dspy`: imported in the autonomy harness but LAZY-installed, NOT in requirements.txt.
  This is deliberate -- litellm's tokenizers>=0.21 broke `import transformers` (2026-06-07 incident); the
  lazy-install keeps the base env safe. Adding them to requirements would re-trigger the conflict.
- Dead deps `quantstats`/`bokeh`/`optuna` (declared, unused): low-value to remove (churn); optuna is worth
  adopting only when a sweep exceeds ~50 configs (none does today).
- pandas_ta in the 3 remaining hand-roll `_sma`/`_ema` sites (`ma_oracle_engine`, `data_loader`,
  `regime_classifier`): the custom MA math is fine + fast; swapping is churn for ~150 lines. Skipped.

## Autonomy harness (separate side) -- already integrated

`langgraph` (5 files), `litellm` (2), `mem0` (2) are imported + used. The integrate-don't-reinvent pass for
the autonomy engine was done 2026-06-07. That side is not reinvented.

## The general lesson

When you hit a PERF wall hand-optimizing a generic component, check for an OSS framework BEFORE optimizing
-- the wall is often the signal you're reinventing. BUT verify the framework actually FITS before porting:
here the wall turned out to be an O(D^2) bug in justified-custom code, and the framework (vectorbt) couldn't
express the workload. A wrong port is worse than the status quo. Both halves of "integrate, don't reinvent"
cut: adopt where it fits (pandas_ta), keep + fix where it doesn't (the oracle hot path).
