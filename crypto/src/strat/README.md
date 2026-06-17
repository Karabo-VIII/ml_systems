# `src/strat/` — the trading-strategy APPARATUS (validation gate)

> **What this is:** the trustworthy *measurement* layer for the V4 trading project — the gate every
> candidate strategy must pass before it can be believed. It contains **no strategy and endorses none**;
> it is the foundation toolkit the *solving phase* uses to test the avenues in
> [`docs/AVENUE_SPECS_2026_06_05.md`](../../docs/AVENUE_SPECS_2026_06_05.md).
>
> **Status:** rebuilt + consolidated 2026-06-05 from `runs/staging/*_2026_06_04.py` onto the kept
> `wealth_bot.harness`. (The pre-reset `src/strat/` was archived in the 2026-06-04 reset; this is the
> clean port the `discover`/`trader` skills reference.) **Staged for review — not committed.**

## The gate chain (every candidate, in order)

```
cost-honest backtest (taker 0.24%)        wealth_bot.harness.CanonicalHarness
  -> relative leak probe                  wealth_bot.leak_probe.relative_leak_test (auto-wired)
  -> cost-matched random-entry null       strat.firewall.random_entry_null          [PRIMARY GATE]
  -> robustness battery                   strat.battery.evaluate  (Lens A/B/C)
       (block-bootstrap p05>0, jk2>0, jk3>0, n_eff>=8, maxDD<20%)
  -> benchmark-excess incl. bear          strat.benchmark.benchmark_excess  (per-regime vs beta-matched static)
  -> DSR/Holm @ TRUE family-N             src/audit/check_dsr_holm.py  (FIXED 2026-06-05; ship-fails-Holm -> exit 2)
```

> **On "block-bootstrap p05>0" vs the project's "10/10 seeds positive" robustness bar:** the harness
> strategies here are DETERMINISTIC rule systems (no training seed), so the seed-equivalent robustness
> check is the stationary block-bootstrap of the trade sequence (`battery.block_bootstrap_p05_p95`). The
> "10-seed" criterion (CLAUDE.md / RETEST_PLAN) applies to STOCHASTIC/ML candidates — those must add a
> 10-seed outer loop on top of this battery before claiming the 10/10-seed bar.

The one callable that runs cost → leak → firewall → battery and returns a consolidated verdict:

```python
from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec, sma_past_only
from strat import evaluate_candidate
# build a harness with PAST-ONLY indicator columns + a StrategySpec (cost_rt=0.0024 taker), then:
verdict = evaluate_candidate(harness, family_n=<total cells swept>)
# -> {"CONSOLIDATED": "SHIP-TIER" | "NOT-SHIP (...)", "battery": {...}, "firewall_beats_held_out": ...,
#     "leak_probe": {...}, "cost_warning": None|str}
```

## Modules

| Module | Role | Public symbols |
|---|---|---|
| `battery.py` | robustness Lens A/B/C (jackknife, n_eff, block-bootstrap p05, monthly) | `evaluate`, `evaluate_setup_chaser`, + primitives |
| `firewall.py` | LD-4 cost-matched random-entry null (the primary "is it timing or beta?" gate). `regime_matched=True` draws the null from gate-ON bars only (the fairer firewall for a regime-gated candidate). Tiered-cost ladder = deferred solving-phase enhancement. | `random_entry_null` |
| `fill_model.py` | LD-1 cost+fill realism (taker / maker_pessimistic / ideal_ref) | `apply_fill_model`, `MODES` |
| `benchmark.py` | STEP 5 benchmark-excess: candidate-net vs beta-matched costless static hold, per regime (incl. bear) | `benchmark_excess` |
| `candidate_gate.py` | the integrated gate — chains cost→leak→firewall→battery→benchmark | `evaluate_candidate`, `build_clean_reference` |
| `discover.py` | discovery front-end (`discriminate`) + discovery→validation loop (`scan`) | `discriminate`, `scan` |
| (leak probe) | `wealth_bot.leak_probe.relative_leak_test` — cadence-robust look-ahead verdict | — |

## Run the self-tests (RWYB)

```
python src/strat/selftest_all.py     # ONE-SHOT data-free regression (battery + dsr + gate-power + benchmark) -- run this first
python src/strat/battery.py          # synthetic: ship passes, ghost fails A+B, chaser gated
python src/strat/firewall.py         # BTC 1d R12 vs random-entry null (beta-in-disguise demo)
python src/strat/fill_model.py       # taker vs maker_pessimistic (maker collapses)
python src/strat/benchmark.py        # STEP 5: genuine synthetic edge vs beta-matched passive (per regime)
python src/strat/candidate_gate.py   # PEPE whale-gated coarse-SMA through the full hardened gate
python src/strat/discover.py         # discriminate(PEPE) + a tiny scan proof-grid
python src/strat/positive_control.py # POWER check: a synthetic GENUINE edge must beat firewall + be recognized
```

**Two-sided soundness:** the gate both REJECTS (ghost→FAIL, beta→BETA-IN-DISGUISE, whipsaw→FAIL, scan→0
SHIP) AND ACCEPTS (`positive_control.py`: a genuine past-only timing edge beats the firewall, is positive
every window, leak-clean, and is battery-recognized) → calibrated, not a reject-everything sieve. SHIP-TIER
(Lens A) additionally requires n≥15 / n_eff≥15 — a *low-frequency* genuine edge correctly tops out at
PRAGMATIC/PROVISIONAL on a short held-out window (sample-size discipline; SHIP-TIER wants a finer-bar substrate).

## Hardening (2026-06-05 apparatus red-audit)

Verified findings fixed **in this layer**: **F6** (block-bootstrap off-by-one, `battery.py`), **F12**
(adverse-selection sign, `fill_model.py`), **F2** (firewall zero-trade-window bypass, `firewall.py`),
**F11** (integrated gate now uses the cadence-robust `relative_leak_test`, `candidate_gate.py`), **F9**
(taker-cost enforcement/warning, `candidate_gate.py`), **F5** (boundary-crossing forward labels dropped,
`discover.py`), **F7** (p05 truthiness clarified, `battery.py`).

Verified findings that are **canonical-harness** items (documented, review-required — NOT silently
changed): **F8** (`signal_flip` exit logic), **F3** (trade window label uses signal bar not fill bar),
**F9** (StrategySpec default cost is maker), **F13** (`train_start` unused). REFUTED by code-verification:
**F1** (MtM reconciliation — wrong bug class; this engine is trade-level, single-position), **F4**
(cross-window null exit — intentional parity with the real engine).

Full triage with evidence: [`docs/APPARATUS_AUDIT_2026_06_05.md`](../../docs/APPARATUS_AUDIT_2026_06_05.md).

## Upstream: oracle-decomposition (is there ANY learnable entry signal?)

`candidate_gate` tests a SPECIFIC strategy spec. **Before** committing to a signal family, the
oracle-decomposition apparatus answers the prior question — *can the oracle's entries be timed AT ALL from
the available features?* — and so prunes whole avenues cheaply. Hardened + proven 2026-06-06
(`docs/SELF_EVOLUTION_LEDGER.md`, `experiments/adaptive_ma/STRATEGIC_FRONTIER_2026_06_06.md`):

- **Oracle (the realizable ceiling):** `runs/research/oracle_ceiling_builder.py` — perfect-foresight long-only
  max-capture DP (honest open fill + taker cost, hold<7d). **`oracle_high_capture(..., min_move_net=)` is the
  key knob: the oracle objective is a DESIGN VARIABLE.** No floor → a SCALP oracle (decomposes into ~2-bar
  wiggles at every cadence — an unfair test for trend instruments). `min_move_net=0.03..0.05` → a SWING oracle
  (multi-day holds, 5-15% mean net/move = the project's target unit). **Always state the per-move floor + hold
  band so the oracle matches the unit-of-trading you intend.**
- **DNA falsifier (two-sided):** `experiments/adaptive_ma/sol/oracle_dna_shuffled_falsifier.py` — fits
  P(oracle-entry | past-only features) and clears it through shuffled-label control + positive control +
  regime-matched firewall. `--min-move-net` (swing) and `--model {logistic,gbm}` (gbm = nonlinear MA-crossover
  interactions + regime conditioning). **Soundness gate uses the shuffle MEAN (collapse-on-average), genuineness
  uses the p95 tail — do not mix the statistics** (a p95-tail soundness gate gives false leak alarms).
- **Mandatory before believing any "genuine":** **seed-robustness** (stochastic models — a single-seed hit is
  noise; tonight's BTC-1d GBM hit cleared at seed 7, failed at 17/27) **AND OOS→UNSEEN persistence** (OOS capture
  +58% → UNSEEN +0.1% = OOS-overfit, not an edge). These compose with the battery; they are the seed-equivalent
  of the 10/10-seed bar for the *classifier* layer.

Result of the first full application (2026-06-06): no bar-level feature family (MA+orderflow+momentum+micro+
liquidation+book+positioning), linear or nonlinear, times even multi-day swing entries out-of-sample — the
entry-timing avenue is closed at daily/4h; the evidence converges on sub-bar/HF resolution.

## Hard constraints (non-negotiable, all candidates)

- **LONG-ONLY, SPOT, LEVERAGE = 1.** Any backtest that enables shorting/leverage is invalid.
- **TAKER 0.24% round-trip** is the honest cost baseline. Maker-only survival is rejected (calibration provisional).
- **UNSEEN touched once** per candidate, after all spec/hyperparameter choices are final.
- **Objective: WEALTH (compound %)** under the robustness constraint — Sharpe is a secondary diagnostic.
