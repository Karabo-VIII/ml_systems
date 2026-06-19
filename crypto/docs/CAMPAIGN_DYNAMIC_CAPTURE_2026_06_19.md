# Campaign report — Dynamic TI×TF Move-Capture Engine (autonomous, 2026-06-18/19)

Charter: [`project-dynamic-capture-engine-charter-2026-06-18`]. Mandate: build a REPEATABLE, DYNAMIC
TI×Timeframe×Asset move-capture engine that ADAPTS to regimes, trend-following (not prediction), ranked by
WEALTH, dev on 2020 / iron on 2021, using the whole project + its lessons. Run autonomously with ~5-min heartbeats.
Commits: 43f284f, 3a8969c, c13abb0, ec3c2c0, e967386, caa6ccd (+ a clean strict re-run).

## What was built (repeatable framework)
- **`src/strat/dynamic_capture_engine.py`** — orchestrator: 8 MA × 6 TF × u10, TAKER-floor grade, WEALTH rank,
  capture-rate diagnostics, p05 block-bootstrap, reproducibility fingerprint (config-hash + git SHA), tiered
  register, two-sided selftest. Tier-1 SMA-200 position GATE (`--gate sma200`).
- **`src/strat/ti_capture_sweep.py`** — non-MA TI expansion: reuses `deep2020_ti_pipeline.INDICATORS` (base/iron
  held_fns + grids) through the same gated stack + dev-2020/forward-2021/2022 protocol + band-ensemble.
- Extended **`ma_strat_builder.py`** — configurable cost (taker floor), static-only path, **strict** SMA-200 gate,
  forward-span replay helper (`_net_on_span`), 2021/2022 forward eval. Selftests pass.

## Phases + results (RWYB, dev 2020 6/3/3, iron/forward 2021, 2022 no-touch bear, taker, fixed-EW u10)
1. **P0 static floor** (8 MA × 6 TF, ungated): 45 B_preserve / 0 A_allweather. Every cell bleeds the 2022 bear
   (-10..-30). De-risked beta across the whole grid.
2. **P2 Tier-1 SMA-200 gate**: fixes the universal bear-bleed (bear22 -> ~0). Forward-validated — 43/48 cells
   positive on unseen 2021; 15 all-weather MA cells (1d/2h HMA/TEMA/DEMA/WMA/VIDYA...).
3. **P1 non-MA TI expansion**: 9 all-weather TI cells, all at 1d, all TREND-following (ADX top, +
   SUPERTREND/DONCHIAN/MACD/VORTEX/TSI/KELTNER/PSAR/ROC). Mean-reversion (RSI/STOCH/CCI/BBPCT/WILLR) is DEAD
   (doesn't translate to 2021). Fine TFs (1h/30m/15m) DEAD (over-gated + cost-bound).
4. **The killer null** + **adversarial verification** (Workflow, 5 agents) caught **TWO real bugs**:
   - `_sma` `min_periods=1` -> the SMA-200 gate gave 3 late-listing assets (SOL/DOGE/AVAX) a partial-window avg
     hugging their listing pump = a fabricated trend filter; inflated gate-only +937% (100% from those 3 names).
     FIXED: strict `min_periods=N` gate. **An earlier headline ("gate dominates buy-hold") was RETRACTED.**
   - `ma_strat_builder.py:404` selected the static band on `oo>0` (OOS-in-band) -> net_oos inflated +0.6..+6pp. FIXED.
5. **RESCUE-B** (conviction-scaled exposure) + **validation** (shuffle-null + forward): the TI has a real, modest,
   bull-only within-bull asset-SELECTION skill for MACD/DONCHIAN (conviction tilt beats gate-only AND a matched-gross
   shuffle null by 2-3σ on held-out 2020 + forward 2021); TI-dependent (SUPERTREND noisy, KELTNER overfits).

## The verified verdict (corrected continuous 2020-10..2022-12, EW-u10, taker, long-only spot, no leverage)
| strategy | full-cycle net% | maxDD% |
|---|---:|---:|
| **raw buy-hold** | **+549** | -79 |
| strict SMA-200 gate (buy-hold) | +324 | -54 |
| engine cells (ADX/MACD/DONCHIAN/SUPERTREND 1d) | +115..+179 | -13..-25 |
| conviction-tilt (gate + TI selection) | +365..+370 | -35..-55 |

- **On WEALTH: raw buy-hold WINS. No internal long-only signal beats it.** The move-capture signal over-de-risks.
- The engine is a **minimum-drawdown de-risked-beta allocator** (DD -13..-35% vs BH -79%), valuable only under a
  hard maxDD-constrained mandate.
- The **one genuinely-new positive**: a real (modest, bull-only, MACD/DONCHIAN) within-bull SELECTION edge from the
  TI as a conviction tilt — Pareto-improves the gate (+365%/-35% vs +324%/-54%). Not wealth alpha.
- **Move CAPTURE (discrimination) is real; harvestable WEALTH alpha over buy-hold is NOT** — re-earned per-TI,
  forward-validated, with two bugs caught by adversarial verification. Consistent with all prior internal-data work.

## What this means / frontier
- Internal-data, long-only, spot, no-leverage move-capture is **exhausted for WEALTH** (buy-hold is the bar and
  nothing internal clears it); the harvestable products are **drawdown-preservation** (the gate / daily_engine) and
  a **modest conviction-selection refinement** (MACD/DONCHIAN tilt) — both de-risked beta.
- Open frontier (charter-DEFERRED, unchanged): **EXTERNAL event data** (Coinbase/Upbit listing announcements);
  sub-bar execution. The internal lane is honestly mapped and closed for wealth alpha.

## Repeatability
Every number is from an actual run (RWYB); engine selftests pass; artifacts under
`runs/periods/TRAIN/2020/DEEP_DIVE/dynamic_capture_*.json` + `ti_capture_*.json` + `DYNAMIC_CAPTURE_REGISTER.jsonl`
with reproducibility fingerprints; verification re-derivations under `runs/periods/ALL_WEATHER/quant_*.py`. Full
ledger: `runs/periods/TRAIN/2020/DEEP_DIVE/DYNAMIC_CAPTURE_FINDINGS.md`.
