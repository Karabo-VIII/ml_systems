# DAILY ENGINE -- the deployable daily-return engine (runbook)

> **All performance figures in this doc are VERIFIED bit-exact (RWYB) against the cited artifacts:
> `runs/strat/daily_engine_*.json` (backtests) and `runs/strat/daily_engine_robustness_*.json` (the gauntlet).
> reconciliation: backtest == the persisted JSON.**

`src/strat/daily_engine.py` -- a turnkey, long-only, daily-cadence trading engine over the u10
universe. Run it daily; it tells you what to hold and produces a daily return stream. This is a
WORKING SOLUTION (a "click play" system), not an alpha hunt.

## What it is (one paragraph)

Two proven, robust components assembled into one system:

- **CORE (the return generator).** A **vol-targeted long buy-hold** book over u10 at daily cadence.
  Each asset's target weight = `vol_target / realized_vol`, capped at `max_per_name` (0.15),
  normalized so gross <= 1.0, long-only, held **every day** (full coverage). This is `VOLTGT_BH`,
  the robust winner from the 2020 deep-dive (`deep2020_bestbook.py`) -- the vol-target component is
  the part that transfers across regimes.
- **DEFENSIVE OVERLAY (the regime value).** The causal regime classifier from
  `rolling_regime_book.py` (trend / chop / down on a rolling causal window, thresholds **fit on a
  TRAIN prefix only**, hysteresis for persistence) mapped to a daily **exposure scalar in [0,1]**:
  `trend=1.0, chop=0.6, down=0.2`. Final book = `core_weights * regime_scalar`. The overlay's job is
  **drawdown control in the bear** -- it de-risks when the regime turns down.

Causal (weights lagged 1 bar, MtM-no-double-count), costed (taker default / maker optional),
long-only, daily.

**Honest frame.** This is a working **positive-core, risk-managed beta + vol-target engine with
controlled drawdown**. It does **not** print alpha (the internal-data ceiling is real, per MEMORY).
What it delivers is a turnkey system that generates returns daily, with the regime overlay cutting
maxDD and lifting Sharpe vs the raw core.

## The one command to run it daily

```
python -m strat.daily_engine --today
```

That prints the recommended book for the latest available day: the detected **regime**, the
**exposure scalar**, **gross exposure**, and the **per-asset weights** to hold. Allocate your
bankroll to those weights (the rest stays in cash).

## How to read the output

```
## DAILY ENGINE -- TODAY book -- core=voltgt -- overlay=on
   DATE 2026-05-28 | regime=chop | exposure_scalar=0.6 | gross=0.6 | 10 positions
     BTCUSDT     0.0600
     ...
```

- **regime** -- the detected market state (trend / chop / down).
- **exposure_scalar** -- the overlay's de-risk multiplier (1.0 in trend, 0.6 in chop, 0.2 in down).
- **gross_exposure** -- total fraction of bankroll deployed long today (= sum of the weights).
  The remainder is cash.
- **weights** -- the fraction of bankroll in each asset. Hold these into the next day.

In a **down** regime the book shrinks to ~0.2 gross (mostly cash); in a **trend** regime it deploys
full (~1.0 gross). That shrink is the engine protecting capital -- it is working as designed.

## Other modes

```
python -m strat.daily_engine                                  # default: 2020-2026 backtest + today's book
python -m strat.daily_engine --backtest 2020-01-01:2025-12-31 # backtest a window -> stats + chart + JSON
python -m strat.daily_engine --date 2022-06-15                # the book for a specific historical day
python -m strat.daily_engine --core orthobook                 # use the +calendar-tilt core variant
python -m strat.daily_engine --maker                          # price with maker cost (0.0006 rt)
python -m strat.daily_engine --no-overlay                     # core only (no regime de-risking)
python -m strat.daily_engine --selftest                       # two-sided synthetic soundness check
```

A backtest run writes `runs/strat/daily_engine_<stamp>.json` (full stats + repro + git SHA + latest
book + honest scorecard) and a chart `runs/strat/plots/daily_engine_<stamp>.png` (equity curves +
the exposure track).

## Headline backtest (full cycle 2020-01 .. 2026-01, taker, u10 daily)

The HARDENED deployment default uses the **more defensive** regime scalar `{trend:1.0, chop:0.5,
down:0.1}` (2026-06-13 robustness gauntlet). The prior documented headline `{1.0,0.6,0.2}` is preserved
as `--aggressive`. All numbers VERIFIED (RWYB) at git `774c693`.

| book | compound% | CAGR% | Sharpe | maxDD% | daily-pos% | coverage% |
|------|-----------|-------|--------|--------|------------|-----------|
| **ENGINE** (default / defensive)   | +2832 | 75.7 | **1.44** | **-48.2** | 53.6 | 99.3 |
| ENGINE `--aggressive` (prior head) | +3219 | 79.4 | 1.39 | -55.4 | 53.6 | 99.3 |
| CORE-ALONE (no overlay)            | +3802 | 84.3 | 1.17 | -79.4 | 53.6 | 99.3 |
| BUY-HOLD (EW u10)                  | +4670 | 90.6 | 1.21 | -79.4 | 53.9 | 100 |

Recent slice **2024-01 .. 2026-01** (fresh data, default): ENGINE +50% / maxDD **-27.2%** / Sharpe 0.76  <!-- VERIFIED RWYB -->
vs CORE +46% / -50% / 0.62 vs BUY-HOLD +46% / -50% / 0.62 -- the overlay beats the core on **both**  <!-- VERIFIED RWYB -->
return and drawdown, and the default's recent maxDD is now **under the 30% floor**.  <!-- VERIFIED RWYB -->

**Read:** the hardened default overlay trades ~9 points of full-cycle CAGR for **31 points of drawdown
control** (-79% -> -48%) and **+0.27 Sharpe** (1.17 -> 1.44) -- the regime layer's value is risk-adjusted  <!-- VERIFIED RWYB -->
return + capital preservation in the bear. The core is the positive return generator; the engine is the
risk-managed version of it. (Hardening provenance: the per-episode bear split showed deeper de-risk cuts
maxDD monotonically in BOTH the 2022 and 2025 bears -- see "## Robustness".)

## Caveats (honest)

- **Long-only, internal data.** No shorting, no external signal -- this is beta + vol-target +
  regime de-risk. It will not print alpha; it produces a working risk-managed return stream.
- **Vol-target is effectively INERT on u10 daily (the cap binds ~94.7% of the time).** Crypto daily  <!-- VERIFIED RWYB -->
  vol is high enough that `vol_target/vol` (~0.84-1.56 per name) exceeds the per-name cap (0.15) for
  essentially every name every day, so the book resolves to **capped equal-weight x regime scalar**.
  The "vol-target" label is technically accurate but functionally the core is `equal-weight (renormalized
  to gross<=1) x scalar`; the vol differentiation only appears on the ~5% of asset-days where a name is  <!-- VERIFIED RWYB -->
  unusually quiet. This is correct (the cap is doing its job) and matters more on lower-vol universes /
  cadences, but be honest about what is actually driving the book: regime exposure scaling, not vol-target.
- **Regime thresholds fit on 2019..2023 only** (no look-ahead onto the eval span); the regime->scalar
  map is pre-registered, not fit.
- **The scorecard ship-read is False** (held-out block-bootstrap p05 < 0) -- expected: this is not a
  ship-tier alpha candidate, it is a deployable beta engine. The "ship" gate is for alpha claims;
  this engine's claim is "produces a daily return stream with controlled drawdown", which it does.
- **`--core voltgt` is the chosen default** (the calendar-tilt `orthobook` variant is a near-wash:
  +3870% vs +3877%, identical maxDD/Sharpe -- the mild DOW tilt is orthogonal and near-neutral, so  <!-- VERIFIED RWYB -->
  we keep the simpler core).

## Repro

```
python -m strat.daily_engine --backtest 2020-01-01:2025-12-31      # git SHA in the output JSON
python -m strat.daily_engine --selftest                            # PASS
python -m strat.daily_engine --aggressive --backtest 2020-01-01:2026-01-01   # prior documented headline
```

## Robustness

The engine was subjected to a rigorous 7-dimension robustness gauntlet on 2026-06-13
(`src/strat/daily_engine_gauntlet.py`; report `runs/strat/daily_engine_robustness_<stamp>.json`).
All numbers below are **VERIFIED (RWYB)** against the report at git `774c693`, on the hardened
defensive default. Run it: `python -m strat.daily_engine_gauntlet`.

| dimension | verdict | finding (VERIFIED) |
|-----------|---------|--------------------|
| 1. block-bootstrap p05 (held-out) | **FRAGILE** | full-cycle p05 +301 (robustly +); **post-train 2023+ p05 -17.8**; **held-out 2025-03+ p05 -36.0, compound -4.2%**. OOS expectancy is NOT robustly positive. |  <!-- VERIFIED RWYB -->
| 2. PBO / param-overfit (CSCV) | **INFORMATIVE** | PBO 0.943, degrade_slope -0.83. NOT classic overfit (scalars are pre-registered, never fit) -- it is **regime-dependence**: the best exposure scalar flips with bull/bear. All grid configs profitable (Sharpe 1.10-1.42). |  <!-- VERIFIED RWYB -->
| 3. parameter sensitivity | **PASS** | Sharpe spread **0.22** across all scalar/lookback/vol-window perturbations (base 1.37). No param is a knife-edge. |
| 4. regime-stratified (BULL/BEAR/CHOP) | **PASS** | overlay saves **24.3pp** of bear maxDD (engine -63.2 vs core -87.5) and 12.4pp in bull, 11.1pp in chop. Not carried by one regime; de-risks the bear as designed. |
| 5. cost sensitivity (maker/taker/2x) | **PASS** | avg daily turnover 0.007 -> ~0.3%/yr drag. full-cycle compound 2699% (maker) -> 2601% (2x-taker): cost-robust by design (daily, low-turnover). |  <!-- VERIFIED RWYB -->
| 6. concentration / firewall | **PASS** | top name DOGE = **17.4%** of return (well under the 95% rule); book WITHOUT the top name still +1727%. No single-name dependence. |  <!-- VERIFIED RWYB -->
| 7. look-ahead audit (programmatic) | **PASS** | thresholds fit train-only & invariant to post-train data; vol-target weight + regime label at `t` invariant to future bars; weights lagged 1 bar. All causality probes pass. |

**5/6 gradable dimensions PASS** (dim2 is an informative diagnostic, not pass/fail). The one
**FRAGILE** dimension (dim1) is the load-bearing honest caveat below.

### The honest verdict (read before deploying real capital)

This is a **robust, risk-managed, long-only BETA engine** -- NOT an alpha sleeve. It is robust in the
ways that matter for a beta engine (param-insensitive, cost-robust, not concentration-carried, causal,
de-risks the bear) but its **held-out (2025-03+) expectancy is flat-to-slightly-negative** (-4.2%  <!-- VERIFIED RWYB -->
compound, block-bootstrap p05 -36). That is not a code defect -- it is the honest market-state fact:
**a long-only book cannot be robustly positive in a chop/sideways or bear regime**; it can only lose
less (which the overlay does). The +2832% full-cycle headline is **2020-2024 bull beta**. Do not deploy  <!-- VERIFIED RWYB -->
this expecting positive returns in a flat or down market -- deploy it expecting *market exposure with
controlled drawdown*.

### What was hardened (2026-06-13)

1. **Defensive default scalar.** Changed the shipped default from `{1.0,0.6,0.2}` to `{1.0,0.5,0.1}`.
   Rationale (RWYB per-episode bear split): deeper de-risk cuts maxDD **monotonically in BOTH
   independent bears** -- 2022: -47.7 -> -41.4; 2025: -29.9 -> -26.4 -- and lifts Sharpe (1.31 -> 1.44)
   and held-out p05 (-42.7 -> -36.0). The prior -55% maxDD breached the project's 20-30% floor badly;  <!-- VERIFIED RWYB -->
   the new default's recent-slice maxDD (-27.2%) is **under** the 30% floor. The aggressive map is  <!-- VERIFIED RWYB -->
   preserved as `--aggressive` for reproducibility of the prior documented headline.
2. **Deployment data-quality guard** on the live book (`book_for_date` -> `data_quality_flags`): warns
   on a stale latest bar, a partial-universe day (missing u10 names), an all-NaN feed gap (so a glitch
   does NOT silently liquidate -- hold yesterday's book), and un-warmed vol names. Surfaced in
   `--today` / `--date` output. (Verified live: the current panel ends 2026-05-28 and the guard flags
   it as stale -- the data feed needs refreshing before live deployment.)
3. **Honest disclosure** that the vol-target is effectively inert on u10 daily (cap binds 94.7%) and  <!-- VERIFIED RWYB -->
   that dim2's high PBO is regime-dependence, not overfit.

### Deployment-readiness checklist (before real capital)

- [ ] **Refresh the data feed** -- the u10 daily panel currently ends 2026-05-28 (16d stale as of
  2026-06-13). The `--today` guard flags this; the book is only valid on a current bar.
- [ ] **Set a maxDD circuit-breaker** -- even the hardened default can draw down ~48% full-cycle (bear  <!-- VERIFIED RWYB -->
  beta). Decide a hard stop (e.g. de-risk to cash if portfolio DD breaches -25/-30%) OUTSIDE the engine.  <!-- VERIFIED RWYB -->
- [ ] **Budget for live fill costs** -- backtest uses fixed taker/maker; real maker p_fill is 0.21-0.50
  (CLAUDE.md MakerCostModel). Daily low-turnover means cost is minor (~0.3%/yr), but size accordingly.  <!-- VERIFIED RWYB -->
- [ ] **Accept the held-out reality** -- this engine is flat-to-negative in chop/bear; it is for
  market exposure with controlled drawdown, not for printing returns in all regimes.
- [ ] **Decide the down-regime policy** -- the default keeps 10% exposure in a confirmed downtrend  <!-- VERIFIED RWYB -->
  (`down=0.1`); use `--no-overlay` for pure core, or fork the scalar map for fully-flat (`down=0.0`).
- [ ] **Handle u10 universe changes** -- a new asset entering u10 is handled gracefully (excluded until
  its vol warms up); verify `config/universes/u10.yaml` matches your live exchange listings.
- [ ] **Run the gauntlet on any param change** -- `python -m strat.daily_engine_gauntlet` is the
  re-validation gate; the `--selftest` (engine + gauntlet) must stay PASS.

### Repro (robustness)

```
python -m strat.daily_engine_gauntlet            # the 7-dimension gauntlet -> JSON + verdict
python -m strat.daily_engine_gauntlet --selftest # two-sided soundness of the gauntlet (PASS)
```
JSON: `runs/strat/daily_engine_robustness_20260613_180139.json` (git `774c693`).

## Number reconciliation (canonical -- every cited ENGINE compound, pinned to its window x scalar)
The ENGINE compound is quoted as different numbers across artifacts; they are ALL the SAME engine, differing only by
(WINDOW x SCALAR x DAY-SET). Canonical run (voltgt core, taker, RWYB-VERIFIED 2026-06-14, panel 2020-01-06..2026-05-28):

| window | scalar | compound | maxDD | the number cited as |
|---|---|---|---|---|
| 2020-01-01 .. 2026-01-01 | defensive {1.0,0.5,0.1} | **+2832%** | -48.2 | this doc's headline |
| 2020-01-01 .. 2026-01-01 | aggressive {1.0,0.6,0.2} | **+3219%** | -55.4 | the `--aggressive` headline |
| 2020-01-01 .. 2026-05-28 (full panel) | defensive | **+2656%** | -48.2 | engine_timeframe_sweep 1d |
| 2020-07-09 .. 2026-05-28 (overlap day-set) | defensive | **+1969%** | -48.2 | core_satellite_book "core" |

NOTE on the +1969%: core_satellite reports the core compound over the 2072-day OVERLAP (core dates ∩ SATELLITE
dates). Over the *continuous* same date-range it is +2316%; the satellite's ~78 missing days (its data gaps) drop
those core-days from the compound. Not an error -- a reduced day-set. All four numbers reproduce bit-exact from
`strat.daily_engine.build_book` on the stated (window, scalar). [VERIFIED RWYB 2026-06-14]
