# CURRENT — live state (Claude-written each cycle; committed; <2KB)

PHASE: BUILD (V1.1 fine-TF campaign LIVE)
UPDATED: 2026-06-21

## CURRENT GOAL
Build the ability to **trade ANY SLICE** — any setup from any asset across the **u100 universe** — compounding, with
~100% coverage of **2%+ daily moves**, a **7-day lookback** window (a known prior weakness to fix), reusing the whole
project's move-catch infra. This is the live acid test: it must NOT collapse to "trade BTC only."

## SURFACE TO COVER (all four, or the null is not earned)
- [ ] UNIVERSE: >=20 distinct u100 symbols touched in executed code (not BTC-only).
- [ ] WINDOW: 7d lookback as a real executed parameter.
- [ ] COVERAGE: the 2%+ daily-move set enumerated/measured (a coverage manifest: realized vs available capture).
- [ ] INFRA: extend `src/strat/move_catch_book.py` / `oracle_capture_lab.py` / chimera loader — not a fresh BTC script.
  - For **MA/TI** work specifically: canonical configs = the BAND grid at `runs/periods/TRAIN/2020/DEEP_DIVE/config_leaderboard.json` (+ `robust_ma_runners.json`), via `src/strat/{ma_2020_config_leaderboard,robust_ma_runners,working_band_rolling}.py`. Use the fixed-EW ENSEMBLE of `band.band_configs` per (type,TF), NEVER `ranked[0]` (hindsight). WEALTH leg must run u100 via `portfolio_replay.run(universe="u100")` (the u10-hardwired sleeve is a breadth trap).

## STANDING TRUTH (the floor, not the verdict)
Internal long-only TI work has repeatedly landed on **de-risked beta** (preserves bears, doesn't beat buy-hold on net).
That is the FLOOR to build within, not a stop. The genuinely-untested corners: the move-CAPTURE-RATE lens, dynamic
regime->config-set switching, chimera x TI conditioning, the ironed FINE-TF band (cost doesn't bind a properly-ironed
fine-TF band; under-participation does). The held-out POSITIVE so far is funding-dispersion carry (dollar-neutral perps;
boxed pending the user). External-listing-data is the open user-gated frontier.

## NEXT ACTIONS (MA engine build)
**1st VERIFIED SUCCESS — committed efb23a4b (2026-06-20).** MA drawdown-insurance sleeve at **1d/4h** over u100:
HALVES portfolio maxDD (~-35% vs HODL -82%) via GENUINE same-exposure timing skill (+6..15.5pp beyond a phi-matched
churn-immune dumb book, z up to +3.5), robust across 4 selloffs (COVID/May-2021/2022/2023-chop), all 8 MA types.
NOT alpha (zero up-mkt timing alpha) = capital-preservation within long-only (the charter's honest fallback, now
EARNED). 15m BROKEN. Aggressively red-teamed GO; leak-free; survivorship widens the gap. Doc:
`docs/engines/V1_ma_drawdown_insurance.md` (index `docs/engines/ENGINE_BUILDS.md`); artifacts:
`runs/engines/V1_ma_drawdown_insurance/` (engines/ verification/ step1/ slices/ outoffit/ = VERIFIED; README.md manifest).
(2 fabrications caught by the gates en route: +5828% MtM-look, +4854pp baseline.)
OPEN (un-verified, artifacts on disk — verify before use):
`runs/engines/V1_ma_drawdown_insurance/open_unverified/` — `val_forward_*.json` (held-out forward windows),
`decoupled_7slice_14lb_*.json` (7d-slice/14d-lookback).

**V1 DOCUMENTED + REORGANIZED — committed dd1c7b8f (2026-06-21).** Out-of-fit (2021→2024-05, 2020 excluded):
maxDD-halving HOLDS in all 6/6 cells (39-55pp cut); HMA|4h compound flips below HODL → compound is NEVER the headline.

**V1.1 FINE-TF CAMPAIGN — HONEST NULL + a real leak FIX (2026-06-21).** Goal was: make 2h/1h/30m (salvage 15m)
profitable in 7d slices via an exit harness + mechanical interventions. Built `src/strat/exit_harness.py` (8 exits +
entry de-bounce + HTF-confirm) + a two-sided eval window on `run_wealth` (2022+ loader-sealed). 2020 sweep + 2021
iron looked like ALL 16 fine-TF cells FIXED/HOLD — **but the red-team caught a LOOK-AHEAD LEAK**: `htf_entry_gate`'s
`align_htf_on` aligned on coarser START-labels → read the not-yet-closed coarse bar (HMA|1h +879%→+109% de-leaked).
**FIXED** (close-time alignment; selftest causal; verified vs auditor). HONEST de-leaked verdict: fine-TF exit-harness
is **DOMINATED by the coarse engine on net 0/16**, loses to always-long slice7d 15/16; only drawdown-insurance
survives (redundant with V1). **EARNED NULL = D81; deployable scope STAYS 1d/4h (V1, unaffected).** 3rd fabrication
caught by the operating model. Doc: `docs/engines/V1.1_finetf_exits.md`; artifacts `runs/engines/V1.1_finetf_exits/`
(deleaked_2021/ = honest; sweep_2020/sweep_2021/ = leak-inflated, caveat attached). Ledger: CAMPAIGN_LEDGER.md.

## DO NOT
Collapse to one asset. Default a single TF. Propose the short side. Stop at a "de-risked beta, done" verdict without a
constructed within-constraint book.
