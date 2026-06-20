# CURRENT — live state (Claude-written each cycle; committed; <2KB)

PHASE: BUILD
UPDATED: 2026-06-20

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
`docs/MA_DRAWDOWN_INSURANCE_FINDINGS.md`. (2 fabrications caught by the gates en route: +5828% MtM-look, +4854pp baseline.)
OPEN (un-verified, artifacts on disk — verify before use): `val_forward_*.json` (held-out forward windows),
`decoupled_7slice_14lb_*.json` (7d-slice/14d-lookback). AWAITING user's next instruction.

## DO NOT
Collapse to one asset. Default a single TF. Propose the short side. Stop at a "de-risked beta, done" verdict without a
constructed within-constraint book.
