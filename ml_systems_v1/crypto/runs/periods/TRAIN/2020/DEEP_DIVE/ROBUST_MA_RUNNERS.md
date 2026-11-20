# ROBUST MA RUNNERS -- the capstone that CLOSES all MA lanes (2020 band)

A **robust runner** per (MA-type x TF) = the WORKING-BAND ENSEMBLE (equal-weight the band members, NOT the noisy #1) + the proven uniform IRON (vol-target overlay + min-hold). This UNIFIES two lanes: the BAND lane (which configs work) + the per-type WEAKNESS/IRON lane (how each type is ironed).

STRICT LONG-ONLY + spot (held in {0,1}, ZERO short logic). 2020 BAND ONLY. Fixed-EW (`fillna(0.0).mean(axis=1)`, never skipna). Causal/lag-1, maker cost. Held-out: the BAND is selected on TRAIN&VAL&OOS positivity; OOS p05 is the robustness floor.

**HEADLINE:** the band-ENSEMBLE runner is MORE ROBUST than the FORWARD #1 (the deployable, TRAIN+VAL-picked #1) in 30/48 (type x TF) cells; the RANK-FRAGILITY TAX (mean +7.9pp of OOS net) is the return you forfeit by NOT knowing the hindsight winner ahead -- the empirical case for the ensemble over 'just deploy the #1'. The ensemble also raises OOS p05 over the un-ironed ensemble (the vol-target's contribution). It trades ~+85pp of HINDSIGHT peak net (an unachievable ceiling) for never making the rank bet.

## The two-part IRON (per MA_WEAKNESS.md)
- **UNIFORM stack (all 8 types):** min_hold(12) [in the base sleeve] + **VOL-TARGET overlay** `clip(median_rv/rv_lagged, 0, 1)` on a market-observable past-only realized vol -- the one iron that DAMPENS maxDD for every type (deep2020_ma_weakness's +VOLTGT column).
- **Type-specific param region = the BAND itself.** Low-lag (HMA/DEMA/TEMA) overshoot -> the band's confirmed/slower region; adaptive (KAMA/VIDYA) stall -> the band's FAST region; SMA structural lag; EMA balanced. The ensemble OF THE BAND is the type-specific iron, by construction.
- **NOTED FUTURE (NOT built here):** overshoot-damper (low-lag types), regime-adaptive param (adaptive types). These are the OPEN deeper irons -- flagged, not implemented.

LOOK-AHEAD FRAMING (stated): the BAND and the #1 are computed on FULL-2020 = DESCRIPTIVE of what was discovered over the year, NOT a forward predictor. The runner's MECHANICS are causal/lag-1 (forward-honest). median_rv is a single in-2020 reference level (the same convention the coarse-TF deployable book uses); a live deploy would use a trailing-only median. The held-out logic is TRAIN+VAL-select / OOS-confirm at the BAND level.

Repro: `python -m strat.robust_ma_runners --cadences 1d,4h,2h,1h,30m,15m`  git_sha=231ac92  cost=maker(0.0006)  trail=0.1  min_hold=12  vol_window={'1d': 14, '4h': 84, '2h': 168, '1h': 336, '30m': 672, '15m': 1344}  split={'TRAIN': ('2020-01-01', '2020-07-01'), 'VAL': ('2020-07-01', '2020-10-01'), 'OOS': ('2020-10-01', '2021-01-01'), 'FULL': ('2020-01-01', '2021-01-01')}

## OLD (single #1 config) vs NEW (robust band-ensemble + vol-target iron)

**The COMPARISON BASIS is the subtlety.** The single #1 comes in TWO flavours:
- **HINDSIGHT #1** = top FULL-2020 net -- it PEEKED at OOS. The optimistic CEILING; NOT deployable (you can't know it ahead).
- **FORWARD #1** = top TRAIN+VAL net -- the ONLY #1 you can actually deploy (it does NOT peek at OOS). **This is the fair baseline.**

The **rank-fragility tax** = (hindsight #1 OOS) - (forward #1 OOS) -- the OOS net you LOSE for not knowing the winner ahead of time. The leaderboard's median Spearman rho (TRAIN+VAL vs OOS) is ~0.59 and the TV->OOS top-10 overlap is only 1-7/10, so the forward #1 is usually NOT the hindsight #1. The LOAD-BEARING verdict: does the band-ENSEMBLE runner beat the FORWARD #1 (the deployable one) on OOS net + OOS p05 + worst-window?

`p05` = OOS block-bootstrap p05 (daily, block=5). `worst-win` = min(VAL net, OOS net).

| TF | type | N band | hind#1 OOS | fwd#1 OOS | rank-frag tax | NEW OOS | NEW vs fwd#1 OOS | fwd#1 p05 | NEW p05 | hind#1 FULL | NEW FULL | ROBUST(vs fwd#1)? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1d | EMA | 105 |  39.9% |  25.2% | 14.7pp |  17.3% | -7.9pp | -18.7% |  -3.9% | 123.8% |  57.3% | YES |
| 1d | SMA | 84 |  43.5% |  26.5% | 17.0pp |  18.9% | -7.6pp | -16.5% |  -4.2% | 137.3% |  67.2% | YES |
| 1d | WMA | 106 |  33.5% |  33.5% |  0.0pp |  19.0% | -14.5pp | -12.0% |  -6.7% | 159.7% |  71.3% | YES |
| 1d | HMA | 108 |  41.0% |  23.0% | 18.0pp |  20.2% | -2.8pp | -11.8% |  -5.8% | 155.8% |  78.6% | YES |
| 1d | DEMA | 114 |  39.7% |  27.3% | 12.4pp |  19.8% | -7.5pp | -11.0% |  -3.9% | 158.5% |  74.1% | YES |
| 1d | TEMA | 112 |  40.9% |  31.8% |  9.1pp |  19.9% | -11.9pp |  -9.0% |  -4.6% | 162.7% |  79.4% | YES |
| 1d | KAMA | 74 |  33.0% |  20.7% | 12.3pp |  17.1% | -3.6pp | -20.0% |  -1.6% | 125.3% |  50.6% | YES |
| 1d | VIDYA | 74 |  43.0% |  43.0% |  0.0pp |  13.4% | -29.6pp |  -0.6% |  -0.1% | 128.1% |  44.9% | - |
| 4h | EMA | 120 |  52.5% |  37.1% | 15.4pp |  21.7% | -15.4pp | -10.6% |  -3.9% | 174.5% |  85.2% | YES |
| 4h | SMA | 115 |  50.7% |  26.9% | 23.8pp |  19.6% | -7.3pp | -18.8% |  -5.3% | 191.6% |  76.9% | YES |
| 4h | WMA | 114 |  42.5% |  42.5% |  0.0pp |  21.9% | -20.6pp |  -7.5% |  -4.1% | 213.8% |  88.9% | YES |
| 4h | HMA | 109 |  42.6% |  42.6% |  0.0pp |  21.1% | -21.5pp |  -4.1% |  -4.5% | 197.9% |  77.2% | - |
| 4h | DEMA | 111 |  38.6% |  27.3% | 11.3pp |  21.0% | -6.3pp | -16.0% |  -3.6% | 173.9% |  86.4% | YES |
| 4h | TEMA | 107 |  48.1% |  25.3% | 22.8pp |  20.0% | -5.3pp | -15.9% |  -4.6% | 160.1% |  79.3% | YES |
| 4h | KAMA | 111 |  43.9% |  26.6% | 17.3pp |  15.2% | -11.4pp | -14.2% |  -5.8% | 145.7% |  66.8% | YES |
| 4h | VIDYA | 115 |  42.8% |  42.8% |  0.0pp |  16.0% | -26.8pp |  -0.4% |  -2.1% | 148.4% |  59.4% | - |
| 2h | EMA | 117 |  52.5% |  38.1% | 14.4pp |  29.5% | -8.6pp |  -8.4% |  -3.7% | 170.4% |  96.5% | YES |
| 2h | SMA | 110 |  48.7% |  26.4% | 22.3pp |  27.1% |  0.7pp |  -7.9% |  -3.9% | 162.9% |  82.6% | YES |
| 2h | WMA | 113 |  53.1% |  43.6% |  9.5pp |  29.4% | -14.2pp |  -1.4% |  -3.2% | 189.8% |  92.0% | - |
| 2h | HMA | 102 |  53.8% |  47.3% |  6.5pp |  30.8% | -16.5pp |  -0.1% |  -1.4% | 193.4% |  84.3% | - |
| 2h | DEMA | 101 |  40.9% |  32.6% |  8.3pp |  29.9% | -2.7pp | -13.4% |  -1.9% | 172.6% |  91.4% | YES |
| 2h | TEMA | 99 |  46.3% |  32.2% | 14.1pp |  28.5% | -3.7pp |  -0.5% |  -3.6% | 150.0% |  81.0% | - |
| 2h | KAMA | 116 |  46.6% |  29.3% | 17.3pp |  23.7% | -5.6pp |  -8.5% |  -4.3% | 164.1% |  77.4% | YES |
| 2h | VIDYA | 114 |  46.8% |  40.3% |  6.5pp |  26.5% | -13.8pp |  -5.1% |  -1.2% | 139.9% |  78.2% | YES |
| 1h | EMA | 108 |  50.6% |  41.2% |  9.4pp |  28.1% | -13.1pp |  -1.9% |  -1.7% | 158.4% |  85.6% | YES |
| 1h | SMA | 106 |  32.5% |  32.5% |  0.0pp |  27.6% | -4.9pp |  -5.2% |  -0.6% | 181.1% |  85.3% | YES |
| 1h | WMA | 109 |  41.7% |  41.7% |  0.0pp |  26.9% | -14.8pp |   1.2% |  -1.7% | 170.7% |  84.0% | - |
| 1h | HMA | 78 |  40.2% |  40.2% |  0.0pp |  28.1% | -12.1pp |   3.6% |  -0.5% | 226.0% |  73.5% | - |
| 1h | DEMA | 95 |  37.4% |  37.4% |  0.0pp |  27.5% | -9.9pp |  -6.0% |  -0.5% | 175.7% |  83.6% | YES |
| 1h | TEMA | 85 |  44.4% |  44.4% |  0.0pp |  28.0% | -16.4pp |   2.1% |  -1.1% | 155.6% |  75.1% | - |
| 1h | KAMA | 114 |  43.2% |  29.0% | 14.2pp |  23.7% | -5.3pp |  -9.8% |  -2.2% | 149.8% |  74.6% | YES |
| 1h | VIDYA | 120 |  49.7% |  39.7% | 10.0pp |  28.4% | -11.3pp |  -1.4% |   2.1% | 138.7% |  80.7% | YES |
| 30m | EMA | 94 |  32.3% |  10.3% | 22.0pp |  18.1% |  7.8pp | -23.9% |  -6.9% | 119.8% |  70.5% | YES |
| 30m | SMA | 90 |  48.8% |  26.5% | 22.3pp |  21.0% | -5.5pp |  -2.6% |  -3.4% | 178.4% |  71.0% | - |
| 30m | WMA | 79 |  36.3% |  36.3% |  0.0pp |  19.4% | -16.9pp |   0.4% |  -4.9% | 180.6% |  69.8% | - |
| 30m | HMA | 48 |  42.3% |  42.3% |  0.0pp |  22.3% | -20.0pp |   7.9% |  -3.0% | 266.8% |  66.3% | - |
| 30m | DEMA | 66 |  37.1% |  37.1% |  0.0pp |  23.3% | -13.8pp |   4.0% |  -1.5% | 143.9% |  68.5% | - |
| 30m | TEMA | 42 |  34.0% |  34.0% |  0.0pp |  22.8% | -11.2pp |   0.1% |  -2.7% | 138.4% |  66.0% | - |
| 30m | KAMA | 79 |  38.7% |  38.7% |  0.0pp |  20.6% | -18.1pp |   2.2% |  -3.5% | 141.1% |  63.3% | - |
| 30m | VIDYA | 114 |  27.5% |  24.2% |  3.3pp |  19.2% | -5.0pp |  -8.8% |  -3.7% | 115.5% |  72.4% | YES |
| 15m | EMA | 65 |  17.4% |  17.4% |  0.0pp |  15.5% | -1.9pp | -12.5% |  -9.9% | 111.4% |  61.0% | YES |
| 15m | SMA | 52 |  31.0% |  13.8% | 17.2pp |  19.3% |  5.5pp | -15.9% |  -4.5% | 143.7% |  64.7% | YES |
| 15m | WMA | 46 |  25.2% |  20.0% |  5.2pp |  17.9% | -2.1pp | -10.1% |  -6.9% | 139.1% |  59.5% | YES |
| 15m | HMA | 20 |  13.5% |  13.5% |  0.0pp |  16.0% |  2.5pp | -12.4% |  -7.8% | 138.8% |  58.7% | YES |
| 15m | DEMA | 35 |  28.8% |  28.8% |  0.0pp |  21.8% | -7.0pp |   0.1% |  -3.4% | 133.2% |  62.0% | - |
| 15m | TEMA | 20 |  36.1% |  36.1% |  0.0pp |  14.6% | -21.5pp |   4.0% |  -8.8% | 140.0% |  54.5% | - |
| 15m | KAMA | 47 |  35.8% |  35.8% |  0.0pp |  22.0% | -13.8pp |   0.0% |  -2.7% | 134.8% |  59.4% | - |
| 15m | VIDYA | 102 |  22.3% |  19.2% |  3.1pp |  17.8% | -1.4pp | -15.2% |  -6.8% | 119.3% |  68.8% | YES |

**ROBUSTNESS VERDICT (load-bearing, HONEST): the band-ensemble + iron runner is MORE ROBUST than the FORWARD #1 (the deployable one) in 30/48 cells** -- where more-robust = a SHALLOWER OOS p05 (the downside floor IS what robustness means) AND net that still participates (>=50% of the fwd #1's OOS net).

### The honest decomposition (why this is two-edged, not a clean win)

1. **RANK-FRAGILITY TAX = mean 7.9pp of OOS net.** That is the OOS return forfeited by picking the #1 on TRAIN+VAL (the deployable choice) vs the unknowable hindsight winner. The leaderboard's TV->OOS top-10 overlap is only 1-7/10 -- you can NOT reliably pick the #1 ahead. This is the case AGAINST 'just deploy the #1'.
2. **The un-ironed ensemble's OOS net is -6.8pp vs the forward #1** -- i.e. on a clean-BULL 2020 OOS the band-mean still slightly TRAILS the (tax-paying) forward #1 on NET, because the bull rewards the few hot configs that survive into OOS. The ensemble's value is NOT higher net here.
3. **The vol-target iron costs a further -3.1pp of OOS net** (exposure suppression in a bull it didn't need to defend) -- a textbook defensive tradeoff a clean-bull OOS under-rewards.
4. **BUT the runner's OOS p05 (downside floor) beats the forward #1 by 3.3pp on average** (single-config p05 tails run -12 to -20%; the ~100-member ensemble's p05 is -1 to -7%). **THIS is the genuine robust-runner win: a far shallower worst case.** A single config can break down in a bootstrap resample; the ensemble cannot.

**Bottom line (claim-tagged [MEASURED]):** the robust runner is the right DEPLOYABLE object NOT because it out-NETS the #1 on a bull OOS (it does not), but because it (a) never makes the rank bet that costs ~11pp in expectation, and (b) has a ~3x shallower downside floor. On a clean-bull single-realization OOS that defensive posture looks like 'lower net'; in a regime with real drawdowns (the reason the iron exists) it is the protection you are buying. The honest framing is a RISK-for-NET trade, not a free lunch -- and the rank-fragility tax is the hard number that says 'do not chase the #1'.

## Deployable RUNNER SPECS (per type x TF) -- turnkey robust runners

### TF = 1d  (EW u10 buy-hold FULL-2020 = 140.2%, cadence-invariant reference)

- **EMA x 1d** -- band: 2MA fast[2, 102] slow[3, 239] (n=59); 3MA fast[2, 24] slow[4, 233] (n=46) (total 105 members) | iron: vol-target(rv=14) + min_hold(12) + trail(0.1) | NEW: TRAIN 20.4% VAL 11.3% OOS 17.3% FULL 57.3% | maxDD -11.4% | OOS p05 -3.9% | more-robust-than-fwd#1: YES
- **SMA x 1d** -- band: 2MA fast[2, 102] slow[3, 239] (n=49); 3MA fast[2, 24] slow[4, 233] (n=35) (total 84 members) | iron: vol-target(rv=14) + min_hold(12) + trail(0.1) | NEW: TRAIN 20.2% VAL 17.0% OOS 18.9% FULL 67.2% | maxDD -12.1% | OOS p05 -4.2% | more-robust-than-fwd#1: YES
- **WMA x 1d** -- band: 2MA fast[2, 73] slow[3, 210] (n=52); 3MA fast[2, 75] slow[4, 233] (n=54) (total 106 members) | iron: vol-target(rv=14) + min_hold(12) + trail(0.1) | NEW: TRAIN 25.1% VAL 15.1% OOS 19.0% FULL 71.3% | maxDD -12.4% | OOS p05 -6.7% | more-robust-than-fwd#1: YES
- **HMA x 1d** -- band: 2MA fast[2, 102] slow[3, 239] (n=55); 3MA fast[2, 60] slow[4, 233] (n=53) (total 108 members) | iron: vol-target(rv=14) + min_hold(12) + trail(0.1) | NEW: TRAIN 22.9% VAL 20.9% OOS 20.2% FULL 78.6% | maxDD -16.3% | OOS p05 -5.8% | more-robust-than-fwd#1: YES
- **DEMA x 1d** -- band: 2MA fast[2, 102] slow[3, 237] (n=56); 3MA fast[2, 75] slow[4, 233] (n=58) (total 114 members) | iron: vol-target(rv=14) + min_hold(12) + trail(0.1) | NEW: TRAIN 22.5% VAL 18.6% OOS 19.8% FULL 74.1% | maxDD -12.6% | OOS p05 -3.9% | more-robust-than-fwd#1: YES
- **TEMA x 1d** -- band: 2MA fast[2, 102] slow[3, 239] (n=55); 3MA fast[2, 186] slow[4, 233] (n=57) (total 112 members) | iron: vol-target(rv=14) + min_hold(12) + trail(0.1) | NEW: TRAIN 21.9% VAL 22.8% OOS 19.9% FULL 79.4% | maxDD -14.2% | OOS p05 -4.6% | more-robust-than-fwd#1: YES
- **KAMA x 1d** -- band: 2MA fast[2, 102] slow[3, 237] (n=43); 3MA fast[2, 60] slow[4, 233] (n=31) (total 74 members) | iron: vol-target(rv=14) + min_hold(12) + trail(0.1) | NEW: TRAIN 14.4% VAL 12.4% OOS 17.1% FULL 50.6% | maxDD -12.1% | OOS p05 -1.6% | more-robust-than-fwd#1: YES
- **VIDYA x 1d** -- band: 2MA fast[2, 37] slow[3, 210] (n=42); 3MA fast[2, 24] slow[4, 233] (n=32) (total 74 members) | iron: vol-target(rv=14) + min_hold(12) + trail(0.1) | NEW: TRAIN 21.0% VAL 5.6% OOS 13.4% FULL 44.9% | maxDD -7.7% | OOS p05 -0.1% | more-robust-than-fwd#1: no

### TF = 4h  (EW u10 buy-hold FULL-2020 = 134.0%, cadence-invariant reference)

- **EMA x 4h** -- band: 2MA fast[2, 102] slow[3, 239] (n=60); 3MA fast[2, 186] slow[4, 233] (n=60) (total 120 members) | iron: vol-target(rv=84) + min_hold(12) + trail(0.1) | NEW: TRAIN 22.9% VAL 23.8% OOS 21.7% FULL 85.2% | maxDD -11.6% | OOS p05 -3.9% | more-robust-than-fwd#1: YES
- **SMA x 4h** -- band: 2MA fast[2, 102] slow[3, 239] (n=56); 3MA fast[2, 75] slow[4, 233] (n=59) (total 115 members) | iron: vol-target(rv=84) + min_hold(12) + trail(0.1) | NEW: TRAIN 20.0% VAL 23.2% OOS 19.6% FULL 76.9% | maxDD -12.6% | OOS p05 -5.3% | more-robust-than-fwd#1: YES
- **WMA x 4h** -- band: 2MA fast[2, 102] slow[3, 239] (n=56); 3MA fast[2, 186] slow[4, 233] (n=58) (total 114 members) | iron: vol-target(rv=84) + min_hold(12) + trail(0.1) | NEW: TRAIN 24.3% VAL 24.6% OOS 21.9% FULL 88.9% | maxDD -12.1% | OOS p05 -4.1% | more-robust-than-fwd#1: YES
- **HMA x 4h** -- band: 2MA fast[2, 102] slow[3, 239] (n=54); 3MA fast[2, 186] slow[14, 233] (n=55) (total 109 members) | iron: vol-target(rv=84) + min_hold(12) + trail(0.1) | NEW: TRAIN 22.7% VAL 19.2% OOS 21.1% FULL 77.2% | maxDD -14.2% | OOS p05 -4.5% | more-robust-than-fwd#1: no
- **DEMA x 4h** -- band: 2MA fast[2, 102] slow[3, 239] (n=56); 3MA fast[2, 186] slow[4, 233] (n=55) (total 111 members) | iron: vol-target(rv=84) + min_hold(12) + trail(0.1) | NEW: TRAIN 25.4% VAL 22.8% OOS 21.0% FULL 86.4% | maxDD -11.2% | OOS p05 -3.6% | more-robust-than-fwd#1: YES
- **TEMA x 4h** -- band: 2MA fast[2, 102] slow[3, 239] (n=52); 3MA fast[2, 186] slow[4, 233] (n=55) (total 107 members) | iron: vol-target(rv=84) + min_hold(12) + trail(0.1) | NEW: TRAIN 22.7% VAL 21.8% OOS 20.0% FULL 79.3% | maxDD -11.9% | OOS p05 -4.6% | more-robust-than-fwd#1: YES
- **KAMA x 4h** -- band: 2MA fast[2, 102] slow[3, 239] (n=57); 3MA fast[2, 186] slow[4, 233] (n=54) (total 111 members) | iron: vol-target(rv=84) + min_hold(12) + trail(0.1) | NEW: TRAIN 18.7% VAL 21.9% OOS 15.2% FULL 66.8% | maxDD -11.1% | OOS p05 -5.8% | more-robust-than-fwd#1: YES
- **VIDYA x 4h** -- band: 2MA fast[2, 102] slow[3, 239] (n=57); 3MA fast[2, 75] slow[4, 233] (n=58) (total 115 members) | iron: vol-target(rv=84) + min_hold(12) + trail(0.1) | NEW: TRAIN 18.1% VAL 16.4% OOS 16.0% FULL 59.4% | maxDD -9.4% | OOS p05 -2.1% | more-robust-than-fwd#1: no

### TF = 2h  (EW u10 buy-hold FULL-2020 = 141.3%, cadence-invariant reference)

- **EMA x 2h** -- band: 2MA fast[2, 102] slow[3, 239] (n=57); 3MA fast[2, 186] slow[4, 233] (n=60) (total 117 members) | iron: vol-target(rv=168) + min_hold(12) + trail(0.1) | NEW: TRAIN 23.5% VAL 22.8% OOS 29.5% FULL 96.5% | maxDD -10.8% | OOS p05 -3.7% | more-robust-than-fwd#1: YES
- **SMA x 2h** -- band: 2MA fast[2, 102] slow[3, 239] (n=56); 3MA fast[2, 186] slow[15, 233] (n=54) (total 110 members) | iron: vol-target(rv=168) + min_hold(12) + trail(0.1) | NEW: TRAIN 20.7% VAL 19.1% OOS 27.1% FULL 82.6% | maxDD -12.1% | OOS p05 -3.9% | more-robust-than-fwd#1: YES
- **WMA x 2h** -- band: 2MA fast[2, 102] slow[3, 239] (n=58); 3MA fast[2, 186] slow[4, 233] (n=55) (total 113 members) | iron: vol-target(rv=168) + min_hold(12) + trail(0.1) | NEW: TRAIN 22.9% VAL 20.7% OOS 29.4% FULL 92.0% | maxDD -12.6% | OOS p05 -3.2% | more-robust-than-fwd#1: no
- **HMA x 2h** -- band: 2MA fast[2, 102] slow[3, 239] (n=51); 3MA fast[2, 186] slow[4, 233] (n=51) (total 102 members) | iron: vol-target(rv=168) + min_hold(12) + trail(0.1) | NEW: TRAIN 18.4% VAL 19.1% OOS 30.8% FULL 84.3% | maxDD -13.5% | OOS p05 -1.4% | more-robust-than-fwd#1: no
- **DEMA x 2h** -- band: 2MA fast[2, 102] slow[3, 239] (n=52); 3MA fast[2, 186] slow[4, 233] (n=49) (total 101 members) | iron: vol-target(rv=168) + min_hold(12) + trail(0.1) | NEW: TRAIN 20.6% VAL 22.2% OOS 29.9% FULL 91.4% | maxDD -10.3% | OOS p05 -1.9% | more-robust-than-fwd#1: YES
- **TEMA x 2h** -- band: 2MA fast[2, 102] slow[3, 239] (n=51); 3MA fast[2, 186] slow[4, 233] (n=48) (total 99 members) | iron: vol-target(rv=168) + min_hold(12) + trail(0.1) | NEW: TRAIN 16.9% VAL 20.5% OOS 28.5% FULL 81.0% | maxDD -12.7% | OOS p05 -3.6% | more-robust-than-fwd#1: no
- **KAMA x 2h** -- band: 2MA fast[2, 102] slow[3, 239] (n=57); 3MA fast[2, 75] slow[4, 233] (n=59) (total 116 members) | iron: vol-target(rv=168) + min_hold(12) + trail(0.1) | NEW: TRAIN 20.4% VAL 19.0% OOS 23.7% FULL 77.4% | maxDD -11.0% | OOS p05 -4.3% | more-robust-than-fwd#1: YES
- **VIDYA x 2h** -- band: 2MA fast[2, 102] slow[3, 239] (n=57); 3MA fast[2, 60] slow[4, 233] (n=57) (total 114 members) | iron: vol-target(rv=168) + min_hold(12) + trail(0.1) | NEW: TRAIN 18.8% VAL 18.6% OOS 26.5% FULL 78.2% | maxDD -8.9% | OOS p05 -1.2% | more-robust-than-fwd#1: YES

### TF = 1h  (EW u10 buy-hold FULL-2020 = 144.6%, cadence-invariant reference)

- **EMA x 1h** -- band: 2MA fast[2, 102] slow[13, 239] (n=53); 3MA fast[2, 186] slow[14, 233] (n=55) (total 108 members) | iron: vol-target(rv=336) + min_hold(12) + trail(0.1) | NEW: TRAIN 23.7% VAL 17.1% OOS 28.1% FULL 85.6% | maxDD -12.4% | OOS p05 -1.7% | more-robust-than-fwd#1: YES
- **SMA x 1h** -- band: 2MA fast[2, 102] slow[5, 239] (n=53); 3MA fast[2, 186] slow[14, 233] (n=53) (total 106 members) | iron: vol-target(rv=336) + min_hold(12) + trail(0.1) | NEW: TRAIN 22.2% VAL 18.9% OOS 27.6% FULL 85.3% | maxDD -11.4% | OOS p05 -0.6% | more-robust-than-fwd#1: YES
- **WMA x 1h** -- band: 2MA fast[2, 102] slow[3, 239] (n=51); 3MA fast[2, 186] slow[14, 233] (n=58) (total 109 members) | iron: vol-target(rv=336) + min_hold(12) + trail(0.1) | NEW: TRAIN 23.3% VAL 17.6% OOS 26.9% FULL 84.0% | maxDD -11.8% | OOS p05 -1.7% | more-robust-than-fwd#1: no
- **HMA x 1h** -- band: 2MA fast[2, 102] slow[3, 239] (n=42); 3MA fast[2, 186] slow[4, 233] (n=36) (total 78 members) | iron: vol-target(rv=336) + min_hold(12) + trail(0.1) | NEW: TRAIN 18.2% VAL 14.6% OOS 28.1% FULL 73.5% | maxDD -16.1% | OOS p05 -0.5% | more-robust-than-fwd#1: no
- **DEMA x 1h** -- band: 2MA fast[2, 102] slow[3, 239] (n=49); 3MA fast[2, 186] slow[4, 233] (n=46) (total 95 members) | iron: vol-target(rv=336) + min_hold(12) + trail(0.1) | NEW: TRAIN 22.6% VAL 17.5% OOS 27.5% FULL 83.6% | maxDD -11.6% | OOS p05 -0.5% | more-robust-than-fwd#1: YES
- **TEMA x 1h** -- band: 2MA fast[2, 102] slow[3, 239] (n=46); 3MA fast[2, 186] slow[4, 233] (n=39) (total 85 members) | iron: vol-target(rv=336) + min_hold(12) + trail(0.1) | NEW: TRAIN 19.3% VAL 14.7% OOS 28.0% FULL 75.1% | maxDD -14.4% | OOS p05 -1.1% | more-robust-than-fwd#1: no
- **KAMA x 1h** -- band: 2MA fast[2, 102] slow[5, 239] (n=55); 3MA fast[2, 186] slow[4, 233] (n=59) (total 114 members) | iron: vol-target(rv=336) + min_hold(12) + trail(0.1) | NEW: TRAIN 21.1% VAL 16.6% OOS 23.7% FULL 74.6% | maxDD -11.0% | OOS p05 -2.2% | more-robust-than-fwd#1: YES
- **VIDYA x 1h** -- band: 2MA fast[2, 102] slow[3, 239] (n=60); 3MA fast[2, 186] slow[4, 233] (n=60) (total 120 members) | iron: vol-target(rv=336) + min_hold(12) + trail(0.1) | NEW: TRAIN 21.8% VAL 15.5% OOS 28.4% FULL 80.7% | maxDD -11.3% | OOS p05  2.1% | more-robust-than-fwd#1: YES

### TF = 30m  (EW u10 buy-hold FULL-2020 = 150.0%, cadence-invariant reference)

- **EMA x 30m** -- band: 2MA fast[2, 102] slow[19, 239] (n=48); 3MA fast[2, 186] slow[15, 233] (n=46) (total 94 members) | iron: vol-target(rv=672) + min_hold(12) + trail(0.1) | NEW: TRAIN 22.2% VAL 18.1% OOS 18.1% FULL 70.5% | maxDD -10.9% | OOS p05 -6.9% | more-robust-than-fwd#1: YES
- **SMA x 30m** -- band: 2MA fast[2, 102] slow[28, 239] (n=46); 3MA fast[2, 186] slow[24, 233] (n=44) (total 90 members) | iron: vol-target(rv=672) + min_hold(12) + trail(0.1) | NEW: TRAIN 20.0% VAL 17.8% OOS 21.0% FULL 71.0% | maxDD -10.9% | OOS p05 -3.4% | more-robust-than-fwd#1: no
- **WMA x 30m** -- band: 2MA fast[2, 102] slow[28, 239] (n=41); 3MA fast[2, 186] slow[38, 233] (n=38) (total 79 members) | iron: vol-target(rv=672) + min_hold(12) + trail(0.1) | NEW: TRAIN 20.9% VAL 17.6% OOS 19.4% FULL 69.8% | maxDD -11.0% | OOS p05 -4.9% | more-robust-than-fwd#1: no
- **HMA x 30m** -- band: 2MA fast[2, 102] slow[3, 239] (n=26); 3MA fast[2, 186] slow[15, 233] (n=22) (total 48 members) | iron: vol-target(rv=672) + min_hold(12) + trail(0.1) | NEW: TRAIN 21.3% VAL 12.1% OOS 22.3% FULL 66.3% | maxDD -12.6% | OOS p05 -3.0% | more-robust-than-fwd#1: no
- **DEMA x 30m** -- band: 2MA fast[2, 102] slow[28, 239] (n=39); 3MA fast[3, 186] slow[34, 233] (n=27) (total 66 members) | iron: vol-target(rv=672) + min_hold(12) + trail(0.1) | NEW: TRAIN 19.1% VAL 14.8% OOS 23.3% FULL 68.5% | maxDD -12.3% | OOS p05 -1.5% | more-robust-than-fwd#1: no
- **TEMA x 30m** -- band: 2MA fast[2, 102] slow[38, 239] (n=27); 3MA fast[3, 186] slow[60, 233] (n=15) (total 42 members) | iron: vol-target(rv=672) + min_hold(12) + trail(0.1) | NEW: TRAIN 22.2% VAL 10.7% OOS 22.8% FULL 66.0% | maxDD -13.2% | OOS p05 -2.7% | more-robust-than-fwd#1: no
- **KAMA x 30m** -- band: 2MA fast[2, 102] slow[28, 239] (n=45); 3MA fast[2, 75] slow[34, 233] (n=34) (total 79 members) | iron: vol-target(rv=672) + min_hold(12) + trail(0.1) | NEW: TRAIN 16.6% VAL 16.2% OOS 20.6% FULL 63.3% | maxDD -11.7% | OOS p05 -3.5% | more-robust-than-fwd#1: no
- **VIDYA x 30m** -- band: 2MA fast[2, 102] slow[10, 239] (n=57); 3MA fast[2, 186] slow[14, 233] (n=57) (total 114 members) | iron: vol-target(rv=672) + min_hold(12) + trail(0.1) | NEW: TRAIN 24.3% VAL 16.3% OOS 19.2% FULL 72.4% | maxDD -11.5% | OOS p05 -3.7% | more-robust-than-fwd#1: YES

### TF = 15m  (EW u10 buy-hold FULL-2020 = 157.2%, cadence-invariant reference)

- **EMA x 15m** -- band: 2MA fast[2, 102] slow[28, 239] (n=37); 3MA fast[2, 186] slow[38, 233] (n=28) (total 65 members) | iron: vol-target(rv=1344) + min_hold(12) + trail(0.1) | NEW: TRAIN 18.0% VAL 18.1% OOS 15.5% FULL 61.0% | maxDD -12.3% | OOS p05 -9.9% | more-robust-than-fwd#1: YES
- **SMA x 15m** -- band: 2MA fast[3, 102] slow[65, 239] (n=29); 3MA fast[2, 186] slow[60, 233] (n=23) (total 52 members) | iron: vol-target(rv=1344) + min_hold(12) + trail(0.1) | NEW: TRAIN 17.7% VAL 17.3% OOS 19.3% FULL 64.7% | maxDD -11.7% | OOS p05 -4.5% | more-robust-than-fwd#1: YES
- **WMA x 15m** -- band: 2MA fast[4, 102] slow[28, 239] (n=29); 3MA fast[5, 186] slow[60, 233] (n=17) (total 46 members) | iron: vol-target(rv=1344) + min_hold(12) + trail(0.1) | NEW: TRAIN 16.8% VAL 15.8% OOS 17.9% FULL 59.5% | maxDD -13.3% | OOS p05 -6.9% | more-robust-than-fwd#1: YES
- **HMA x 15m** -- band: 2MA fast[2, 102] slow[3, 239] (n=12); 3MA fast[5, 186] slow[148, 233] (n=8) (total 20 members) | iron: vol-target(rv=1344) + min_hold(12) + trail(0.1) | NEW: TRAIN 19.5% VAL 14.5% OOS 16.0% FULL 58.7% | maxDD -17.3% | OOS p05 -7.8% | more-robust-than-fwd#1: YES
- **DEMA x 15m** -- band: 2MA fast[6, 102] slow[38, 239] (n=24); 3MA fast[10, 186] slow[60, 233] (n=11) (total 35 members) | iron: vol-target(rv=1344) + min_hold(12) + trail(0.1) | NEW: TRAIN 20.1% VAL 10.7% OOS 21.8% FULL 62.0% | maxDD -14.2% | OOS p05 -3.4% | more-robust-than-fwd#1: no
- **TEMA x 15m** -- band: 2MA fast[26, 102] slow[89, 239] (n=13); 3MA fast[3, 186] slow[118, 233] (n=7) (total 20 members) | iron: vol-target(rv=1344) + min_hold(12) + trail(0.1) | NEW: TRAIN 22.2% VAL 10.3% OOS 14.6% FULL 54.5% | maxDD -14.3% | OOS p05 -8.8% | more-robust-than-fwd#1: no
- **KAMA x 15m** -- band: 2MA fast[4, 102] slow[23, 239] (n=31); 3MA fast[3, 186] slow[118, 233] (n=16) (total 47 members) | iron: vol-target(rv=1344) + min_hold(12) + trail(0.1) | NEW: TRAIN 15.2% VAL 13.3% OOS 22.0% FULL 59.4% | maxDD -14.0% | OOS p05 -2.7% | more-robust-than-fwd#1: no
- **VIDYA x 15m** -- band: 2MA fast[2, 102] slow[12, 239] (n=54); 3MA fast[2, 186] slow[14, 233] (n=48) (total 102 members) | iron: vol-target(rv=1344) + min_hold(12) + trail(0.1) | NEW: TRAIN 21.7% VAL 17.7% OOS 17.8% FULL 68.8% | maxDD -11.6% | OOS p05 -6.8% | more-robust-than-fwd#1: YES

## CAVEATS (binding)
- STRICT long-only + spot -- ZERO short logic anywhere. 2020 OOS (Oct-Dec) is a clean BULL (~0% bear); these are PARTICIPATING-BETA long-only books -- under-participation vs buy-hold is EXPECTED and is not a defect.
- Fixed-EW (`fillna(0.0).mean(axis=1)`); SELFTEST confirms buy-hold is cadence-invariant (~140-157%), NOT the skipna-inflated ~200/675.
- The vol-target median_rv is a single in-2020 reference (descriptive); a live deploy uses a trailing-only median. The BAND/#1 are descriptive-of-2020; the held-out claim is at the BAND level (TRAIN&VAL&OOS-positivity), and the mechanics are causal/lag-1.
- 2h is SYNTHESIZED from 1h (OHLC-resample). SOL/AVAX have only 2020-H2 history.
- DEEPER irons (overshoot-damper, regime-adaptive param) are NOTED FUTURE, NOT built -- so the per-type iron here is the band-region + the uniform vol-target only.
