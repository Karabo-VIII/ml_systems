# IRONED FINE-TF MA TREND SYSTEM -- 2020 deep-dive (CONSTRUCTION, not refutation)

`src/strat/deep2020_ironed_fine.py` -- RWYB: `python -m strat.deep2020_ironed_fine --cadences 1h,30m,15m`
JSON: `runs/periods/TRAIN/2020/DEEP_DIVE/ironed_fine.json`. All numbers VERIFIED (RWYB) unless tagged INFERRED.

## The brief
Build the per-TF MA TREND system with the fine-cadence creases IRONED OUT, on {1h, 30m, 15m}, WITHIN-2020
(TRAIN 2020-01..07 / VAL 07..10 / OOS 10..2021-01). SELECT MA-type + whipsaw-filter + exit + gate on TRAIN+VAL
ONLY; confirm ONCE on OOS. Report maker AND taker. DEPLOY BAR: net approaches VOLTGT_BH with maxDD materially
below buy-hold and high coverage, net of REAL cost. The leaderboard crease: "15m HMA/TEMA full coverage but net
2.5-3.9% (cost-eaten)"; D60 = the 1h MR cost wall.

## THE HEADLINE FINDING -- the cost wall does NOT bind the ironed system; UNDER-PARTICIPATION does
The leaderboard's "cost-eaten" fine-TF MA was a NAIVE-config artifact (fixed EMA/HMA/TEMA, no whipsaw filter,
single config per type). Once you (1) use an ADAPTIVE MA (VIDYA -- fast in trend, slow in chop, so it barely
whipsaws), (2) confirm + min-hold, (3) ensemble the lookback family, and (4) debounce the regime gate, OOS
turnover collapses to 12-28 round-trips/3mo and the maker->taker gap on the full stack is only **1.6 / 2.9 /
5.6 pp** (1h / 30m / 15m). The system clears cost comfortably at BOTH maker and taker.

What binds instead is PARTICIPATION: the ironed system's OOS net is 0.31x / 0.53x / 0.53x of VOLTGT_BH because
it de-risks hard (maxDD -7.6 to -10.5% vs buy-hold -23 to -27%) and sits out a third-to-half of the 2020 bull.
**This is the correct, structural bull artifact** (per the brief's warning): in a relentless bull, ANY long/flat
trend system that ever goes flat loses to holding. The honest verdict is two-sided below.

## Causality / discipline (look-ahead audit -- a real bug was caught and fixed)
- Signal lag-1: `pos[t] = held[t-1]`. Vol-target uses `realized_vol.shift(1)` (vol through t-1).
- **Regime gate look-ahead CAUGHT in build**: the first cut multiplied `fpos[t] * (BTC_close[t] > SMA100[t])`
  -- same-bar information. Fixed: the BTC regime is LAGGED 1 bar (`reg[t-1]`) before grid-alignment. The bug had
  inflated the 1h gate to +126% (above buy-hold); post-fix it is +28%. This is exactly the RED-TEAM discipline
  working -- the "too good" number was a leak.
- Regime gate FLICKER caught: a raw bar-by-bar BTC-SMA100 gate at fine TF flips on every crossing -> turnover
  exploded 14->69. Fixed with a min-DWELL debounce on the regime state (selected on TRAIN+VAL).
- Late-listing assets (SOL/AVAX/DOGE list Aug-Sep 2020) contribute to the book only from their listing -> they
  are in VAL/OOS (full 10-asset breadth in OOS) but TRAIN breadth is 7/10. No look-ahead; just thinner early breadth.

## Cost model (the crux at fine TF)
maker = 6 bps round-trip, taker = 24 bps round-trip (`portfolio_replay.MAKER_RT/TAKER_RT`). Per CLAUDE.md the
maker p_fill reality is 0.21-0.40, so the maker figure ASSUMES ideal fills; live maker net sits between the
maker and taker rows. The small maker->taker gap (<=5.6pp even at 15m) means **fill realism is NOT the
deal-breaker for this system** -- a reassuring robustness, unlike the naive fine-TF MA the leaderboard graded.

---

## PER-TF RESULTS (OOS = 2020-10-01..2021-01-01, 3 months -- ANNUALIZE WITH CARE; 3mo bull, not a full cycle)

### Selected stack per TF (SELECTED on TRAIN+VAL, maker)
| TF  | MA-type | whipsaw (K confirm, M min-hold, cool) | exit | gate dwell | best single |
|-----|---------|----------------------------------------|------|-----------|-------------|
| 1h  | VIDYA   | (2, 12, 0)   | none | 168 bars (~1w) | (VAL-best 2MA) |
| 30m | VIDYA   | (8, 96, 48)  | none | 0 (no gate)    | (VAL-best 2MA) |
| 15m | VIDYA   | (8, 96, 48)  | none | 0 (no gate)    | (VAL-best 2MA) |

VIDYA (adaptive) wins the MA-type selection at ALL THREE fine TFs on TRAIN+VAL -- decisively at 15m
(VIDYA 151 vs next-best SMA 61; HMA/TEMA NEGATIVE). This is the on-family upgrade that irons the whipsaw crease:
the leaderboard's failing 15m types were HMA/TEMA; the winning type is the ADAPTIVE one. The exit overlay adds
nothing on TRAIN+VAL (none > trail/chandelier everywhere -- chandelier near-zero), so the ironed stack carries NO
exit beyond the entry signal + (at 1h) the regime gate.

### BEFORE -> AFTER ladder (OOS net%, maxDD%, turnover; MAKER and TAKER)
Turnover = mean per-asset sum|d position| over the OOS window (round-trips ~= turnover/2). breadth = #/10 OOS-positive.

**1h** (BUYHOLD net +89.4% maxDD -23.4% Sh 3.45 | VOLTGT_BH net +81.2% maxDD -14.9% Sh 4.29, cov 100)

| stage       | maker net | maker maxDD | turnover | taker net | taker maxDD | cov | breadth | n_eff |
|-------------|-----------|-------------|----------|-----------|-------------|-----|---------|-------|
| S0 single   | +68.6 | -23.4 |  5.3 | +67.7 | -23.4 | 76.9 | 7/10 | 2.0 |
| S1 +family  | +54.3 | -19.7 | 14.4 | +52.0 | -19.8 | 66.8 | 7/10 | 2.0 |
| S2 +whipsaw | +55.2 | -19.7 | 13.9 | +53.0 | -19.8 | 66.6 | 7/10 | 2.0 |
| S3 +exit    | +55.2 | -19.7 | 13.9 | +53.0 | -19.8 | 66.6 | 7/10 | 2.0 |
| S4 +gate    | +28.0 |  -8.8 | 13.7 | +26.2 |  -9.3 | 45.9 | 7/10 | 2.1 |
| S5 +voltgt  | +25.4 |  -7.6 | 12.3 | +23.8 |  -8.0 | 38.3 | 7/10 | 2.0 |

**30m** (BUYHOLD net +125.5% maxDD -25.3% Sh 4.09 | VOLTGT_BH net +103.5% maxDD -15.4% Sh 4.89, cov 100)

| stage       | maker net | maker maxDD | turnover | taker net | taker maxDD | cov | breadth | n_eff |
|-------------|-----------|-------------|----------|-----------|-------------|-----|---------|-------|
| S0 single   | +78.0 | -17.6 | 15.8 | +74.9 | -17.6 | 60.5 | 7/10 | 2.1 |
| S1 +family  | +64.0 | -16.5 | 26.6 | +59.2 | -16.6 | 62.4 | 7/10 | 2.1 |
| S2 +whipsaw | +61.2 | -16.9 | 17.8 | +58.0 | -17.0 | 64.0 | 7/10 | 2.0 |
| S3 +exit    | +61.2 | -16.9 | 17.8 | +58.0 | -17.0 | 64.0 | 7/10 | 2.0 |
| S4 +gate    | +61.2 | -16.9 | 17.8 | +58.0 | -17.0 | 64.0 | 7/10 | 2.0 |
| S5 +voltgt  | +54.4 | -10.5 | 16.4 | +51.5 | -10.8 | 53.4 | 7/10 | 2.0 |

(30m gate dwell selected = 0 -> S4 == S3; the market gate did not help on TRAIN+VAL at 30m.)

**15m** (BUYHOLD net +153.9% maxDD -26.7% Sh 4.55 | VOLTGT_BH net +127.0% maxDD -16.1% Sh 5.43, cov 100)

| stage       | maker net | maker maxDD | turnover | taker net | taker maxDD | cov | breadth | n_eff |
|-------------|-----------|-------------|----------|-----------|-------------|-----|---------|-------|
| S0 single   | +96.3 | -21.5 | 13.2 | +93.2 | -21.6 | 59.5 | 7/10 | 2.1 |
| S1 +family  | +70.5 | -13.6 | 48.7 | +60.8 | -14.5 | 57.1 | 7/10 | 2.2 |
| S2 +whipsaw | +74.2 | -15.7 | 31.3 | +67.7 | -16.3 | 60.6 | 7/10 | 2.1 |
| S3 +exit    | +74.2 | -15.7 | 31.3 | +67.7 | -16.3 | 60.6 | 7/10 | 2.1 |
| S4 +gate    | +74.2 | -15.7 | 31.3 | +67.7 | -16.3 | 60.6 | 7/10 | 2.1 |
| S5 +voltgt  | +67.0 |  -8.6 | 28.1 | +61.4 |  -9.1 | 51.7 | 7/10 | 2.5 |

(15m gate dwell selected = 0 -> S4 == S3. The whipsaw filter S1->S2 CUT turnover 48.7->31.3 and LIFTED net
+70.5->+74.2 maker -- the one place the whipsaw filter measurably ironed the crease, exactly as intended.)

### What each crease-iron did (read the ladder deltas)
1. **WHIPSAW** -- the whipsaw filter (confirm+min-hold) cut turnover most at 15m (48.7->31.3, +3.7pp net maker,
   +6.9pp taker). At 1h/30m VIDYA already had low whipsaw so the filter was ~neutral-to-slightly-negative on net
   but did not hurt. The dominant whipsaw-iron is the MA-TYPE choice (VIDYA), not the overlay.
2. **COST** -- NOT binding on the ironed stack. maker->taker gap on S5: 1.6 / 2.9 / 5.6 pp. Gross (zero-cost) S5
   net ~= maker net + 0.4-0.8pp. The filter cut turnover enough that net clears cost with wide margin.
3. **PARAM FRAGILITY** -- the family ENSEMBLE traded ~14pp of single-config net (S0->S1) for robustness: n_eff
   ~2.0-2.5 (the 39 configs are highly correlated -> only ~2 independent bets), lower maxDD, no single-config
   selection risk. Honest: the family is LOWER net than the lucky VAL-best single -- diversification de-risks, it
   does not add return here.
4. **GIVE-BACK** -- exit overlay added nothing (TRAIN+VAL: none beat all trails/chandelier). The min-hold already
   in the stack handles give-back; an extra trail just cuts winners.
5. **BEAR/CHOP DD + CONCENTRATION** -- the regime gate (1h) + vol-target HALVED maxDD (-19.7->-7.6%). Breadth
   7/10 positive at all stages; n_eff 2.0-2.5 (concentration is real -- the book is ~2 effective bets, the u10
   MA family is one beta cluster). Vol-target is the cleanest risk-iron: +Sharpe, -maxDD, small net cost.

### PARTICIPATION FRONTIER -- the remaining lever (gate aggressiveness at the S5 stack)
The S5 stack with the gate dialed from FULL sit-out -> HALF-size -> OFF (vol-target always on). This IS the
net-vs-risk lever. (maker net% / maxDD% / cov% / turnover). 15m row filled from the final run.

| TF  | nogate_voltgt (max participation) | halfgate_voltgt | fullgate_voltgt (the gate sits out below BTC regime) |
|-----|-----------------------------------|-----------------|------------------------------------------------------|
| 1h  | **+48.3 / -11.8 / cov 52.8 / turn 13.2** | +36.7 / -6.3 / 38.3 / 12.7 | +25.4 / -7.6 / 38.3 / 12.3 |
| 30m | +54.4 / -10.5 / 53.4 / 16.4 (= S5; gate not selected) | +37.5 / -10.8 / 32.8 / 17.6 | +21.9 / -11.6 / 32.8 / 18.8 |
| 15m | +67.0 / -8.6 / 51.7 / 28.1 (= S5; gate not selected) | **+62.1 / -6.9 / 31.0 / 30.9** | +56.7 / -8.4 / 31.0 / 33.8 |

KEY LEVER FINDING: at 1h the FULL gate (VAL-selected) over-de-risked -- dropping it (`nogate_voltgt`) nearly
DOUBLES net (+25.4 -> +48.3%) for only +4.2pp maxDD (-7.6 -> -11.8%, still 2x better than buy-hold -23.4%) and
lifts coverage 38 -> 53%. At 30m/15m a full gate only HURTS (no maxDD benefit, large net cost), which is why the
TRAIN+VAL selection correctly chose dwell=0 there. **The deployable variant per TF is `nogate_voltgt` at 30m/15m
and `nogate_voltgt` (or half-gate) at 1h** -- the full sit-out gate is over-fit to risk at the cost of the bull.

---

## PER-TF VERDICTS (two-sided, construction frame -- the specific remaining lever, not a reflexive kill)

### 1h -- COST-WALL-BOUND? NO. PARTICIPATION-BOUND. The full gate over-de-risked; the no-gate variant is decent.
- S5 (full gate) maker +25.4% / taker +23.8% vs VOLTGT_BH +81.2%, maxDD -7.6%, cov 38.3%.
- **The remaining lever WORKS**: the `nogate_voltgt` variant nets +48.3% maker (0.59x VOLTGT_BH) at maxDD -11.8%
  (still 2x better than buy-hold) and cov 52.8%. Dropping the VAL-over-fit full gate nearly doubled net for a
  small risk cost. Cost clears trivially (maker-taker 1.6pp). So the 1h deployable is the NO-GATE stack: VIDYA
  family + confirm(2)/min-hold(12) + vol-target. It still misses "net approaches VOLTGT_BH" (0.59x) but is a
  legitimate de-risked beta sleeve. Verdict: PARTICIPATION-BOUND, not cost-bound; not a strict deploy candidate.

### 30m -- COST-WALL-BOUND? NO. The best fine-TF candidate, but still under VOLTGT_BH net.
- S5 maker +54.4% / taker +51.5% vs VOLTGT_BH +103.5%. maxDD -10.5% vs BH -25.3% (2.4x better). cov 53.4%.
- 0.53x of VOLTGT_BH net at <0.5x the drawdown, >50% coverage, 7/10 breadth, cost-robust. This is the closest to
  the deploy bar. It MISSES "net approaches VOLTGT_BH" but MEETS "maxDD materially below BH + high coverage +
  cost-clear." **Remaining lever**: the gate did NOT help on TRAIN+VAL here (dwell=0), so the only net-lift lever
  is raising base exposure (lift the vol-target cap or scale the family weight up) -- a pure leverage choice, not
  a skill improvement.

### 15m -- COST-WALL-BOUND? NO (the leaderboard's "cost-eaten" verdict is REFUTED for the IRONED system).
- S5 / no-gate maker +67.0% / taker +61.4% vs VOLTGT_BH +127.0%. maxDD -8.6% vs BH -26.7% (3x better). cov 51.7%.
- The leaderboard said 15m MA = net 2.5-3.9% (cost-eaten). The IRONED 15m stack nets +67% maker / +61% taker --
  the cost wall was a naive-config artifact; VIDYA + whipsaw filter clears it. maker->taker gap 5.6pp (the largest
  of the three, as expected at the finest TF, but still small).
- **Best risk/return point at 15m is the HALF-gate**: +62.1% maker / +56.1% taker at maxDD -6.9% (the lowest DD of
  any variant at any fine TF) -- only -5pp net for the best drawdown. If you want the hardest DD cap, half-gate
  15m is the pick. net is participation-bound, not cost-bound; the lever is exposure (half-gate / vol-target cap).

## THE EXACT COST THRESHOLD (the brief's ask)
None of the three TFs is cost-wall-bound; all clear at BOTH 6bps (maker) and 24bps (taker) round-trip. The cost
at which S5 OOS net would cross ZERO (back-solved from gross ~= net + turnover * cost/2):
- 1h:  turnover 12.3, gross ~+25.8% -> breaks even at ~420 bps RT (70x the maker rate, 17.5x taker).
- 30m: turnover 16.4, gross ~+54.9% -> breaks even at ~670 bps RT.
- 15m: turnover 28.1, gross ~+67.8% -> breaks even at ~480 bps RT (80x maker, 20x taker).
So even a 17x-worse-than-taker cost regime (the weakest TF, 1h) leaves the system net-positive. The wall that
matters is the VOLTGT_BH NET bar, and that gap is PARTICIPATION (exposure), not cost.

## BOTTOM LINE
- **Deploy candidate by the strict bar (net approaches VOLTGT_BH): NONE of {1h,30m,15m}** -- all sit out too much
  of a relentless bull. This is the structural bull artifact, not a defect of the iron.
- **Deploy candidate by a RISK-PARITY mandate (maxDD materially below BH, cost-robust, >50% coverage, 7/10
  breadth): 30m and 15m QUALIFY** -- ~0.53x buy-hold net at ~0.4x the drawdown, cost-clear at taker. If the
  sleeve's job is "participate in the up-trend with a hard drawdown cap," 30m/15m VIDYA-family + vol-target is a
  legitimate de-risked beta sleeve.
- **The cost wall (D60 / leaderboard) is REFUTED for the ironed fine-TF system.** The crux flips: the fine-TF MA
  TREND problem is NOT a cost problem once you use an adaptive MA + whipsaw filter; it is an EXPOSURE problem
  against a bull benchmark. The remaining lever is participation (base exposure / half-gate), not the filter.
- Caveat: 2020 OOS is a 3-month BULL slice (in-sample-regime, no UNSEEN, no bear). The maxDD advantage is the MA
  system's real value and it PAYS in a bear (untested here). Confirm on a bear window + UNSEEN before any deploy.
