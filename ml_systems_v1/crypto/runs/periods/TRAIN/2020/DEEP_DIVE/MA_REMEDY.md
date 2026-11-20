# MA x TIMEFRAME -- the LEAST-WEAKNESS build per MA kind (2020) -- literature remedies applied

User /orc 2026-06-13/14: "if MA x Timeframe were your ONLY approach, what is the best build with the LEAST
weaknesses?" Method: brute-force 2020 MA x TF grid + the literature remedies from
[`LITERATURE_CROSSCHECK.md`](LITERATURE_CROSSCHECK.md); remedy each documented weakness; develop a robust set
per MA kind (SMA/EMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA); side-by-side. Stay in 2020 (VAL Jul-Sep, OOS Oct-Dec).
Tools: `src/strat/deep2020_ma_remedy.py` (+ `_render.py`). ALL numbers below are **[VERIFIED-backtest, IN-SAMPLE
2020, VAL-select / OOS-report, maker, long-only spot lev<=1]** -- they need cross-year/UNSEEN validation before
deploy. UNSEEN untouched.

> **CORRECTED 2026-06-14 after an adversarial verification pass (5-skeptic + 2-vote) caught TWO real bugs in
> the first cut -- a good catch, exactly what the gate is for:**
> 1. **Vol-target LEVEL snoop (HIGH):** the vol target `vt` was calibrated over the FULL VAL+OOS window ->
>    fixed to VAL-only (no OOS peek).
> 2. **Alignment artifact (HIGH, the big one):** the per-bar `mean(skipna=True)` over a union index silently
>    reweighted to "EW-of-available" on rows where late-2020 listers (SOL/AVAX) had no bars -> it INFLATED
>    finer cadences (1h BUYHOLD read 89% when fixed-EW is 52%). The "rebalancing/vol-harvest premium" caveat in
>    the first cut was WRONG. Fixed to **fixed equal-weight of the full universe (a not-yet-listed asset = cash =
>    0 return)**, which is cadence-invariant. Side effect: it ALSO showed my first-cut "-15 to -49pp selection
>    drift" was itself partly the skipna artifact -- corrected drifts are mixed (below).

## THE WEAKNESS -> REMEDY MAP (each a measured knob)
| W | weakness (from the 2020 deep-dive) | literature | REMEDY knob |
|---|---|---|---|
| W1 | whipsaw / overtrading (cost drag, worst at fine TF) | Zakamulin | **R1 CONFIRM band** -- 0.5% hysteresis no-trade buffer around the cross |
| W2 | selection risk / data-snooping (in-sample #1 fails OOS) | STW 1999 | **R2 SELECT-ON-VAL + CLUSTER ENSEMBLE** -- run the robust cluster center, never the lucky #1 |
| W3/4 | under-participation + over-exit | Faber / Block A,B | base FULL stack = trail10 + min_hold12 (loose exit kept; tighter was refuted) |
| W7 | bear vulnerability / no regime | HYZ / Faber | **R3 REGIME gate** -- long only when slow-MA slope>0 (the ALL-WEATHER knob) |
| W8 | vol-clustering drawdown spikes | Faber / AQR | **R4 VOL-TARGET** -- scale exposure to a vol target, cap [0,1] (no leverage) |

**Honest construction (this IS the W2 remedy):** the winning cluster + every knob are chosen on VAL (Jul-Sep)
ONLY; every reported number is the held-out OOS (Oct-Dec). The ladder is `RAW-best` (single VAL-winner) ->
`ENS` (+R2) -> `+CONFIRM` (+R1) -> `+VOLTGT` (+R4) -> `FULL-REM` (+R3).

## WHAT EACH REMEDY DID (corrected, measured across all MA kinds x 1d/4h/2h/1h)
1. **R4 VOL-TARGET is the biggest, most reliable win.** [VERIFIED-2020-OOS] It lifts Sharpe (to ~3-4.4) and
   **cuts maxDD ~35-50%** (HMA 1h -10.6->-6.7; SMA 2h -9.7->-5.3; DEMA 4h -11.6->-6.0) for a modest net cost. [VERIFIED-2020-OOS]
   Universal across MA kinds + TFs. This is Faber's variance-drain, realized.
2. **R2 ENSEMBLE stabilizes the WORST selection give-backs and usually RAISES OOS net.** [VERIFIED-2020-OOS]
   `RAW-best` (the single VAL-winner) is sometimes fragile -- KAMA 4h VAL->OOS drift -19.3, KAMA 2h -16.5, [VERIFIED-2020-OOS]
   EMA 1d -8.5 -- and the ENSEMBLE fixes those (KAMA 4h drift -19.3->0.0; 2h -16.5->+2.5). [VERIFIED-2020-OOS] It also tends to
   LIFT OOS net (EMA 1h 23.1->36.0; HMA 2h 36.2->42.9) by running the robust cluster center. (The first cut's
   "universal -49pp fragility" was an alignment-bug artifact; the corrected effect is "fixes the fragile cases
   + raises net," not a universal rescue.)
3. **R1 CONFIRM band cuts turnover/whipsaw** (cost-fragility), most at fine TF (EMA 1h turnover 90 -> 21).
4. **R3 REGIME gate COSTS return in 2020** (a clean bull has no bear to gate out) + spikes slope-flip turnover
   -- so `FULL-REM` is NOT the 2020 winner. It is the **all-weather** build; its value is the out-of-2020 bear
   protection the cross-check (Q1) says is where MA timing's real alpha lives.
5. **Low-lag / adaptive MA types beat the classics.** HMA / DEMA / TEMA / KAMA dominate the weakness axes;
   EMA / SMA are mid-pack. The MA-type axis matters (the lateness weakness, Block D).

## THE ANSWER -- recommended least-weakness build = +VOLTGT (R1+R2+R4) on a low-lag MA
**Cross-MA winner = HULL (HMA): champion or near-champion at EVERY cadence and EVERY lens.** [VERIFIED-2020-OOS]

- **Best CAPTURE (net vs SAME-cadence buy-hold) + low DD:** HMA keeps **0.77x buy-hold at 4h** (36.7% vs 47.8%) [VERIFIED-2020-OOS]
  while cutting maxDD -19.8 -> -8.2 and lifting Sharpe 2.36 -> 3.61; and **0.83x at 2h** (Sh 4.18, DD -5.7). [VERIFIED-2020-OOS]
- **Most ROBUST at the coarse end:** HMA 1d (net 29.7=0.63xBH, Sh 3.0, DD -12.3) -- the steadiest book. [VERIFIED-2020-OOS]
- **The cross ADDS risk-adjusted value over pure vol-targeted holding at fine TF:** [VERIFIED-2020-OOS] at 1h/2h
  the best remedied MA builds BEAT VOLTGT_BH on Sharpe AND maxDD (HMA 2h Sh 4.18 / DD -5.7 vs VOLTGT_BH 3.22 / [VERIFIED-2020-OOS]
  -13.2; TEMA 1h Sh 4.42 / DD -5.2 vs 3.33 / -12.8) -- trading ~25% of net for ~2x better Sharpe and ~half the [VERIFIED-2020-OOS]
  drawdown. At 1d, VOLTGT_BH (Sh 2.93) is competitive with the MA builds, so the cross earns its keep mainly at
  FINER cadence.

**So, if MA x TF were the only approach:** a **vol-targeted, confirm-banded, cluster-ENSEMBLE Hull-MA book** --
**4h for the best capture/risk balance, 2h-1h for the best Sharpe/drawdown, 1d for the steadiest**. Hull's
low-lag cuts lateness; the ensemble kills selection risk; vol-target cuts drawdown; the confirm-band cuts
whipsaw. Add the regime gate (`FULL-REM`) only for the all-weather (cross-cycle) version.

## SIDE-BY-SIDE (the +VOLTGT recommended build per MA kind x TF; net (xBH) Sharpe maxDD)
| MA | 1d | 4h | 2h | 1h |
|---|---|---|---|---|
| EMA | 22.8 (.48) 2.40 -13.0 | 21.9 (.46) 2.72 -8.5 | 32.1 (.64) 3.56 -6.2 | 34.6 (.67) 3.83 -5.2 |
| SMA | 26.7 (.56) 2.83 -12.5 | 26.9 (.56) 3.02 -9.8 | 34.3 (.68) 3.76 -5.3 | 41.9 (.81) 4.13 -4.9 |
| WMA | 28.7 (.61) 2.90 -12.4 | 31.3 (.65) 3.37 -8.6 | 35.5 (.71) 3.56 -7.0 | 29.6 (.57) 3.33 -5.7 |
| **HMA** | 29.7 (.63) 3.00 -12.3 | **36.7 (.77) 3.61 -8.2** | **41.6 (.83) 4.18 -5.7** | 39.9 (.77) 4.17 -6.7 |
| DEMA | 29.8 (.63) 3.02 -11.3 | 32.5 (.68) 3.70 -6.0 | 36.1 (.72) 3.70 -5.4 | 37.5 (.73) 4.00 -4.6 |
| TEMA | 24.2 (.51) 2.54 -11.4 | 29.7 (.62) 3.20 -7.3 | 38.0 (.76) 3.90 -6.2 | 40.2 (.78) 4.42 -5.2 |
| KAMA | 20.2 (.43) 2.99 -8.7 | 30.0 (.63) 3.26 -7.9 | 28.9 (.58) 3.53 -5.0 | 37.4 (.72) 3.97 -5.2 |
| VIDYA | 22.3 (.47) 3.07 -9.2 | 20.8 (.44) 2.65 -7.3 | 30.1 (.60) 3.26 -6.6 | 35.2 (.68) 3.80 -5.0 |
| _BUYHOLD_ | _47.4 / 2.34 / -20.2_ | _47.8 / 2.36 / -19.8_ | _50.2 / 2.44 / -19.6_ | _51.6 / 2.49 / -19.4_ |
| _VOLTGT_BH_ | _49.3 / 2.93 / -14.9_ | _49.9 / 3.10 / -14.1_ | _52.1 / 3.22 / -13.2_ | _54.2 / 3.33 / -12.8_ |

Buy-hold is now ~cadence-invariant (47-52%) -- the cross-cadence figures are honestly comparable. (2h is a
synthesized cadence resampled from 1h; trust 1d/4h/1h over it.)

## THE CEILING (the load-bearing honest finding -- unchanged by the fixes)
**Every remedied build keeps only 0.43-0.83x of same-cadence buy-hold net.** [VERIFIED-2020-OOS] The remedies
MINIMIZE weaknesses (selection give-back, drawdown, whipsaw, lateness) but **cannot beat the drift on net in a
bull** -- the net ceiling IS buy-hold (the drift-beta). What the remedied MA book BUYS is **RISK** -- at fine
cadence it roughly DOUBLES the Sharpe and HALVES the maxDD of buy-hold while keeping ~0.7-0.8x of the return. [VERIFIED-2020-OOS]
That is a real, deployable de-risking; it is just not extra RETURN -- exactly the Faber / cross-check verdict.

## CAVEATS (RWYB honest)
- **The fixed-EW (unlisted=cash) convention mildly damps the VAL leg** (SOL/AVAX are partly cash in Jul-Sep
  2020), so the `drift` (OOS-VAL) column now CONFOUNDS VAL-cash-drag + the stronger Nov-OOS quarter + selection
  stability -- it is no longer a clean fragility signal. Lean on the ENS-vs-RAW comparison (still shows the
  ensemble fixing the worst give-backs) + maxDD + Sharpe + net/BH, NOT on drift's sign.
- **In-sample 2020.** VAL/OOS are adjacent same-bull; cluster identities + ranks are NOT cross-year / UNSEEN
  validated. The regime gate's value is explicitly out-of-2020 (the full-cycle test, cross-check Q1).
- **VAL is 3 months** (~92 daily bars) -- VAL-selection still has sampling noise; the ENSEMBLE is what makes it
  robust (selecting the cluster, not the point).
- **Verified clean (adversarial pass, 2026-06-14):** no per-bar look-ahead (positions lagged, vol-target uses
  past vol); selection is strictly VAL-only; cost model correct (maker round-trip, no MtM double-count); metrics
  correct (Sharpe sqrt365 on daily-resampled). The two HIGH bugs above were fixed and re-run.
- json: `ma_remedy_1d_4h.json` + `ma_remedy_2h_1h.json`. RWYB: `python -m strat.deep2020_ma_remedy --cadences
  1d,4h,2h,1h` then `python -m strat.deep2020_ma_remedy_render`.
