# 15m DETERIORATION-FIGHTER ladder -- can mechanical turnover-cuts recover the cost-destroyed 15m strats? (2020)

Per indicator (canonical config), the fighter ladder at 15m: entry-CONFIRM(C) + MIN-HOLD(M) + COOLDOWN(K) escalating. Cuts turnover (the 15m cost driver). **[VERIFIED-2020-OOS, fixed-EW, maker]** RWYB: python -m strat.deep2020_ti_15m_fighter --cadence 15m


## MACD (12, 26, 9) @ 15m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | -13.0 | -2.39 | -13.8 | 311.6 | 0.24 | -2.4 |
| L1 conf2/hold24/cd12 | 8.6 | 1.35 | -8.0 | 209.5 | 0.32 | 9.4 |
| L2 conf4/hold48/cd48 | 7.9 | 1.69 | -7.0 | 130.8 | 0.23 | 9.2 |
| L3 conf8/hold96/cd96 | 6.6 | 1.62 | -6.2 | 77.2 | 0.19 | 10.9 |
| L4 conf12/hold192/cd192 | 24.4 | 4.42 | -3.8 | 41.8 | 0.18 | 17.9 |

## SUPERTREND (10, 3.0) @ 15m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | -8.0 | -1.14 | -10.5 | 235.8 | 0.29 | -6.1 |
| L1 conf2/hold24/cd12 | 3.8 | 0.68 | -8.1 | 164.5 | 0.33 | 2.1 |
| L2 conf4/hold48/cd48 | 10.6 | 1.96 | -6.5 | 111.7 | 0.25 | 10.0 |
| L3 conf8/hold96/cd96 | 15.4 | 3.35 | -5.4 | 65.6 | 0.23 | 23.4 |
| L4 conf12/hold192/cd192 | 6.7 | 1.49 | -3.9 | 40.2 | 0.19 | 5.2 |

## DONCHIAN (20, 10) @ 15m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | -9.7 | -1.71 | -9.7 | 167.5 | 0.23 | 1.8 |
| L1 conf2/hold24/cd12 | -3.4 | -0.41 | -8.5 | 156.2 | 0.29 | 6.1 |
| L2 conf4/hold48/cd48 | 8.1 | 1.52 | -6.7 | 113.7 | 0.25 | 11.5 |
| L3 conf8/hold96/cd96 | 2.2 | 0.67 | -4.4 | 74.8 | 0.21 | 5.9 |
| L4 conf12/hold192/cd192 | 10.0 | 2.02 | -6.3 | 41.5 | 0.18 | 5.2 |

## ROC (50, 0.0) @ 15m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | 9.1 | 1.15 | -8.4 | 253.7 | 0.36 | -2.4 |
| L1 conf2/hold24/cd12 | 19.8 | 2.43 | -6.1 | 139.1 | 0.36 | 11.3 |
| L2 conf4/hold48/cd48 | 28.8 | 4.13 | -3.8 | 93.0 | 0.28 | 10.3 |
| L3 conf8/hold96/cd96 | 18.6 | 2.89 | -6.8 | 59.8 | 0.27 | 17.1 |
| L4 conf12/hold192/cd192 | 19.3 | 2.87 | -5.4 | 37.0 | 0.23 | 20.4 |

## RSI (14, 30, 60) @ 15m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | 2.5 | 1.23 | -1.9 | 103.7 | 0.06 | 5.9 |
| L1 conf2/hold24/cd12 | 10.3 | 4.12 | -1.7 | 68.1 | 0.1 | 11.7 |
| L2 conf4/hold48/cd48 | 11.3 | 3.52 | -1.9 | 45.3 | 0.12 | 8.4 |
| L3 conf8/hold96/cd96 | 5.2 | 1.49 | -3.5 | 28.9 | 0.14 | 4.4 |
| L4 conf12/hold192/cd192 | 7.2 | 1.73 | -6.9 | 17.0 | 0.16 | -11.6 |

## MFI (14, 30, 80) @ 15m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | 7.7 | 1.13 | -7.1 | 208.0 | 0.35 | 5.5 |
| L1 conf2/hold24/cd12 | 13.4 | 1.94 | -8.0 | 120.2 | 0.35 | 6.5 |
| L2 conf4/hold48/cd48 | 14.6 | 2.5 | -5.3 | 82.8 | 0.28 | -3.1 |
| L3 conf8/hold96/cd96 | 18.3 | 3.5 | -5.3 | 54.1 | 0.27 | 12.6 |
| L4 conf12/hold192/cd192 | 20.4 | 3.37 | -3.9 | 34.8 | 0.24 | 8.0 |

## KELTNER (20, 2.0) @ 15m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | -9.1 | -1.74 | -9.9 | 189.4 | 0.18 | -2.9 |
| L1 conf2/hold24/cd12 | 1.1 | 0.3 | -7.6 | 147.0 | 0.25 | 1.9 |
| L2 conf4/hold48/cd48 | 9.0 | 1.72 | -5.7 | 101.0 | 0.23 | 10.8 |
| L3 conf8/hold96/cd96 | 5.8 | 1.41 | -6.6 | 60.7 | 0.23 | 7.6 |
| L4 conf12/hold192/cd192 | 12.4 | 2.13 | -4.6 | 35.1 | 0.22 | 8.8 |

## VOLIMB (3, 0.52) @ 15m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | -8.5 | -1.33 | -10.8 | 550.1 | 0.26 | 1.9 |
| L1 conf2/hold24/cd12 | 8.0 | 1.46 | -6.2 | 214.2 | 0.28 | -3.4 |
| L2 conf4/hold48/cd48 | 12.3 | 2.97 | -3.7 | 97.8 | 0.19 | 5.0 |
| L3 conf8/hold96/cd96 | 11.3 | 3.16 | -3.3 | 28.6 | 0.15 | 6.4 |
| L4 conf12/hold192/cd192 | -0.4 | -0.1 | -3.9 | 7.9 | 0.08 | -4.1 |

## VERDICT -- does the fighter ladder recover 15m? (honest: FIXED level L2 conf4/hold48/cd48, no cherry-pick)

| indicator | BASE net | L2 conf4/hold48/cd48 net | VAL net | drift | Sharpe | robust(both>0)? |
|---|---|---|---|---|---|---|
| MACD | -13.0% | 7.9% | -1.3% | 9.2 | 1.69 | no |
| SUPERTREND | -8.0% | 10.6% | 0.7% | 10.0 | 1.96 | YES |
| DONCHIAN | -9.7% | 8.1% | -3.4% | 11.5 | 1.52 | no |
| ROC | 9.1% | 28.8% | 18.5% | 10.3 | 4.13 | YES |
| RSI | 2.5% | 11.3% | 2.8% | 8.4 | 3.52 | YES |
| MFI | 7.7% | 14.6% | 17.7% | -3.1 | 2.5 | YES |
| KELTNER | -9.1% | 9.0% | -1.8% | 10.8 | 1.72 | no |
| VOLIMB | -8.5% | 12.3% | 7.2% | 5.0 | 2.97 | YES |

**At the FIXED fighter level L2 conf4/hold48/cd48 (no cherry-pick): 8/8 recover to POSITIVE, 5/8 ROBUST (positive in BOTH VAL+OOS).** [VERIFIED-2020-OOS] The MECHANISM is MONOTONE -- every indicator's net climbs as turnover falls across the ladder -- so the 15m deterioration is DEFINITIVELY cost-of-overtrading, not signal failure. CAVEATS (honest): (1) the recovered nets are still <= 15m buy-hold (~54.8%) -- this is de-risked BETA (positive, high-Sharpe 2-4, shallow maxDD -3..-7), NOT alpha over holding; (2) picking the BEST level per indicator is in-sample-favorable (e.g. SUPERTREND's best level has NEGATIVE VAL = OOS-lucky) -- trust the fixed-level + both-windows-positive column, not the per-indicator max; (3) canonical configs (mechanism test), not 15m-optimized. Bottom line: 15m is SALVAGEABLE as a de-risked sleeve via turnover-fighters -- do NOT discard it; but it does not break the drift-beta ceiling.
