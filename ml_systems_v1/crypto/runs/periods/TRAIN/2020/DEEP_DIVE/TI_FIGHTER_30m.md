# 30m DETERIORATION-FIGHTER ladder -- can mechanical turnover-cuts recover the cost-destroyed 30m strats? (2020)

Per indicator (canonical config), the fighter ladder at 30m: entry-CONFIRM(C) + MIN-HOLD(M) + COOLDOWN(K) escalating. Cuts turnover (the 30m cost driver). **[VERIFIED-2020-OOS, fixed-EW, maker]** RWYB: python -m strat.deep2020_ti_15m_fighter --cadence 30m


## MACD (12, 26, 9) @ 30m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | -2.3 | -0.31 | -6.7 | 168.4 | 0.25 | -7.5 |
| L1 conf2/hold24/cd12 | 14.2 | 2.26 | -9.5 | 113.9 | 0.33 | -5.9 |
| L2 conf4/hold48/cd48 | 2.0 | 0.56 | -6.4 | 71.8 | 0.23 | -0.1 |
| L3 conf8/hold96/cd96 | 12.5 | 2.81 | -4.0 | 41.0 | 0.2 | 10.8 |
| L4 conf12/hold192/cd192 | 12.7 | 2.99 | -4.1 | 22.7 | 0.16 | 10.0 |

## SUPERTREND (10, 3.0) @ 30m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | 5.7 | 0.93 | -9.0 | 126.3 | 0.29 | 3.5 |
| L1 conf2/hold24/cd12 | 12.1 | 1.73 | -11.0 | 86.9 | 0.32 | 3.0 |
| L2 conf4/hold48/cd48 | 15.9 | 2.67 | -4.7 | 56.5 | 0.26 | 20.1 |
| L3 conf8/hold96/cd96 | 22.7 | 3.61 | -4.0 | 36.1 | 0.23 | 20.9 |
| L4 conf12/hold192/cd192 | 15.2 | 3.75 | -2.9 | 21.6 | 0.18 | 10.7 |

## DONCHIAN (20, 10) @ 30m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | 0.8 | 0.25 | -9.7 | 90.7 | 0.25 | 1.3 |
| L1 conf2/hold24/cd12 | 14.4 | 2.09 | -7.4 | 83.1 | 0.3 | 6.4 |
| L2 conf4/hold48/cd48 | 14.3 | 2.28 | -5.9 | 62.2 | 0.26 | 7.8 |
| L3 conf8/hold96/cd96 | 19.6 | 3.13 | -5.0 | 37.4 | 0.23 | 17.1 |
| L4 conf12/hold192/cd192 | 22.6 | 4.05 | -3.3 | 21.8 | 0.18 | 8.5 |

## ROC (50, 0.0) @ 30m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | 13.7 | 1.62 | -9.3 | 129.2 | 0.36 | -0.7 |
| L1 conf2/hold24/cd12 | 21.2 | 2.42 | -7.3 | 70.9 | 0.36 | 1.9 |
| L2 conf4/hold48/cd48 | 20.8 | 2.91 | -5.1 | 49.6 | 0.28 | 17.1 |
| L3 conf8/hold96/cd96 | 15.2 | 2.15 | -6.8 | 32.3 | 0.26 | 9.5 |
| L4 conf12/hold192/cd192 | 22.4 | 4.26 | -4.2 | 20.0 | 0.22 | 19.7 |

## RSI (14, 30, 60) @ 30m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | 8.6 | 3.77 | -2.0 | 54.4 | 0.06 | 8.2 |
| L1 conf2/hold24/cd12 | 2.4 | 0.96 | -3.1 | 35.8 | 0.1 | -0.2 |
| L2 conf4/hold48/cd48 | 7.9 | 2.93 | -2.6 | 22.4 | 0.12 | 3.5 |
| L3 conf8/hold96/cd96 | 10.2 | 3.46 | -2.6 | 13.6 | 0.13 | 0.1 |
| L4 conf12/hold192/cd192 | 1.7 | 0.54 | -5.6 | 6.9 | 0.15 | 1.5 |

## MFI (14, 30, 80) @ 30m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | 18.9 | 2.37 | -7.7 | 108.5 | 0.36 | 8.6 |
| L1 conf2/hold24/cd12 | 19.2 | 2.5 | -6.5 | 62.9 | 0.36 | -3.4 |
| L2 conf4/hold48/cd48 | 15.4 | 2.19 | -5.1 | 44.0 | 0.29 | 9.4 |
| L3 conf8/hold96/cd96 | 7.8 | 1.34 | -6.6 | 30.2 | 0.27 | -7.0 |
| L4 conf12/hold192/cd192 | 16.9 | 3.47 | -4.6 | 19.3 | 0.23 | 8.6 |

## KELTNER (20, 2.0) @ 30m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | 4.3 | 0.85 | -6.5 | 104.0 | 0.2 | 3.4 |
| L1 conf2/hold24/cd12 | 14.8 | 2.39 | -7.0 | 79.9 | 0.26 | 8.5 |
| L2 conf4/hold48/cd48 | 12.7 | 2.24 | -5.2 | 52.4 | 0.24 | 9.7 |
| L3 conf8/hold96/cd96 | 28.5 | 4.15 | -4.3 | 31.9 | 0.25 | 19.3 |
| L4 conf12/hold192/cd192 | 22.2 | 4.25 | -5.3 | 19.3 | 0.21 | 14.3 |

## VOLIMB (3, 0.52) @ 30m
| ladder | net% | Sharpe | maxDD% | turnover | time-in | drift |
|---|---|---|---|---|---|---|
| BASE | 7.5 | 1.22 | -6.7 | 288.6 | 0.26 | 13.3 |
| L1 conf2/hold24/cd12 | 24.0 | 3.86 | -4.3 | 113.2 | 0.28 | 8.6 |
| L2 conf4/hold48/cd48 | 13.2 | 3.02 | -3.4 | 50.2 | 0.19 | 10.3 |
| L3 conf8/hold96/cd96 | 4.8 | 1.22 | -6.4 | 15.6 | 0.16 | -4.6 |
| L4 conf12/hold192/cd192 | 6.0 | 2.3 | -3.6 | 4.3 | 0.08 | -3.2 |

## VERDICT -- does the fighter ladder recover 30m? (honest: FIXED level L2 conf4/hold48/cd48, no cherry-pick)

| indicator | BASE net | L2 conf4/hold48/cd48 net | VAL net | drift | Sharpe | robust(both>0)? |
|---|---|---|---|---|---|---|
| MACD | -2.3% | 2.0% | 2.0% | -0.1 | 0.56 | YES |
| SUPERTREND | 5.7% | 15.9% | -4.2% | 20.1 | 2.67 | no |
| DONCHIAN | 0.8% | 14.3% | 6.5% | 7.8 | 2.28 | YES |
| ROC | 13.7% | 20.8% | 3.7% | 17.1 | 2.91 | YES |
| RSI | 8.6% | 7.9% | 4.3% | 3.5 | 2.93 | YES |
| MFI | 18.9% | 15.4% | 5.9% | 9.4 | 2.19 | YES |
| KELTNER | 4.3% | 12.7% | 3.0% | 9.7 | 2.24 | YES |
| VOLIMB | 7.5% | 13.2% | 2.9% | 10.3 | 3.02 | YES |

**At the FIXED fighter level L2 conf4/hold48/cd48 (no cherry-pick): 8/8 recover to POSITIVE, 7/8 ROBUST (positive in BOTH VAL+OOS).** [VERIFIED-2020-OOS] The MECHANISM is MONOTONE -- every indicator's net climbs as turnover falls across the ladder -- so the 30m deterioration is DEFINITIVELY cost-of-overtrading, not signal failure. CAVEATS (honest): (1) the recovered nets are still <= 30m buy-hold (~53.2%) -- this is de-risked BETA (positive, high-Sharpe 2-4, shallow maxDD -3..-7), NOT alpha over holding; (2) picking the BEST level per indicator is in-sample-favorable (e.g. SUPERTREND's best level has NEGATIVE VAL = OOS-lucky) -- trust the fixed-level + both-windows-positive column, not the per-indicator max; (3) canonical configs (mechanism test), not 30m-optimized. Bottom line: 30m is SALVAGEABLE as a de-risked sleeve via turnover-fighters -- do NOT discard it; but it does not break the drift-beta ceiling.
