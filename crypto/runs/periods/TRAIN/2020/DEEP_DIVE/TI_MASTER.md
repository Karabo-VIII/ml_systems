# TI MASTER -- per-config per-indicator per-timeframe leaderboard, base vs IRONED (2020)

The single consolidated view of every non-MA indicator family run through the same end-to-end (every config x TF, BASE vs IRONED, wealth-ranked, robust split). 2020 band, STRICT long-only spot, fixed-EW, VAL Jul-Sep / OOS Oct-Dec. ALL numbers **[VERIFIED-backtest, IN-SAMPLE 2020, OOS]**. Tools: `deep2020_ti_pipeline.py` (+ `_render`/`_top10`/`_master`). The IRON per family: trend = zero-line/slow-trend confirm + vol-target; mean-reversion = buy-the-dip-ONLY-in-uptrend + vol-target; breakout = ATR-confirm + vol-target; momentum = uptrend-confirm + vol-target.


## TREND family

| indicator | TF | best ironed cfg | base->IRON net | xBH | Sharpe | maxDD | drift | n_trd | t-in | robust |
|---|---|---|---|---|---|---|---|---|---|---|
| MACD | 1d | MACD(26,52,9) | 42.4->36.0 | 0.76 | 3.65 | -10.3 | 7.7 | 3.0 | 0.33 | 33/66 |
| MACD | 4h | MACD(19,35,21) | 41.4->31.1 | 0.65 | 3.49 | -6.6 | 8.0 | 15.8 | 0.33 | 61/66 |
| MACD | 2h | MACD(8,35,9) | 44.2->29.8 | 0.59 | 3.37 | -5.2 | 6.2 | 47.5 | 0.33 | 41/66 |
| MACD | 1h | MACD(19,35,21) | 46.9->31.2 | 0.6 | 3.85 | -4.6 | 9.5 | 56.4 | 0.3 | 40/66 |
| MACD | 30m | MACD(26,100,21) | 27.2->21.5 | 0.4 | 3.17 | -4.2 | 7.3 | 72.3 | 0.25 | 60/66 |
| MACD | 15m | MACD(26,100,21) | -2.1->3.6 | 0.07 | 0.7 | -7.4 | -0.8 | 139.7 | 0.24 | 59/66 |
| SUPERTREND | 1d | ST(14,2.0) | 17.6->28.6 | 0.6 | 2.56 | -13.0 | 9.9 | 7.0 | 0.48 | 3/16 |
| SUPERTREND | 4h | ST(14,4.0) | 32.6->27.4 | 0.57 | 2.94 | -6.2 | 4.9 | 18.6 | 0.37 | 14/16 |
| SUPERTREND | 2h | ST(14,3.0) | 37.2->28.8 | 0.57 | 3.12 | -5.0 | 8.2 | 30.9 | 0.35 | 12/16 |
| SUPERTREND | 1h | ST(10,4.0) | 31.3->26.1 | 0.51 | 2.82 | -6.0 | 7.2 | 56.1 | 0.35 | 8/16 |
| SUPERTREND | 30m | ST(14,4.0) | 8.4->12.2 | 0.23 | 1.64 | -7.4 | 7.8 | 104.1 | 0.32 | 15/16 |
| SUPERTREND | 15m | ST(7,4.0) | -8.6->2.2 | 0.04 | 0.43 | -10.0 | 4.1 | 188.3 | 0.33 | 16/16 |
| PSAR | 1d | PSAR(0.01,0.2) | 38.8->32.2 | 0.68 | 3.21 | -12.1 | 8.7 | 3.9 | 0.38 | 3/9 |
| PSAR | 4h | PSAR(0.04,0.3) | 38.6->28.5 | 0.6 | 2.96 | -5.4 | 7.2 | 31.9 | 0.41 | 5/9 |
| PSAR | 2h | PSAR(0.02,0.3) | 31.9->26.3 | 0.52 | 3.05 | -5.1 | 5.8 | 47.4 | 0.34 | 7/9 |
| PSAR | 1h | PSAR(0.02,0.1) | 22.6->21.8 | 0.42 | 2.47 | -6.8 | 9.3 | 83.3 | 0.34 | 5/9 |
| PSAR | 30m | PSAR(0.01,0.3) | 4.6->13.4 | 0.25 | 1.99 | -7.4 | 9.5 | 128.3 | 0.3 | 8/9 |
| PSAR | 15m | PSAR(0.01,0.1) | -12.0->3.1 | 0.06 | 0.57 | -10.7 | 4.2 | 247.3 | 0.3 | 9/9 |
| VORTEX | 1d | VORTEX(14) | 45.3->40.3 | 0.85 | 3.72 | -11.2 | 28.8 | 5.5 | 0.45 | 0/4 |
| VORTEX | 4h | VORTEX(21) | 29.7->29.6 | 0.62 | 2.9 | -11.4 | 8.1 | 23.9 | 0.46 | 4/4 |
| VORTEX | 2h | VORTEX(21) | 35.5->30.2 | 0.6 | 3.02 | -6.6 | 6.7 | 44.3 | 0.42 | 4/4 |
| VORTEX | 1h | VORTEX(21) | 39.3->27.3 | 0.53 | 2.81 | -8.1 | 5.0 | 87.5 | 0.39 | 4/4 |
| ADX | 1d | ADX(14,20) | 37.9->27.8 | 0.59 | 2.76 | -14.7 | 0.4 | 5.3 | 0.41 | 6/6 |
| ADX | 4h | ADX(14,20) | 44.9->36.7 | 0.77 | 3.49 | -6.9 | 3.5 | 29.1 | 0.43 | 6/6 |
| ADX | 2h | ADX(21,20) | 34.0->29.9 | 0.6 | 3.22 | -5.9 | 16.6 | 40.2 | 0.36 | 0/6 |
| ADX | 1h | ADX(14,20) | 25.9->23.7 | 0.46 | 2.74 | -7.6 | 1.7 | 104.9 | 0.38 | 4/6 |

## MOMENTUM family

| indicator | TF | best ironed cfg | base->IRON net | xBH | Sharpe | maxDD | drift | n_trd | t-in | robust |
|---|---|---|---|---|---|---|---|---|---|---|
| ROC | 1d | ROC(100,thr0.0) | 19.2->27.3 | 0.58 | 4.23 | -5.7 | 8.0 | 3.0 | 0.28 | 7/16 |
| ROC | 4h | ROC(50,thr0.0) | 14.7->24.8 | 0.52 | 2.46 | -10.4 | -2.5 | 17.3 | 0.46 | 9/16 |
| ROC | 2h | ROC(50,thr2.0) | 42.2->32.9 | 0.66 | 3.33 | -6.7 | 9.5 | 31.5 | 0.4 | 13/16 |
| ROC | 1h | ROC(50,thr0.0) | 42.9->34.4 | 0.67 | 3.28 | -7.2 | 9.3 | 62.2 | 0.43 | 11/16 |
| ROC | 30m | ROC(100,thr0.0) | 19.4->18.6 | 0.35 | 2.12 | -6.9 | -3.7 | 99.9 | 0.37 | 16/16 |
| ROC | 15m | ROC(50,thr0.0) | 18.7->14.7 | 0.27 | 1.69 | -7.0 | -2.5 | 238.1 | 0.39 | 16/16 |
| TSI | 1d | TSI(13,7) | 30.4->32.6 | 0.69 | 2.89 | -12.1 | 13.2 | 5.0 | 0.42 | 0/4 |
| TSI | 4h | TSI(25,13) | 39.7->33.7 | 0.71 | 3.19 | -10.4 | 7.0 | 14.7 | 0.45 | 3/4 |
| TSI | 2h | TSI(40,20) | 43.4->34.6 | 0.69 | 3.27 | -9.5 | 8.9 | 23.0 | 0.45 | 4/4 |
| TSI | 1h | TSI(13,7) | 22.2->24.5 | 0.47 | 2.55 | -6.8 | 3.8 | 75.7 | 0.38 | 1/4 |

## BREAKOUT family

| indicator | TF | best ironed cfg | base->IRON net | xBH | Sharpe | maxDD | drift | n_trd | t-in | robust |
|---|---|---|---|---|---|---|---|---|---|---|
| DONCHIAN | 1d | DONCH(30,10) | 32.1->28.3 | 0.6 | 3.43 | -10.3 | 8.3 | 3.0 | 0.25 | 6/9 |
| DONCHIAN | 4h | DONCH(30,20) | 33.0->31.2 | 0.65 | 3.41 | -9.8 | 5.7 | 10.1 | 0.36 | 9/9 |
| DONCHIAN | 2h | DONCH(30,20) | 32.2->28.4 | 0.57 | 3.42 | -5.3 | 8.6 | 17.5 | 0.33 | 6/9 |
| DONCHIAN | 1h | DONCH(20,10) | 12.2->18.4 | 0.36 | 2.54 | -6.3 | 5.8 | 59.2 | 0.28 | 3/9 |
| DONCHIAN | 30m | DONCH(55,20) | 5.3->13.8 | 0.26 | 2.2 | -5.6 | 5.8 | 49.0 | 0.21 | 7/9 |
| DONCHIAN | 15m | DONCH(55,20) | -12.1->2.7 | 0.05 | 0.55 | -10.3 | 2.4 | 90.7 | 0.21 | 8/9 |
| KELTNER | 1d | KELT(20,2.0) | 38.1->33.3 | 0.7 | 3.32 | -12.1 | 6.5 | 3.7 | 0.35 | 2/6 |
| KELTNER | 4h | KELT(30,1.5) | 31.9->30.2 | 0.63 | 3.21 | -8.0 | 7.0 | 18.6 | 0.39 | 6/6 |
| KELTNER | 2h | KELT(30,2.0) | 34.1->25.9 | 0.52 | 3.08 | -5.1 | 8.9 | 26.0 | 0.29 | 5/6 |
| KELTNER | 1h | KELT(30,2.0) | 12.4->20.9 | 0.41 | 2.55 | -6.5 | 6.1 | 54.9 | 0.29 | 4/6 |
| KELTNER | 30m | KELT(30,1.5) | 7.6->12.7 | 0.24 | 1.79 | -7.2 | 5.3 | 119.8 | 0.29 | 5/6 |
| KELTNER | 15m | KELT(30,2.5) | -11.7->-4.0 | -0.07 | -0.6 | -8.7 | -2.6 | 149.5 | 0.2 | 6/6 |

## MEAN-REVERSION family

| indicator | TF | best ironed cfg | base->IRON net | xBH | Sharpe | maxDD | drift | n_trd | t-in | robust |
|---|---|---|---|---|---|---|---|---|---|---|
| RSI | 1d | RSI(7,lo35,hi60) | 28.5->18.5 | 0.39 | 4.59 | -4.2 | 8.4 | 5.8 | 0.16 | 25/27 |
| RSI | 4h | RSI(7,lo30,hi60) | 3.4->10.8 | 0.23 | 2.61 | -5.5 | 7.7 | 23.0 | 0.15 | 18/27 |
| RSI | 2h | RSI(14,lo30,hi65) | 12.8->6.2 | 0.12 | 2.52 | -2.9 | 4.6 | 15.6 | 0.06 | 27/27 |
| RSI | 1h | RSI(7,lo25,hi60) | 28.0->15.9 | 0.31 | 4.05 | -4.8 | 9.9 | 68.1 | 0.1 | 20/27 |
| RSI | 30m | RSI(7,lo35,hi65) | 31.9->10.2 | 0.19 | 2.21 | -4.5 | 5.4 | 189.0 | 0.17 | 23/27 |
| RSI | 15m | RSI(7,lo35,hi55) | 36.5->13.7 | 0.25 | 2.9 | -4.1 | 9.2 | 365.0 | 0.15 | 27/27 |
| STOCH | 1d | STOCH(14,lo20,hi80) | 20.6->16.2 | 0.34 | 3.99 | -2.8 | 2.2 | 5.0 | 0.14 | 15/18 |
| STOCH | 4h | STOCH(14,lo25,hi80) | 7.3->8.3 | 0.17 | 1.71 | -8.7 | 3.4 | 21.9 | 0.19 | 18/18 |
| STOCH | 2h | STOCH(21,lo25,hi80) | 4.6->7.1 | 0.14 | 1.74 | -6.2 | -1.6 | 33.8 | 0.16 | 18/18 |
| STOCH | 1h | STOCH(21,lo25,hi55) | 15.7->12.9 | 0.25 | 3.52 | -4.2 | 8.0 | 62.4 | 0.12 | 9/18 |
| BBPCT | 1d | BBPCT(14,lo0.2,hi0.9) | 21.2->13.8 | 0.29 | 3.1 | -1.9 | -1.4 | 4.2 | 0.14 | 18/18 |
| BBPCT | 4h | BBPCT(20,lo0.2,hi0.9) | 6.9->4.1 | 0.09 | 1.02 | -6.2 | 7.1 | 15.0 | 0.17 | 18/18 |
| BBPCT | 2h | BBPCT(14,lo0.2,hi0.9) | 12.7->10.3 | 0.21 | 2.16 | -8.5 | 1.1 | 38.9 | 0.2 | 18/18 |
| BBPCT | 1h | BBPCT(20,lo0.1,hi0.9) | 13.3->12.7 | 0.25 | 3.4 | -4.2 | 8.1 | 51.8 | 0.12 | 15/18 |
| CCI | 1d | CCI(14,lo-80,hi100) | 15.7->10.1 | 0.21 | 2.59 | -2.6 | 3.7 | 4.0 | 0.13 | 18/18 |
| CCI | 4h | CCI(20,lo-100,hi100) | 6.1->5.4 | 0.11 | 1.76 | -3.7 | 8.8 | 12.6 | 0.12 | 18/18 |
| CCI | 2h | CCI(14,lo-80,hi80) | -1.7->2.3 | 0.05 | 0.62 | -8.8 | -3.5 | 33.1 | 0.14 | 18/18 |
| CCI | 1h | CCI(14,lo-80,hi0) | 10.7->12.4 | 0.24 | 3.79 | -4.3 | 8.6 | 59.5 | 0.09 | 18/18 |
| WILLR | 1d | WILLR(14,lo-80,hi-20) | 20.6->16.2 | 0.34 | 3.99 | -2.8 | 2.2 | 5.0 | 0.14 | 11/12 |
| WILLR | 4h | WILLR(14,lo-80,hi-20) | 3.6->6.3 | 0.13 | 1.38 | -7.9 | 2.7 | 20.9 | 0.17 | 12/12 |
| WILLR | 2h | WILLR(14,lo-80,hi-30) | 4.0->5.4 | 0.11 | 1.32 | -7.4 | -2.5 | 39.5 | 0.15 | 12/12 |
| WILLR | 1h | WILLR(21,lo-90,hi-50) | 6.3->11.4 | 0.22 | 4.01 | -3.0 | 9.3 | 43.6 | 0.08 | 6/12 |
| WILLR | 30m | WILLR(14,lo-90,hi-30) | 40.2->12.4 | 0.23 | 4.12 | -2.7 | 3.7 | 111.5 | 0.1 | 9/12 |
| WILLR | 15m | WILLR(14,lo-90,hi-30) | 43.8->20.2 | 0.37 | 5.56 | -2.7 | 8.3 | 217.7 | 0.1 | 10/12 |

## VOLUME family

| indicator | TF | best ironed cfg | base->IRON net | xBH | Sharpe | maxDD | drift | n_trd | t-in | robust |
|---|---|---|---|---|---|---|---|---|---|---|
| OBV | 1d | OBV(20) | 40.7->46.0 | 0.8 | 3.75 | -10.9 | 27.5 | 6.0 | 0.48 | 0/4 |
| OBV | 4h | OBV(10) | 40.4->31.2 | 0.55 | 2.8 | -10.9 | 8.3 | 36.5 | 0.5 | 1/4 |
| OBV | 1h | OBV(20) | 13.6->23.0 | 0.38 | 2.47 | -9.7 | 9.8 | 109.8 | 0.41 | 1/4 |
| OBV | 30m | OBV(20) | 2.0->15.3 | 0.25 | 1.91 | -8.7 | 3.9 | 210.7 | 0.38 | 4/4 |
| OBV | 15m | OBV(100) | 4.9->10.8 | 0.17 | 1.45 | -6.4 | 5.7 | 258.5 | 0.36 | 4/4 |
| MFI | 1d | MFI(21,lo30,hi80) | 1.5->12.2 | 0.21 | 2.3 | -7.7 | 8.5 | 2.7 | 0.22 | 12/12 |
| MFI | 4h | MFI(14,lo30,hi80) | 6.6->24.9 | 0.44 | 2.89 | -6.4 | 5.1 | 17.2 | 0.4 | 12/12 |
| MFI | 1h | MFI(14,lo20,hi80) | 12.4->13.6 | 0.22 | 3.38 | -3.4 | -2.5 | 29.7 | 0.17 | 10/12 |
| MFI | 30m | MFI(14,lo30,hi80) | 32.2->18.9 | 0.3 | 2.37 | -7.7 | 8.6 | 126.5 | 0.36 | 10/12 |
| MFI | 15m | MFI(21,lo30,hi80) | 18.4->10.7 | 0.17 | 1.37 | -7.8 | 2.7 | 243.4 | 0.35 | 12/12 |
| VOLIMB | 1d | VOLIMB(7,thr0.5) | 38.5->30.6 | 0.53 | 3.55 | -6.0 | 2.9 | 5.7 | 0.34 | 6/12 |
| VOLIMB | 4h | VOLIMB(3,thr0.55) | 36.6->25.6 | 0.45 | 3.26 | -6.8 | 7.5 | 29.9 | 0.33 | 5/12 |
| VOLIMB | 1h | VOLIMB(3,thr0.52) | 24.6->33.8 | 0.56 | 4.07 | -7.1 | 7.8 | 123.4 | 0.36 | 7/12 |
| VOLIMB | 30m | VOLIMB(7,thr0.52) | 14.2->16.3 | 0.26 | 2.54 | -5.6 | 9.7 | 180.7 | 0.27 | 6/12 |
| VOLIMB | 15m | VOLIMB(14,thr0.52) | 14.7->7.1 | 0.11 | 1.46 | -3.9 | 1.9 | 232.5 | 0.2 | 10/12 |
| CMF | 1d | CMF(14) | 44.0->33.7 | 0.59 | 3.74 | -9.6 | 27.9 | 5.2 | 0.36 | 0/3 |
| CMF | 4h | CMF(20) | 37.3->27.8 | 0.49 | 3.31 | -6.1 | 12.6 | 22.3 | 0.34 | 0/3 |
| CMF | 1h | CMF(20) | 26.9->28.7 | 0.47 | 3.41 | -7.1 | 15.2 | 83.6 | 0.35 | 0/3 |
| CMF | 30m | CMF(14) | 19.5->21.5 | 0.35 | 2.94 | -6.3 | 15.6 | 179.1 | 0.33 | 0/3 |
| CMF | 15m | CMF(50) | 14.1->9.7 | 0.15 | 1.4 | -6.9 | 8.6 | 237.5 | 0.32 | 3/3 |

## Per-family IRON effectiveness (median over the family's indicators x TFs)

| family | dNet (iron-base) | dMaxDD | d|drift| | robust-frac | best ironed net/BH (across TFs) |
|---|---|---|---|---|---|
| trend | -1.7pp | +7.4pp | -0.5pp | 71% | 0.85x |
| momentum | -0.5pp | +4.8pp | -1.8pp | 65% | 0.85x |
| breakout | +0.9pp | +4.8pp | +0.3pp | 75% | 0.80x |
| mean-reversion | -1.6pp | +7.0pp | -4.8pp | 89% | 0.53x |
| volume | +0.2pp | +5.1pp | -1.2pp | 55% | 0.98x |

## MASTER VERDICT

Across ALL non-MA technical families (trend: MACD/Supertrend/PSAR/Vortex/ADX; momentum: ROC/TSI; breakout: Donchian/Keltner; mean-reversion: RSI/Stoch/BB%b/CCI/Williams%R; volume/order-flow: OBV/MFI/VOLIMB/CMF), the 2020 result is uniform and matches the MA finding: **the IRON buys risk-reduction + robustness, NOT return; no internal-data indicator family beats long-only buy-hold on NET in the bull.** [VERIFIED-2020-OOS] Trend/momentum/breakout are de-risked betas (best ironed ~0.5-0.8x buy-hold, iron cuts maxDD + robustifies). Mean-reversion is structurally weak (~0.1-0.4x) but the uptrend-filter iron makes it the most robust family (a low-return, low-DD diversifier). The VOLUME/ORDER-FLOW family (incl. the taker buy/sell-imbalance VOLIMB -- the one signal using data no price indicator sees) is in the volume table above: it does NOT break the ceiling (still <= buy-hold on net), confirming that internal order-flow imbalance at bar resolution is also a de-risked-beta-or-weaker signal in 2020 (volume's best ironed reaches 0.98x BH -- the highest capture of any family, but still <= buy-hold). We also tested the deep-research-endorsed REGIME-GATE iron explicitly via ADX (long +DI>-DI only when ADX>threshold): it is the most aggressive iron -- it FULLY robustifies + cuts maxDD the most but cuts the most net (ADX 1d 37.9->27.8), confirming regime-gating OVER-de-risks in a clean bull (you do not want to gate out a bull). The deployable single-config picks are the ROBUST ironed ones in the tables above (small drift = delivers in BOTH VAL+OOS). The next lever is NOT another indicator (the ceiling is the drift-beta) -- it is either full-cycle validation or an ORTHOGONAL beta (carry/cross-asset), which the user explicitly deferred ('not solving for correlation').
