# TI CANDIDATES 2020 <-> 2021 -- per family, per TI type, per TF (6/3/3, all 6 TFs)

2020 methodology (per-config per-TI per-TF, 6mo TRAIN / 3mo VAL / 3mo OOS, robust = TRAIN&VAL>0, OOS HELD OUT, fixed-EW u10, ironed sleeve, maker) REPLICATED for 2021. A "CANDIDATE (TI, TF)" cell below = a robust band exists AND the best-robust ironed config is OOS-positive in BOTH years -- i.e. the (type,TF) CLASS/region recurs each year. 2022 NOT included (out of scope until go-ahead). [VERIFIED within-year].

> **CORRECTION (2026-06-17, see TRANSLATION_RIGOR_2020_2021.md):** these cells show that a robust band INDEPENDENTLY
> re-exists in each year -- they do NOT show that the 2020 band TRANSLATES. The cross-year member-transfer test
> found the 2020 band gives ~ZERO predictive lift for 2021 OOS (median lift +0.000 over base rate). NEITHER the
> config-#1 (rank~0) NOR the frozen 2020 band predicts 2021. What carries: the de-risked-beta CLASS + ROLLING
> (recent/trailing) re-selection -- NOT the frozen 2020 region. Read the cells as "the region recurs", not "deploy
> the 2020 band". Also: u10 here is NOT the 2021 universe -- see pit_universe_2021.json (44 coins, 10 new listings).

## Per-TI-type x TF candidate map (cells = best-robust OOS net 2020->2021; CAND if robust+positive both)
| family | TI | 1d | 4h | 2h | 1h | 30m | 15m |
|---|---|---|---|---|---|---|---|
| trend | MACD | 38.6->15.2**C** | 35.9->9.4**C** | 33.0->6.9**C** | 38.1->9.0**C** | 22.2->5.3**C** | 2.9->3.3**C** |
| trend | VORTEX | 40.7->13.6**C** | 30.3->16.9**C** | 30.6->10.1**C** | 27.8->1.1**C** | 20.6->0.6**C** | -0.7->-5.9 |
| trend | ADX | 27.7->4.5**C** | 38.3->6.3**C** | 30.7->6.5**C** | 34.5->15.3**C** | 22.7->-3.0 | - |
| trend | SUPERTREND | 34.5->17.8**C** | 34.6->12.9**C** | 29.3->3.1**C** | 27.9->0.6**C** | 12.0->4.3**C** | - |
| trend | PSAR | 35.0->12.7**C** | 34.3->4.5**C** | 26.7->1.2**C** | 27.7->-1.1 | 14.8->-1.5 | - |
| momentum | TSI | 32.7->7.4**C** | 37.2->6.7**C** | 35.1->-2.1 | 35.7->0.9**C** | 21.6->4.6**C** | 12.4->-1.3 |
| momentum | ROC | 38.1->14.7**C** | 41.6->16.5**C** | 37.5->8.2**C** | 35.0->5.1**C** | 18.4->13.5**C** | 14.7->6.2**C** |
| breakout | DONCHIAN | 38.7->7.2**C** | 32.0->1.3**C** | 29.1->-1.8 | 28.6->-2.9 | 14.0->-0.3 | 2.1->-8.9 |
| breakout | KELTNER | 38.5->9.8**C** | 31.0->5.6**C** | 26.4->2.7**C** | 25.0->-3.0 | 12.6->3.6**C** | -6.7->-3.6 |
| volume | CMF | 33.7->11.2**C** | 28.5->11.9**C** | - | 29.2->1.0**C** | 22.1->-3.7 | 9.8->-12.6 |
| volume | OBV | 46.6->7.3**C** | 37.4->18.1**C** | - | 32.2->6.4**C** | 15.4->9.9**C** | - |
| volume | MFI | 12.0->2.0**C** | 25.1->8.4**C** | - | 40.7->0.1**C** | 18.4->15.1**C** | -0.5->0.5 |
| volume | VOLIMB | 57.2->24.4**C** | 37.1->9.5**C** | - | 34.6->9.5**C** | 20.7->8.8**C** | 14.2->4.5**C** |
| mean-reversion | WILLR | - | 6.4->7.2**C** | 5.4->-1.3 | 24.9->4.6**C** | 13.1->13.5**C** | 23.4->2.6**C** |
| mean-reversion | RSI | 19.3->10.7**C** | 15.2->2.6**C** | 6.4->3.5**C** | 27.5->0.4**C** | 10.7->10.6**C** | 14.7->-0.9 |
| mean-reversion | STOCH | - | 8.5->7.5**C** | 7.2->-1.0 | 25.4->4.3**C** | 11.3->13.0**C** | 22.8->2.4**C** |
| mean-reversion | BBPCT | 8.0->8.7**C** | 4.0->11.1**C** | 10.4->4.1**C** | 21.8->2.3**C** | 19.7->15.2**C** | 23.3->1.7**C** |
| mean-reversion | CCI | 10.4->6.5**C** | 5.0->3.4**C** | 2.2->4.3**C** | 12.9->2.4**C** | 13.0->9.4**C** | 13.3->6.3**C** |

## Emerging candidates per family (robust + OOS-positive in BOTH 2020 AND 2021)
- **trend**: MACD@[1d,4h,2h,1h,30m,15m] (best 1d: 38.6->15.2%, 2.71xBH) ; VORTEX@[1d,4h,2h,1h,30m] (best 4h: 30.3->16.9%, 2.56xBH) ; ADX@[1d,4h,2h,1h] (best 1h: 34.5->15.3%, 2.43xBH) ; SUPERTREND@[1d,4h,2h,1h,30m] (best 1d: 34.5->17.8%, 3.18xBH) ; PSAR@[1d,4h,2h] (best 1d: 35.0->12.7%, 2.27xBH)
- **momentum**: TSI@[1d,4h,1h,30m] (best 1d: 32.7->7.4%, 1.32xBH) ; ROC@[1d,4h,2h,1h,30m,15m] (best 4h: 41.6->16.5%, 2.5xBH)
- **breakout**: DONCHIAN@[1d,4h] (best 1d: 38.7->7.2%, 1.29xBH) ; KELTNER@[1d,4h,2h,30m] (best 1d: 38.5->9.8%, 1.75xBH)
- **volume**: CMF@[1d,4h,1h] (best 4h: 28.5->11.9%, 0.87xBH) ; OBV@[1d,4h,1h,30m] (best 4h: 37.4->18.1%, 1.32xBH) ; MFI@[1d,4h,1h,30m] (best 30m: 18.4->15.1%, 0.97xBH) ; VOLIMB@[1d,4h,1h,30m,15m] (best 1d: 57.2->24.4%, 2.14xBH)
- **mean-reversion**: WILLR@[4h,1h,30m,15m] (best 30m: 13.1->13.5%, 1.99xBH) ; RSI@[1d,4h,2h,1h,30m] (best 1d: 19.3->10.7%, 1.91xBH) ; STOCH@[4h,1h,30m,15m] (best 30m: 11.3->13.0%, 1.91xBH) ; BBPCT@[1d,4h,2h,1h,30m,15m] (best 30m: 19.7->15.2%, 2.24xBH) ; CCI@[1d,4h,2h,1h,30m,15m] (best 30m: 13.0->9.4%, 1.38xBH)

## HEADLINE: 80 (TI x TF) cross-year candidates (robust + OOS-positive 2020 AND 2021) across 18 TIs x 6 TFs. The candidate is the BAND/region per (type, TF); the tradeable config is rolling-picked from it. xBH<1 expected (long-only de-risked beta under-participates the bull). 2020->2021 transfer: the robust BAND reproduces where a candidate exists; config #1 rank does not (use the band, not the #1). NEXT (on go-ahead): all-weather / 2022 bear.