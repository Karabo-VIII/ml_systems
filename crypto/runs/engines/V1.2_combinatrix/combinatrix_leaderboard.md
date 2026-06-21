# V1.2 COMBINATRIX -- LEADERBOARD (structured search: singles / pairs / beam-triples / a1a2)

> Ranked by DEFLATED (OOS+IRON)/2 net wealth. Every combo gated: block-bootstrap p05 (vs best-single
> AND vs buy-hold), CSCV-PBO, White's Reality Check + deflated-Sharpe (n_trials = #combos tried).
> 2020 VAL = SELECT / 2020 OOS = REPORT / 2021 IRON = CONFIRM / 2022+ SEALED (asserted never loaded).

- **Combos evaluated:** 275 (pairs=75, triples=192, a1a2=8); n_trials(deflation)=275
- **Breadth:** n_names=27 distinct u100 names (>=20 floor met).
- **CSCV-PBO** over all combos = **0.7063** (OVERFIT).  **White's Reality Check** (best combo vs BH) p=**0.893** (FAILS deflation).
- **Buy-hold benchmark:** DEV val=25.6 OOS=32.5 dd=-15.6 | IRON-2021 H2=61.9.
- **VAL-selected combo:** `OR(CKSP@1d+ULCER@1d+BBSQZ@1d)`.

## SINGLES recap (the cross-mechanism inputs -- all de-risked betas)

| input | mechanism | OOS net | xBH | IRON net |
|---|---|--:|--:|--:|
| CKSP@1d | trend | 20.1 | 0.618 | 24.1 |
| KELTNER@4h | breakout | 15.9 | 0.489 | 23.7 |
| AO@4h | momentum | 17.7 | 0.545 | 15.3 |
| RSI@1h | mean-reversion | 12.5 | 0.385 | 15.0 |
| ULCER@1d | volatility-sizing | 16.1 | 0.495 | 18.5 |
| ADX@4h | trend2 | 11.5 | 0.354 | 7.7 |
| BBSQZ@1d | volatility2 | 9.7 | 0.298 | 10.8 |

## TOP-25 COMBOS (by deflated OOS+IRON net)

| # | combo | tier | val net (xBH) | OOS net (xBH) | IRON net (xBH) | maxDD | p05 vs BH | p05 vs SGL | PBO | defl net | defl OOS+IRON | n |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `OR(KELTNER@4h+ULCER@1d+BBSQZ@1d)` | triple | 21.1 (0.824) | 26.5 (0.815) | 46.2 (0.746) | -11.0 | -43.086 | -2.504 | 0.7063 | 2.5 | 3.4 | 27 |
| 2 | `OR(RSI@1h+BBSQZ@1d)` | pair | 11.3 (0.441) | 21.5 (0.662) | 18.5 (0.299) | -7.5 | -62.828 | 2.144 | 0.7063 | 3.4 | 3.2 | 27 |
| 3 | `OR(KELTNER@4h+ULCER@1d)` | pair | 19.0 (0.742) | 25.2 (0.775) | 45.1 (0.729) | -11.3 | -45.486 | -3.461 | 0.7063 | 2.2 | 3.1 | 27 |
| 4 | `AND_2(KELTNER@4h+ADX@4h)` | pair | 11.1 (0.434) | 17.7 (0.545) | 14.3 (0.231) | -6.6 | -66.792 | -6.617 | 0.7063 | 3.4 | 3.0 | 27 |
| 5 | `WEIGHTED_0.67(KELTNER@4h+ADX@4h)` | pair | 11.1 (0.434) | 17.7 (0.545) | 14.3 (0.231) | -6.6 | -66.792 | -6.617 | 0.7063 | 3.4 | 3.0 | 27 |
| 6 | `GATE[KELTNER@4h|ADX@4h](KELTNER@4h+ADX@4h)` | pair | 11.1 (0.434) | 17.7 (0.545) | 14.3 (0.231) | -6.6 | -66.792 | -6.617 | 0.7063 | 3.4 | 3.0 | 27 |
| 7 | `OR(CKSP@1d+KELTNER@4h+BBSQZ@1d)` | triple | 21.7 (0.848) | 24.4 (0.751) | 53.5 (0.864) | -13.3 | -42.687 | -6.643 | 0.7063 | 1.9 | 3.0 | 27 |
| 8 | `GATE[KELTNER@4h|ADX@4h](CKSP@1d+KELTNER@4h+ADX@4h)` | triple | 11.1 (0.434) | 17.7 (0.545) | 14.3 (0.231) | -6.6 | -66.792 | -17.483 | 0.7063 | 3.4 | 3.0 | 27 |
| 9 | `GATE[KELTNER@4h|ADX@4h](KELTNER@4h+ULCER@1d+ADX@4h)` | triple | 11.1 (0.434) | 17.7 (0.545) | 14.3 (0.231) | -6.6 | -66.792 | -12.059 | 0.7063 | 3.4 | 3.0 | 27 |
| 10 | `OR(CKSP@1d+KELTNER@4h+ULCER@1d)` | triple | 22.8 (0.891) | 25.8 (0.794) | 50.6 (0.817) | -13.4 | -40.152 | -6.135 | 0.7063 | 2.0 | 2.9 | 27 |
| 11 | `OR(KELTNER@4h+BBSQZ@1d)` | pair | 16.8 (0.656) | 23.0 (0.708) | 37.2 (0.601) | -10.5 | -49.344 | 2.088 | 0.7063 | 2.1 | 2.8 | 27 |
| 12 | `OR(CKSP@1d+KELTNER@4h+RSI@1h)` | triple | 18.2 (0.711) | 25.4 (0.782) | 48.6 (0.785) | -12.4 | -38.713 | -7.783 | 0.7063 | 1.9 | 2.8 | 27 |
| 13 | `OR(CKSP@1d+KELTNER@4h)` | pair | 20.9 (0.816) | 23.9 (0.735) | 50.5 (0.816) | -13.3 | -43.672 | -6.899 | 0.7063 | 1.8 | 2.7 | 27 |
| 14 | `OR(CKSP@1d+RSI@1h+BBSQZ@1d)` | triple | 20.9 (0.816) | 25.1 (0.772) | 33.2 (0.536) | -10.2 | -46.743 | -3.663 | 0.7063 | 2.3 | 2.7 | 27 |
| 15 | `OR(KELTNER@4h+AO@4h+ULCER@1d)` | triple | 21.1 (0.824) | 24.9 (0.766) | 45.3 (0.732) | -12.8 | -43.186 | -4.592 | 0.7063 | 1.9 | 2.6 | 27 |
| 16 | `OR(AO@4h+BBSQZ@1d)` | pair | 20.5 (0.801) | 23.7 (0.729) | 31.0 (0.501) | -10.0 | -48.475 | 1.957 | 0.7063 | 2.2 | 2.5 | 27 |
| 17 | `AND_2(CKSP@1d+ADX@4h+BBSQZ@1d)` | triple | 17.7 (0.691) | 19.3 (0.594) | 12.4 (0.2) | -7.0 | -64.958 | -11.811 | 0.7063 | 3.1 | 2.5 | 27 |
| 18 | `MAJORITY(CKSP@1d+ADX@4h+BBSQZ@1d)` | triple | 17.7 (0.691) | 19.3 (0.594) | 12.4 (0.2) | -7.0 | -64.958 | -11.811 | 0.7063 | 3.1 | 2.5 | 27 |
| 19 | `WEIGHTED_0.5(CKSP@1d+ADX@4h+BBSQZ@1d)` | triple | 17.7 (0.691) | 19.3 (0.594) | 12.4 (0.2) | -7.0 | -64.958 | -11.811 | 0.7063 | 3.1 | 2.5 | 27 |
| 20 | `OR(AO@4h+ULCER@1d+BBSQZ@1d)` | triple | 23.4 (0.914) | 24.7 (0.76) | 40.0 (0.646) | -12.2 | -45.856 | -3.74 | 0.7063 | 1.9 | 2.5 | 27 |
| 21 | `OR(CKSP@1d+RSI@1h)` | pair | 20.1 (0.785) | 24.2 (0.745) | 28.7 (0.464) | -10.2 | -47.941 | -4.443 | 0.7063 | 2.1 | 2.3 | 27 |
| 22 | `OR(AO@4h+ULCER@1d)` | pair | 21.8 (0.852) | 24.1 (0.742) | 37.0 (0.598) | -12.3 | -46.295 | -4.105 | 0.7063 | 1.8 | 2.3 | 27 |
| 23 | `OR(AO@4h+RSI@1h+ULCER@1d)` | triple | 18.0 (0.703) | 24.4 (0.751) | 39.9 (0.645) | -11.1 | -43.968 | -5.585 | 0.7063 | 1.8 | 2.3 | 27 |
| 24 | `OR(CKSP@1d+BBSQZ@1d)` | pair | 27.1 (1.059) | 21.1 (0.649) | 29.1 (0.47) | -10.2 | -58.742 | 0.115 | 0.7063 | 1.8 | 2.2 | 27 |
| 25 | `AND_2(AO@4h+ADX@4h)` | pair | 11.8 (0.461) | 17.4 (0.535) | 13.0 (0.21) | -6.2 | -66.184 | -13.185 | 0.7063 | 2.5 | 2.2 | 27 |

_* = a1a2 within-indicator config-combo (REDUNDANT -- flagged data-snoop trap; shown for completeness)._

## HONEST VERDICT

**NULL (prior-consistent, honest).** NO combo beats its best constituent single AND long-only buy-hold on OOS+IRON net after deflation with PBO low + p05>0. The de-risked-beta wall holds under
combination: CSCV-PBO=0.7063 (overfit) and White's Reality Check p=0.893 (best combo fails the
multiple-comparisons deflation). Combining de-risked-beta TI engines does NOT manufacture alpha.

## CONDITIONING FRONTIER (cuts OOS maxDD vs buy-hold + robust, WITHOUT beating net)

The project's VALIDATED use of TIs (regime/AND-gating = drawdown insurance, per efb23a4b):

| combo | OOS maxDD (BH=-15.6) | OOS net | IRON net | robust |
|---|--:|--:|--:|:--:|
| `AND_3(CKSP@1d+RSI@1h+ADX@4h)` | -1.1 | 4.4 | 7.4 | True |
| `WEIGHTED_0.67(CKSP@1d+RSI@1h+ADX@4h)` | -1.1 | 4.4 | 7.4 | True |
| `AND_3(RSI@1h+ULCER@1d+ADX@4h)` | -1.3 | 3.2 | 7.5 | True |
| `WEIGHTED_0.67(RSI@1h+ULCER@1d+ADX@4h)` | -1.3 | 3.2 | 7.5 | True |
| `AND_2(RSI@1h+ADX@4h)` | -2.2 | 5.3 | 6.6 | True |
| `WEIGHTED_0.67(RSI@1h+ADX@4h)` | -2.2 | 5.3 | 6.6 | True |
| `GATE[RSI@1h|ADX@4h](RSI@1h+ADX@4h)` | -2.2 | 5.3 | 6.6 | True |
| `GATE[RSI@1h|ADX@4h](CKSP@1d+RSI@1h+ADX@4h)` | -2.2 | 5.3 | 6.6 | True |
| `GATE[RSI@1h|ADX@4h](RSI@1h+ULCER@1d+ADX@4h)` | -2.2 | 5.3 | 6.6 | True |
| `AND_2(CKSP@1d+RSI@1h)` | -2.9 | 8.7 | 10.7 | True |
