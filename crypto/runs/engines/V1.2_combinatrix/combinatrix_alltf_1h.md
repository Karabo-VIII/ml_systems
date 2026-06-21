# V1.2 COMBINATRIX -- LEADERBOARD (structured search: singles / pairs / beam-triples / a1a2)  [combo_cad=1h]

> Ranked by DEFLATED (OOS+IRON)/2 net wealth. Every combo gated: block-bootstrap p05 (vs best-single
> AND vs buy-hold), CSCV-PBO, White's Reality Check + deflated-Sharpe (n_trials = #combos tried).
> 2020 VAL = SELECT / 2020 OOS = REPORT / 2021 IRON = CONFIRM / 2022+ SEALED (asserted never loaded).

- **Combos evaluated:** 285 (pairs=75, triples=202, a1a2=8); n_trials(deflation)=285
- **Breadth:** n_names=27 distinct u100 names (>=20 floor met).
- **CSCV-PBO** over all combos = **0.0754** (not overfit).  **White's Reality Check** (best combo vs BH) p=**0.245** (FAILS deflation).
- **Buy-hold benchmark:** DEV val=25.4 OOS=39.4 dd=-16.0 | IRON-2021 H2=64.3.
- **VAL-selected combo:** `OR(CKSP@1d+ADX@4h)`.

## SINGLES recap (the cross-mechanism inputs -- all de-risked betas)

| input | mechanism | OOS net | xBH | IRON net |
|---|---|--:|--:|--:|
| CKSP@1d | trend | 51.3 | 1.302 | 113.5 |
| KELTNER@4h | breakout | 73.1 | 1.855 | 200.7 |
| AO@4h | momentum | 59.7 | 1.515 | 122.1 |
| RSI@1h | mean-reversion | 18.2 | 0.462 | 1.4 |
| ULCER@1d | volatility-sizing | 34.9 | 0.886 | 58.4 |
| ADX@4h | trend2 | 54.5 | 1.383 | 70.8 |
| BBSQZ@1d | volatility2 | 21.8 | 0.553 | 43.8 |

## TOP-25 COMBOS (by deflated OOS+IRON net)

| # | combo | tier | val net (xBH) | OOS net (xBH) | IRON net (xBH) | maxDD | p05 vs BH | p05 vs SGL | PBO | defl net | defl OOS+IRON | n |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `AND_2(KELTNER:KELT(20,1.5)+KELTNER:KELT(30,2.5))` | a1a2* | 48.6 (1.913) | 65.3 (1.657) | 175.8 (2.734) | -3.4 | -21.193 | -17.079 | 0.0754 | 62.3 | 115.1 | 27 |
| 2 | `AND_2(KELTNER:KELT(20,2.0)+KELTNER:KELT(30,2.0))` | a1a2* | 50.5 (1.988) | 66.6 (1.69) | 166.9 (2.596) | -3.6 | -19.401 | -0.858 | 0.0754 | 62.9 | 110.2 | 27 |
| 3 | `AND_2(CKSP@1d+KELTNER@4h+ADX@4h)` | triple | 60.4 (2.378) | 68.9 (1.749) | 150.2 (2.336) | -4.3 | -15.836 | 2.376 | 0.0754 | 65.8 | 104.6 | 27 |
| 4 | `MAJORITY(CKSP@1d+KELTNER@4h+ADX@4h)` | triple | 60.4 (2.378) | 68.9 (1.749) | 150.2 (2.336) | -4.3 | -15.836 | 2.376 | 0.0754 | 65.8 | 104.6 | 27 |
| 5 | `WEIGHTED_0.5(CKSP@1d+KELTNER@4h+ADX@4h)` | triple | 60.4 (2.378) | 68.9 (1.749) | 150.2 (2.336) | -4.3 | -15.836 | 2.376 | 0.0754 | 65.8 | 104.6 | 27 |
| 6 | `AND_2(CKSP@1d+KELTNER@4h+AO@4h)` | triple | 55.0 (2.165) | 69.5 (1.764) | 176.9 (2.751) | -3.7 | -11.612 | -1.266 | 0.0754 | 58.9 | 104.3 | 27 |
| 7 | `MAJORITY(CKSP@1d+KELTNER@4h+AO@4h)` | triple | 55.0 (2.165) | 69.5 (1.764) | 176.9 (2.751) | -3.7 | -11.612 | -1.266 | 0.0754 | 58.9 | 104.3 | 27 |
| 8 | `WEIGHTED_0.5(CKSP@1d+KELTNER@4h+AO@4h)` | triple | 55.0 (2.165) | 69.5 (1.764) | 176.9 (2.751) | -3.7 | -11.612 | -1.266 | 0.0754 | 58.9 | 104.3 | 27 |
| 9 | `AND_2(KELTNER@4h+AO@4h)` | pair | 50.6 (1.992) | 67.0 (1.701) | 164.9 (2.565) | -3.6 | -15.526 | -18.373 | 0.0754 | 59.9 | 103.6 | 27 |
| 10 | `WEIGHTED_0.67(KELTNER@4h+AO@4h)` | pair | 50.6 (1.992) | 67.0 (1.701) | 164.9 (2.565) | -3.6 | -15.526 | -18.373 | 0.0754 | 59.9 | 103.6 | 27 |
| 11 | `AND_2(KELTNER@4h+AO@4h+ADX@4h)` | triple | 53.0 (2.087) | 67.5 (1.713) | 167.3 (2.602) | -3.7 | -14.301 | -12.611 | 0.0754 | 56.9 | 98.9 | 27 |
| 12 | `MAJORITY(KELTNER@4h+AO@4h+ADX@4h)` | triple | 53.0 (2.087) | 67.5 (1.713) | 167.3 (2.602) | -3.7 | -14.301 | -12.611 | 0.0754 | 56.9 | 98.9 | 27 |
| 13 | `WEIGHTED_0.5(KELTNER@4h+AO@4h+ADX@4h)` | triple | 53.0 (2.087) | 67.5 (1.713) | 167.3 (2.602) | -3.7 | -14.301 | -12.611 | 0.0754 | 56.9 | 98.9 | 27 |
| 14 | `OR(KELTNER:KELT(20,1.5)+KELTNER:KELT(30,2.5))` | a1a2* | 54.3 (2.138) | 68.5 (1.739) | 162.8 (2.532) | -3.6 | -12.209 | -6.09 | 0.0754 | 58.4 | 98.7 | 27 |
| 15 | `OR(KELTNER@4h+BBSQZ@1d)` | pair | 53.0 (2.087) | 69.7 (1.769) | 181.1 (2.816) | -6.4 | -8.819 | -15.411 | 0.0754 | 54.4 | 97.8 | 27 |
| 16 | `OR(KELTNER:KELT(20,2.0)+KELTNER:KELT(30,2.0))` | a1a2* | 54.9 (2.161) | 65.7 (1.668) | 159.1 (2.474) | -3.6 | -16.891 | -2.502 | 0.0754 | 57.0 | 97.5 | 27 |
| 17 | `OR(KELTNER@4h+ADX@4h)` | pair | 70.0 (2.756) | 80.8 (2.051) | 173.4 (2.697) | -6.1 | 13.024 | -7.719 | 0.0754 | 60.8 | 95.6 | 27 |
| 18 | `AND_2(CKSP@1d+KELTNER@4h)` | pair | 54.4 (2.142) | 58.9 (1.495) | 122.5 (1.905) | -2.2 | -30.807 | -7.186 | 0.0754 | 58.7 | 90.4 | 27 |
| 19 | `WEIGHTED_0.67(CKSP@1d+KELTNER@4h)` | pair | 54.4 (2.142) | 58.9 (1.495) | 122.5 (1.905) | -2.2 | -30.807 | -7.186 | 0.0754 | 58.7 | 90.4 | 27 |
| 20 | `GATE[KELTNER@4h|CKSP@1d](CKSP@1d+KELTNER@4h)` | pair | 54.4 (2.142) | 58.9 (1.495) | 122.5 (1.905) | -2.2 | -30.807 | -7.186 | 0.0754 | 58.7 | 90.4 | 27 |
| 21 | `GATE[KELTNER@4h|CKSP@1d](CKSP@1d+KELTNER@4h+ADX@4h)` | triple | 54.4 (2.142) | 58.9 (1.495) | 122.5 (1.905) | -2.2 | -30.807 | -7.186 | 0.0754 | 58.7 | 90.4 | 27 |
| 22 | `GATE[KELTNER@4h|CKSP@1d](CKSP@1d+KELTNER@4h+RSI@1h)` | triple | 54.4 (2.142) | 58.9 (1.495) | 122.5 (1.905) | -2.2 | -30.807 | -7.186 | 0.0754 | 58.7 | 90.4 | 27 |
| 23 | `GATE[KELTNER@4h|CKSP@1d](CKSP@1d+KELTNER@4h+AO@4h)` | triple | 54.4 (2.142) | 58.9 (1.495) | 122.5 (1.905) | -2.2 | -30.807 | -7.186 | 0.0754 | 58.7 | 90.4 | 27 |
| 24 | `GATE[KELTNER@4h|CKSP@1d](CKSP@1d+KELTNER@4h+ULCER@1d)` | triple | 54.4 (2.142) | 58.9 (1.495) | 122.5 (1.905) | -2.2 | -30.807 | -7.186 | 0.0754 | 58.7 | 90.4 | 27 |
| 25 | `GATE[KELTNER@4h|CKSP@1d](CKSP@1d+KELTNER@4h+BBSQZ@1d)` | triple | 54.4 (2.142) | 58.9 (1.495) | 122.5 (1.905) | -2.2 | -30.807 | -7.186 | 0.0754 | 58.7 | 90.4 | 27 |

_* = a1a2 within-indicator config-combo (REDUNDANT -- flagged data-snoop trap; shown for completeness)._

## HONEST VERDICT

**NULL (prior-consistent, honest).** NO combo beats its best constituent single AND long-only buy-hold on OOS+IRON net after deflation with PBO low + p05>0. The de-risked-beta wall holds under
combination: CSCV-PBO=0.0754 (overfit) and White's Reality Check p=0.245 (best combo fails the
multiple-comparisons deflation). Combining de-risked-beta TI engines does NOT manufacture alpha.

## CONDITIONING FRONTIER (cuts OOS maxDD vs buy-hold + robust, WITHOUT beating net)

The project's VALIDATED use of TIs (regime/AND-gating = drawdown insurance, per efb23a4b):

| combo | OOS maxDD (BH=-16.0) | OOS net | IRON net | robust |
|---|--:|--:|--:|:--:|
| `AND_3(KELTNER@4h+RSI@1h+ADX@4h)` | -1.1 | 21.7 | 26.3 | True |
| `WEIGHTED_0.67(KELTNER@4h+RSI@1h+ADX@4h)` | -1.1 | 21.7 | 26.3 | True |
| `AND_3(RSI@1h+ADX@4h+BBSQZ@1d)` | -1.1 | 6.3 | 6.3 | True |
| `WEIGHTED_0.67(RSI@1h+ADX@4h+BBSQZ@1d)` | -1.1 | 6.3 | 6.3 | True |
| `AND_3(CKSP@1d+RSI@1h+ADX@4h)` | -1.4 | 17.9 | 16.3 | True |
| `WEIGHTED_0.67(CKSP@1d+RSI@1h+ADX@4h)` | -1.4 | 17.9 | 16.3 | True |
| `AND_3(AO@4h+ADX@4h+BBSQZ@1d)` | -1.4 | 14.7 | 22.9 | True |
| `WEIGHTED_0.67(AO@4h+ADX@4h+BBSQZ@1d)` | -1.4 | 14.7 | 22.9 | True |
| `AND_3(AO@4h+ULCER@1d+ADX@4h)` | -1.5 | 27.1 | 24.1 | True |
| `WEIGHTED_0.67(AO@4h+ULCER@1d+ADX@4h)` | -1.5 | 27.1 | 24.1 | True |
