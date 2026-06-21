# V1.2 COMBINATRIX -- LEADERBOARD (structured search: singles / pairs / beam-triples / a1a2)  [combo_cad=2h]

> Ranked by DEFLATED (OOS+IRON)/2 net wealth. Every combo gated: block-bootstrap p05 (vs best-single
> AND vs buy-hold), CSCV-PBO, White's Reality Check + deflated-Sharpe (n_trials = #combos tried).
> 2020 VAL = SELECT / 2020 OOS = REPORT / 2021 IRON = CONFIRM / 2022+ SEALED (asserted never loaded).

- **Combos evaluated:** 285 (pairs=75, triples=202, a1a2=8); n_trials(deflation)=285
- **Breadth:** n_names=27 distinct u100 names (>=20 floor met).
- **CSCV-PBO** over all combos = **0.25** (not overfit).  **White's Reality Check** (best combo vs BH) p=**0.468** (FAILS deflation).
- **Buy-hold benchmark:** DEV val=24.4 OOS=38.0 dd=-15.7 | IRON-2021 H2=63.9.
- **VAL-selected combo:** `OR(CKSP@1d+ADX@4h)`.

## SINGLES recap (the cross-mechanism inputs -- all de-risked betas)

| input | mechanism | OOS net | xBH | IRON net |
|---|---|--:|--:|--:|
| CKSP@1d | trend | 48.1 | 1.266 | 108.2 |
| KELTNER@4h | breakout | 56.2 | 1.479 | 119.6 |
| AO@4h | momentum | 48.8 | 1.284 | 83.3 |
| RSI@1h | mean-reversion | 13.0 | 0.342 | 2.9 |
| ULCER@1d | volatility-sizing | 32.4 | 0.853 | 56.6 |
| ADX@4h | trend2 | 39.5 | 1.039 | 43.2 |
| BBSQZ@1d | volatility2 | 20.8 | 0.547 | 42.7 |

## TOP-25 COMBOS (by deflated OOS+IRON net)

| # | combo | tier | val net (xBH) | OOS net (xBH) | IRON net (xBH) | maxDD | p05 vs BH | p05 vs SGL | PBO | defl net | defl OOS+IRON | n |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `AND_2(CKSP@1d+KELTNER@4h)` | pair | 46.7 (1.914) | 50.3 (1.324) | 84.9 (1.329) | -2.2 | -37.144 | -11.127 | 0.25 | 49.6 | 66.7 | 27 |
| 2 | `WEIGHTED_0.67(CKSP@1d+KELTNER@4h)` | pair | 46.7 (1.914) | 50.3 (1.324) | 84.9 (1.329) | -2.2 | -37.144 | -11.127 | 0.25 | 49.6 | 66.7 | 27 |
| 3 | `GATE[KELTNER@4h|CKSP@1d](CKSP@1d+KELTNER@4h)` | pair | 46.7 (1.914) | 50.3 (1.324) | 84.9 (1.329) | -2.2 | -37.144 | -11.127 | 0.25 | 49.6 | 66.7 | 27 |
| 4 | `GATE[KELTNER@4h|CKSP@1d](CKSP@1d+KELTNER@4h+ADX@4h)` | triple | 46.7 (1.914) | 50.3 (1.324) | 84.9 (1.329) | -2.2 | -37.144 | -11.127 | 0.25 | 49.6 | 66.7 | 27 |
| 5 | `GATE[KELTNER@4h|CKSP@1d](CKSP@1d+KELTNER@4h+RSI@1h)` | triple | 46.7 (1.914) | 50.3 (1.324) | 84.9 (1.329) | -2.2 | -37.144 | -11.127 | 0.25 | 49.6 | 66.7 | 27 |
| 6 | `GATE[KELTNER@4h|CKSP@1d](CKSP@1d+KELTNER@4h+AO@4h)` | triple | 46.7 (1.914) | 50.3 (1.324) | 84.9 (1.329) | -2.2 | -37.144 | -11.127 | 0.25 | 49.6 | 66.7 | 27 |
| 7 | `GATE[KELTNER@4h|CKSP@1d](CKSP@1d+KELTNER@4h+ULCER@1d)` | triple | 46.7 (1.914) | 50.3 (1.324) | 84.9 (1.329) | -2.2 | -37.144 | -11.127 | 0.25 | 49.6 | 66.7 | 27 |
| 8 | `GATE[KELTNER@4h|CKSP@1d](CKSP@1d+KELTNER@4h+BBSQZ@1d)` | triple | 46.7 (1.914) | 50.3 (1.324) | 84.9 (1.329) | -2.2 | -37.144 | -11.127 | 0.25 | 49.6 | 66.7 | 27 |
| 9 | `AND_2(CKSP@1d+KELTNER@4h+ADX@4h)` | triple | 51.5 (2.111) | 54.4 (1.432) | 98.2 (1.537) | -4.3 | -29.121 | -6.979 | 0.25 | 43.8 | 61.5 | 27 |
| 10 | `MAJORITY(CKSP@1d+KELTNER@4h+ADX@4h)` | triple | 51.5 (2.111) | 54.4 (1.432) | 98.2 (1.537) | -4.3 | -29.121 | -6.979 | 0.25 | 43.8 | 61.5 | 27 |
| 11 | `WEIGHTED_0.5(CKSP@1d+KELTNER@4h+ADX@4h)` | triple | 51.5 (2.111) | 54.4 (1.432) | 98.2 (1.537) | -4.3 | -29.121 | -6.979 | 0.25 | 43.8 | 61.5 | 27 |
| 12 | `AND_3(CKSP@1d+KELTNER@4h+AO@4h)` | triple | 43.8 (1.795) | 48.4 (1.274) | 75.9 (1.188) | -2.2 | -39.82 | -13.123 | 0.25 | 47.7 | 61.2 | 27 |
| 13 | `WEIGHTED_0.67(CKSP@1d+KELTNER@4h+AO@4h)` | triple | 43.8 (1.795) | 48.4 (1.274) | 75.9 (1.188) | -2.2 | -39.82 | -13.123 | 0.25 | 47.7 | 61.2 | 27 |
| 14 | `AND_2(CKSP@1d+AO@4h)` | pair | 45.6 (1.869) | 48.6 (1.279) | 76.4 (1.196) | -2.6 | -38.873 | -11.361 | 0.25 | 46.1 | 59.3 | 27 |
| 15 | `WEIGHTED_0.67(CKSP@1d+AO@4h)` | pair | 45.6 (1.869) | 48.6 (1.279) | 76.4 (1.196) | -2.6 | -38.873 | -11.361 | 0.25 | 46.1 | 59.3 | 27 |
| 16 | `GATE[AO@4h|CKSP@1d](CKSP@1d+AO@4h)` | pair | 45.6 (1.869) | 48.6 (1.279) | 76.4 (1.196) | -2.6 | -38.873 | -11.361 | 0.25 | 46.1 | 59.3 | 27 |
| 17 | `GATE[AO@4h|CKSP@1d](CKSP@1d+AO@4h+ADX@4h)` | triple | 45.6 (1.869) | 48.6 (1.279) | 76.4 (1.196) | -2.6 | -38.873 | -11.361 | 0.25 | 46.1 | 59.3 | 27 |
| 18 | `GATE[AO@4h|CKSP@1d](CKSP@1d+AO@4h+RSI@1h)` | triple | 45.6 (1.869) | 48.6 (1.279) | 76.4 (1.196) | -2.6 | -38.873 | -11.361 | 0.25 | 46.1 | 59.3 | 27 |
| 19 | `GATE[AO@4h|CKSP@1d](CKSP@1d+KELTNER@4h+AO@4h)` | triple | 45.6 (1.869) | 48.6 (1.279) | 76.4 (1.196) | -2.6 | -38.873 | -11.361 | 0.25 | 46.1 | 59.3 | 27 |
| 20 | `GATE[AO@4h|CKSP@1d](CKSP@1d+AO@4h+BBSQZ@1d)` | triple | 45.6 (1.869) | 48.6 (1.279) | 76.4 (1.196) | -2.6 | -38.873 | -11.361 | 0.25 | 46.1 | 59.3 | 27 |
| 21 | `AND_2(KELTNER:KELT(20,1.5)+KELTNER:KELT(30,2.5))` | a1a2* | 39.0 (1.598) | 48.3 (1.271) | 98.7 (1.545) | -3.8 | -38.06 | -15.38 | 0.25 | 38.9 | 59.2 | 27 |
| 22 | `AND_2(CKSP@1d+KELTNER@4h+BBSQZ@1d)` | triple | 52.7 (2.16) | 48.5 (1.276) | 87.9 (1.376) | -5.7 | -36.271 | -8.594 | 0.25 | 41.3 | 58.1 | 27 |
| 23 | `MAJORITY(CKSP@1d+KELTNER@4h+BBSQZ@1d)` | triple | 52.7 (2.16) | 48.5 (1.276) | 87.9 (1.376) | -5.7 | -36.271 | -8.594 | 0.25 | 41.3 | 58.1 | 27 |
| 24 | `WEIGHTED_0.5(CKSP@1d+KELTNER@4h+BBSQZ@1d)` | triple | 52.7 (2.16) | 48.5 (1.276) | 87.9 (1.376) | -5.7 | -36.271 | -8.594 | 0.25 | 41.3 | 58.1 | 27 |
| 25 | `AND_2(KELTNER@4h+AO@4h)` | pair | 41.5 (1.701) | 53.1 (1.397) | 103.4 (1.618) | -4.0 | -29.44 | -11.914 | 0.25 | 39.3 | 57.9 | 27 |

_* = a1a2 within-indicator config-combo (REDUNDANT -- flagged data-snoop trap; shown for completeness)._

## HONEST VERDICT

**NULL (prior-consistent, honest).** NO combo beats its best constituent single AND long-only buy-hold on OOS+IRON net after deflation with PBO low + p05>0. The de-risked-beta wall holds under
combination: CSCV-PBO=0.25 (overfit) and White's Reality Check p=0.468 (best combo fails the
multiple-comparisons deflation). Combining de-risked-beta TI engines does NOT manufacture alpha.

## CONDITIONING FRONTIER (cuts OOS maxDD vs buy-hold + robust, WITHOUT beating net)

The project's VALIDATED use of TIs (regime/AND-gating = drawdown insurance, per efb23a4b):

| combo | OOS maxDD (BH=-15.7) | OOS net | IRON net | robust |
|---|--:|--:|--:|:--:|
| `AND_3(RSI@1h+ADX@4h+BBSQZ@1d)` | -1.2 | 4.8 | 5.2 | True |
| `WEIGHTED_0.67(RSI@1h+ADX@4h+BBSQZ@1d)` | -1.2 | 4.8 | 5.2 | True |
| `AND_3(AO@4h+ADX@4h+BBSQZ@1d)` | -1.3 | 13.2 | 13.9 | True |
| `WEIGHTED_0.67(AO@4h+ADX@4h+BBSQZ@1d)` | -1.3 | 13.2 | 13.9 | True |
| `AND_3(CKSP@1d+KELTNER@4h+RSI@1h)` | -1.4 | 19.3 | 21.4 | True |
| `WEIGHTED_0.67(CKSP@1d+KELTNER@4h+RSI@1h)` | -1.4 | 19.3 | 21.4 | True |
| `AND_3(AO@4h+ULCER@1d+ADX@4h)` | -1.4 | 22.8 | 15.6 | True |
| `WEIGHTED_0.67(AO@4h+ULCER@1d+ADX@4h)` | -1.4 | 22.8 | 15.6 | True |
| `AND_3(KELTNER@4h+ADX@4h+BBSQZ@1d)` | -1.4 | 14.3 | 17.3 | True |
| `WEIGHTED_0.67(KELTNER@4h+ADX@4h+BBSQZ@1d)` | -1.4 | 14.3 | 17.3 | True |
