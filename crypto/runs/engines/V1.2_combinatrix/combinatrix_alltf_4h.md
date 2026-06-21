# V1.2 COMBINATRIX -- LEADERBOARD (structured search: singles / pairs / beam-triples / a1a2)  [combo_cad=4h]

> Ranked by DEFLATED (OOS+IRON)/2 net wealth. Every combo gated: block-bootstrap p05 (vs best-single
> AND vs buy-hold), CSCV-PBO, White's Reality Check + deflated-Sharpe (n_trials = #combos tried).
> 2020 VAL = SELECT / 2020 OOS = REPORT / 2021 IRON = CONFIRM / 2022+ SEALED (asserted never loaded).

- **Combos evaluated:** 269 (pairs=75, triples=186, a1a2=8); n_trials(deflation)=269
- **Breadth:** n_names=27 distinct u100 names (>=20 floor met).
- **CSCV-PBO** over all combos = **0.2143** (not overfit).  **White's Reality Check** (best combo vs BH) p=**0.7045** (FAILS deflation).
- **Buy-hold benchmark:** DEV val=25.1 OOS=35.2 dd=-15.1 | IRON-2021 H2=63.3.
- **VAL-selected combo:** `OR(CKSP@1d+ADX@4h)`.

## SINGLES recap (the cross-mechanism inputs -- all de-risked betas)

| input | mechanism | OOS net | xBH | IRON net |
|---|---|--:|--:|--:|
| CKSP@1d | trend | 43.3 | 1.23 | 98.7 |
| KELTNER@4h | breakout | 16.3 | 0.463 | 25.9 |
| AO@4h | momentum | 18.1 | 0.514 | 15.7 |
| RSI@1h | mean-reversion | 3.3 | 0.094 | 8.4 |
| ULCER@1d | volatility-sizing | 28.8 | 0.818 | 53.2 |
| ADX@4h | trend2 | 17.4 | 0.494 | 5.7 |
| BBSQZ@1d | volatility2 | 18.8 | 0.534 | 38.5 |

## TOP-25 COMBOS (by deflated OOS+IRON net)

| # | combo | tier | val net (xBH) | OOS net (xBH) | IRON net (xBH) | maxDD | p05 vs BH | p05 vs SGL | PBO | defl net | defl OOS+IRON | n |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `AND_2(CKSP:CKSP(10,2.0)+CKSP:CKSP(20,2.0))` | a1a2* | 50.5 (2.012) | 42.0 (1.193) | 91.0 (1.438) | -7.9 | -36.757 | -9.733 | 0.2143 | 16.4 | 26.0 | 27 |
| 2 | `AND_2(CKSP:CKSP(10,1.0)+CKSP:CKSP(20,3.0))` | a1a2* | 43.4 (1.729) | 42.9 (1.219) | 89.7 (1.417) | -8.9 | -36.467 | -1.194 | 0.2143 | 15.7 | 24.2 | 27 |
| 3 | `OR(CKSP@1d+BBSQZ@1d)` | pair | 48.1 (1.916) | 42.3 (1.202) | 86.3 (1.363) | -9.5 | -34.731 | -2.287 | 0.2143 | 13.4 | 20.4 | 27 |
| 4 | `AND_2(CKSP@1d+ULCER@1d+ADX@4h)` | triple | 40.5 (1.614) | 37.6 (1.068) | 69.7 (1.101) | -9.4 | -44.615 | -14.392 | 0.2143 | 12.0 | 17.1 | 27 |
| 5 | `MAJORITY(CKSP@1d+ULCER@1d+ADX@4h)` | triple | 40.5 (1.614) | 37.6 (1.068) | 69.7 (1.101) | -9.4 | -44.615 | -14.392 | 0.2143 | 12.0 | 17.1 | 27 |
| 6 | `WEIGHTED_0.5(CKSP@1d+ULCER@1d+ADX@4h)` | triple | 40.5 (1.614) | 37.6 (1.068) | 69.7 (1.101) | -9.4 | -44.615 | -14.392 | 0.2143 | 12.0 | 17.1 | 27 |
| 7 | `AND_2(CKSP@1d+ULCER@1d)` | pair | 29.7 (1.183) | 33.7 (0.957) | 59.8 (0.945) | -8.5 | -56.297 | -27.153 | 0.2143 | 11.7 | 16.2 | 27 |
| 8 | `WEIGHTED_0.67(CKSP@1d+ULCER@1d)` | pair | 29.7 (1.183) | 33.7 (0.957) | 59.8 (0.945) | -8.5 | -56.297 | -27.153 | 0.2143 | 11.7 | 16.2 | 27 |
| 9 | `GATE[ULCER@1d|CKSP@1d](CKSP@1d+ULCER@1d)` | pair | 29.7 (1.183) | 33.7 (0.957) | 59.8 (0.945) | -8.5 | -56.297 | -27.153 | 0.2143 | 11.7 | 16.2 | 27 |
| 10 | `GATE[ULCER@1d|CKSP@1d](CKSP@1d+ULCER@1d+ADX@4h)` | triple | 29.7 (1.183) | 33.7 (0.957) | 59.8 (0.945) | -8.5 | -56.297 | -27.153 | 0.2143 | 11.7 | 16.2 | 27 |
| 11 | `GATE[ULCER@1d|CKSP@1d](CKSP@1d+ULCER@1d+BBSQZ@1d)` | triple | 29.7 (1.183) | 33.7 (0.957) | 59.8 (0.945) | -8.5 | -56.297 | -27.153 | 0.2143 | 11.7 | 16.2 | 27 |
| 12 | `GATE[ULCER@1d|CKSP@1d](CKSP@1d+RSI@1h+ULCER@1d)` | triple | 29.7 (1.183) | 33.7 (0.957) | 59.8 (0.945) | -8.5 | -56.297 | -27.153 | 0.2143 | 11.7 | 16.2 | 27 |
| 13 | `GATE[ULCER@1d|CKSP@1d](CKSP@1d+KELTNER@4h+ULCER@1d)` | triple | 29.7 (1.183) | 33.7 (0.957) | 59.8 (0.945) | -8.5 | -56.297 | -27.153 | 0.2143 | 11.7 | 16.2 | 27 |
| 14 | `GATE[ULCER@1d|CKSP@1d](CKSP@1d+AO@4h+ULCER@1d)` | triple | 29.7 (1.183) | 33.7 (0.957) | 59.8 (0.945) | -8.5 | -56.297 | -27.153 | 0.2143 | 11.7 | 16.2 | 27 |
| 15 | `AND_2(CKSP@1d+AO@4h+BBSQZ@1d)` | triple | 42.8 (1.705) | 34.3 (0.974) | 54.6 (0.863) | -8.2 | -47.641 | -18.957 | 0.2143 | 12.3 | 15.9 | 27 |
| 16 | `MAJORITY(CKSP@1d+AO@4h+BBSQZ@1d)` | triple | 42.8 (1.705) | 34.3 (0.974) | 54.6 (0.863) | -8.2 | -47.641 | -18.957 | 0.2143 | 12.3 | 15.9 | 27 |
| 17 | `WEIGHTED_0.5(CKSP@1d+AO@4h+BBSQZ@1d)` | triple | 42.8 (1.705) | 34.3 (0.974) | 54.6 (0.863) | -8.2 | -47.641 | -18.957 | 0.2143 | 12.3 | 15.9 | 27 |
| 18 | `AND_2(CKSP@1d+ADX@4h+BBSQZ@1d)` | triple | 35.3 (1.406) | 32.0 (0.909) | 52.5 (0.829) | -8.2 | -51.64 | -26.223 | 0.2143 | 11.9 | 15.7 | 27 |
| 19 | `MAJORITY(CKSP@1d+ADX@4h+BBSQZ@1d)` | triple | 35.3 (1.406) | 32.0 (0.909) | 52.5 (0.829) | -8.2 | -51.64 | -26.223 | 0.2143 | 11.9 | 15.7 | 27 |
| 20 | `WEIGHTED_0.5(CKSP@1d+ADX@4h+BBSQZ@1d)` | triple | 35.3 (1.406) | 32.0 (0.909) | 52.5 (0.829) | -8.2 | -51.64 | -26.223 | 0.2143 | 11.9 | 15.7 | 27 |
| 21 | `AND_2(CKSP@1d+AO@4h+ULCER@1d)` | triple | 45.3 (1.805) | 34.8 (0.989) | 69.3 (1.095) | -9.0 | -46.516 | -16.319 | 0.2143 | 10.1 | 15.1 | 27 |
| 22 | `MAJORITY(CKSP@1d+AO@4h+ULCER@1d)` | triple | 45.3 (1.805) | 34.8 (0.989) | 69.3 (1.095) | -9.0 | -46.516 | -16.319 | 0.2143 | 10.1 | 15.1 | 27 |
| 23 | `WEIGHTED_0.5(CKSP@1d+AO@4h+ULCER@1d)` | triple | 45.3 (1.805) | 34.8 (0.989) | 69.3 (1.095) | -9.0 | -46.516 | -16.319 | 0.2143 | 10.1 | 15.1 | 27 |
| 24 | `AND_2(CKSP@1d+KELTNER@4h+ULCER@1d)` | triple | 42.1 (1.677) | 34.5 (0.98) | 68.9 (1.088) | -9.0 | -46.824 | -15.509 | 0.2143 | 9.9 | 14.8 | 27 |
| 25 | `MAJORITY(CKSP@1d+KELTNER@4h+ULCER@1d)` | triple | 42.1 (1.677) | 34.5 (0.98) | 68.9 (1.088) | -9.0 | -46.824 | -15.509 | 0.2143 | 9.9 | 14.8 | 27 |

_* = a1a2 within-indicator config-combo (REDUNDANT -- flagged data-snoop trap; shown for completeness)._

## HONEST VERDICT

**NULL (prior-consistent, honest).** NO combo beats its best constituent single AND long-only buy-hold on OOS+IRON net after deflation with PBO low + p05>0. The de-risked-beta wall holds under
combination: CSCV-PBO=0.2143 (overfit) and White's Reality Check p=0.7045 (best combo fails the
multiple-comparisons deflation). Combining de-risked-beta TI engines does NOT manufacture alpha.

## CONDITIONING FRONTIER (cuts OOS maxDD vs buy-hold + robust, WITHOUT beating net)

The project's VALIDATED use of TIs (regime/AND-gating = drawdown insurance, per efb23a4b):

| combo | OOS maxDD (BH=-15.1) | OOS net | IRON net | robust |
|---|--:|--:|--:|:--:|
| `AND_3(CKSP@1d+AO@4h+RSI@1h)` | -1.6 | 9.2 | 10.3 | True |
| `WEIGHTED_0.67(CKSP@1d+AO@4h+RSI@1h)` | -1.6 | 9.2 | 10.3 | True |
| `AND_3(RSI@1h+ADX@4h+BBSQZ@1d)` | -1.8 | 0.5 | 4.2 | True |
| `WEIGHTED_0.67(RSI@1h+ADX@4h+BBSQZ@1d)` | -1.8 | 0.5 | 4.2 | True |
| `AND_3(CKSP@1d+RSI@1h+ADX@4h)` | -2.1 | 3.6 | 10.4 | True |
| `WEIGHTED_0.67(CKSP@1d+RSI@1h+ADX@4h)` | -2.1 | 3.6 | 10.4 | True |
| `AND_3(CKSP@1d+KELTNER@4h+RSI@1h)` | -2.2 | 7.4 | 11.6 | True |
| `WEIGHTED_0.67(CKSP@1d+KELTNER@4h+RSI@1h)` | -2.2 | 7.4 | 11.6 | True |
| `AND_3(RSI@1h+ULCER@1d+ADX@4h)` | -2.3 | 2.2 | 6.4 | True |
| `WEIGHTED_0.67(RSI@1h+ULCER@1d+ADX@4h)` | -2.3 | 2.2 | 6.4 | True |
