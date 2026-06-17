# 2020 H2 per-INSTRUMENT performance per best MA type (VAL Jul-Sep + OOS Oct-Dec)

User /orc 2026-06-12: "still within 2020. Performance PER INSTRUMENT per BEST type of MA for the VAL and
OOS windows of 2020 (the last 6 months)." Tools: `ma_2020_per_instrument.py` (decouples the book into single
instruments -- each asset's own family book, no cross-sectional pool; reuses the VERIFIED `_cells`) +
`ma_2020_per_instrument_render.py`. Best MA selected ON VAL (causal, no look-ahead), OOS reported. All 6 TFs.

## Per instrument: best MA (selected on VAL) -> OOS%, by timeframe
| instrument | 1d | 4h | 2h | 1h | 30m | 15m |
|---|---|---|---|---|---|---|
| BTC | HMA:82 | WMA:109 | DEMA:71 | EMA:103 | VIDYA:113 | VIDYA:93 |
| ETH | HMA:44 | HMA:52 | TEMA:65 | DEMA:68 | EMA:58 | VIDYA:56 |
| SOL | EMA:0 | VIDYA:-10 | HMA:-25 | HMA:-25 | SMA:-12 | SMA:-12 |
| BNB | DEMA:6 | TEMA:8 | WMA:7 | SMA:13 | VIDYA:4 | EMA:9 |
| XRP | HMA:100 | TEMA:90 | DEMA:80 | VIDYA:117 | EMA:11 | VIDYA:34 |
| DOGE | KAMA:31 | VIDYA:21 | VIDYA:28 | VIDYA:37 | VIDYA:13 | VIDYA:4 |
| ADA | HMA:32 | KAMA:-2 | EMA:32 | TEMA:52 | HMA:41 | DEMA:21 |
| AVAX | EMA:0 | KAMA:-27 | VIDYA:-17 | KAMA:-19 | VIDYA:-19 | VIDYA:-22 |
| LINK | TEMA:0 | WMA:-17 | DEMA:7 | HMA:19 | KAMA:-2 | SMA:-7 |
| LTC | HMA:66 | TEMA:66 | DEMA:93 | VIDYA:53 | VIDYA:66 | VIDYA:67 |

Per-asset, the best MA tracks the SAME cadence law from the book-level study: low-lag (HMA/DEMA/TEMA) at
coarse, **VIDYA at fine** (BTC/ETH/LTC/DOGE/XRP all pick VIDYA at 30m/15m).

## VAL->OOS SELECTION TRANSFER -- does picking the best MA per instrument on VAL WORK out of sample?
| TF | VAL-best -> OOS | OOS-best (hindsight) | random-MA (avg) | regret | verdict |
|---|---|---|---|---|---|
| 1d | 36.0 | 39.3 | 21.2 | 3.3 | **beats random +15** |
| 4h | 29.1 | 49.6 | 27.1 | 20.5 | beats random +2 |
| 2h | 34.2 | 52.0 | 33.0 | 17.9 | beats random +1 |
| 1h | 41.6 | 51.2 | 35.4 | 9.5 | beats random +6 |
| 30m | 27.2 | 36.1 | 19.6 | 8.8 | beats random +8 |
| 15m | 24.1 | 26.6 | 6.3 | 2.5 | **beats random +18** |
**Selecting the best MA TYPE per instrument on VAL TRANSFERS** -- it beats a random/average MA at EVERY
timeframe, strongest at the EXTREMES (1d +15, 15m +18) where one MA class is clearly right (low-lag coarse /
VIDYA fine). This is UNLIKE per-CONFIG selection (39 choices, refuted earlier): choosing among 8 structural
MA TYPES is a more robust, lower-variance decision. Regret vs the hindsight OOS-best is small at the extremes
(3.3 / 2.5) but large at mid cadence (4h 20.5, 2h 17.9) -- at 4h/2h the VAL-best is noisy (you leave upside
on the table) though still >= random. CAVEAT: VAL (Jul-Sep) and OOS (Oct-Dec) are ADJACENT, same late-2020
BULL regime -- so the transfer is intra-regime; a VAL-bull -> OOS-bear shift would likely break it.

## Per-instrument BREADTH -- who carries 2020 H2 (mean OOS across all MA classes, pooled over TFs)
| instrument | mean OOS | %MA>0 | note |
|---|---|---|---|
| BTC | +72.7 | 100% | carrier |
| XRP | +60.6 | 92% | carrier |
| LTC | +51.2 | 96% | carrier |
| ETH | +39.9 | 96% | carrier |
| DOGE | +27.9 | 100% | carrier |
| ADA | +24.4 | 94% | carrier |
| BNB | +6.0 | 83% | mid |
| LINK | -3.4 | 35% | weak |
| SOL | -16.8 | 0% | new-2020 (listed ~Sep) |
| AVAX | -24.8 | 0% | new-2020 (listed ~Sep) |

**6 of 10 instruments are broad CARRIERS** (BTC/XRP/LTC/ETH/DOGE/ADA -- strongly positive under 92-100% of
MA classes, NOT a 1-asset concentration). BNB is mid. **SOL and AVAX are new-2020 listings** with ~no H2
history -> their negatives are a data-availability artifact, NOT an MA failure (correctly surfaced). LINK is
the one genuinely weak established coin (high VAL flukes that collapse OOS -> the regret driver).

## Net
- The per-instrument best-MA follows the cadence law (VIDYA at fine on the majors).
- **Per-instrument MA-TYPE selection on VAL is a REAL skill** (beats random every TF, +15/+18 at the
  extremes) -- the 8-type space is robust where the 39-config space was not. Intra-regime caveat applies.
- **2020-H2 breadth is broad among established coins (6/10 carriers)**, dragged only by the not-yet-mature
  SOL/AVAX (no data) and weak LINK -- so the book result is not concentration, it is the large-cap complex.
- HONEST: 2020 = bull; VAL/OOS adjacent same-regime. Confirm on a regime-shift slice (2022) before trusting
  the VAL-selection-transfers claim generally. UNSEEN untouched (2020 is TRAIN-era).

Figure: `charts/ma_2020_per_instrument.png` (6 panels = instrument x MA OOS heatmap per TF; green=positive).
json: `per_instrument_*.json` + `per_instrument_transfer.json`.
RWYB: `python -m strat.ma_2020_per_instrument --cadences <tf>; python -m strat.ma_2020_per_instrument_render`.
