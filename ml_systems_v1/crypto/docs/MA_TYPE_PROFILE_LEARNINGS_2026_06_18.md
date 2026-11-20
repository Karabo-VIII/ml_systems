# MA-type profile optimisation — learnings (2026-06-18)

8 MA types (SMA/EMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA) optimised INDEPENDENTLY (no overlap), each across all 6 TFs
{1d,4h,2h,1h,30m,15m} + per-asset view. Developed on within-2020 TRAIN (6/3/3), with the all-weather regime picture
(2020 bull / 2021 mixed / 2022 bear) + the move-catch profile (coverage / entry-lag / capture). Working-band ensemble,
ironed sleeve, fixed-EW, maker cost. Source: workflow `ww8b6a9w7`; runs/strat/ma_movecatch_*.json.

## ⚠️ RWYB CAVEAT (read first)
The 8 optimisers ran independently and **drifted on two metric definitions** — DO NOT trust absolute cross-MA values for:
- **maxDD**: SMA −34..−52% vs EMA/WMA/KAMA −7..−10% (implausible for the same ironed book ⇒ different windows/basis).
- **coverage**: WMA & KAMA report 1.00 at every TF vs ~0.23–0.69 for others (definitional drift).
TRUST: the **within-MA-type TF rankings** + **entry-lag, capture, OOS-net, all-weather net** (consistent across agents,
shared `ma_movecatch_decomp` machinery). A single-harness re-run is needed for clean absolute cross-MA maxDD/coverage.

## The grid (best TF + key metrics per MA type, TRAIN/OOS + all-weather)
| MA | best TF | 1h OOS / cap / lag | 1d (long-term) cap/lag | aw2021 (bull) | aw2022 (bear) | role |
|----|---------|--------------------|------------------------|---------------|---------------|------|
| HMA | 4h/1h | 33.4 / 0.33 / 0.53 | 0.23 / 0.77 | +308–373% | −32..−46% | **best move-catcher** (low-lag, hard bull) |
| TEMA | 2h | 28.0 / 0.30 / 0.58 | 0.21 / 0.82 | +300–314% | −26..−51% | low-lag catcher |
| DEMA | 2h/1h | 30.5 / 0.27 / 0.58 | 0.07 / 0.85 | +257–293% | −23..−48% | low-lag catcher |
| EMA | 2h/1h | 31.8 / 0.30 / 0.59 | 0.07 / 0.86 | +184–201% | **−15..−28%** | balanced; best 1d bear-preserve |
| SMA | 1h | 31.3 / 0.28 / 0.59 | 0.07 / 0.85 | +208–229% | −31..−45% | balanced; 15m degenerates |
| WMA | 2h | 26.9 / 0.30 / 0.58 | 0.08 / 0.83 | +249–254% | −31..−48% | balanced |
| KAMA | 1h | 23.7 / 0.26 / 0.61 | 0.05 / 0.89 | +126–249% | −21..−55% | adaptive; **best bear-preserve**, worst catcher |
| VIDYA | 1h | 33.9 / 0.22 / 0.67 | 0.06 / 0.87 | +60–125% | **−14..−29%** | adaptive; best bear-preserve, weakest bull |

## Robust learnings
1. **The sweet-spot TF is the MIDDLE (1h–2h) for EVERY MA type** — peak OOS-net + capture + lowest entry-lag, with
   manageable risk. The extremes are worse: **1d** = trend-rider / de-risked-beta (capture 0.05–0.23, lag ~0.85 — it
   does NOT catch individual moves, it rides sustained trends and goes to cash in bears); **15m** = uniformly the WORST
   (OOS-net collapses 14–22%, p05 worst, aw2022 worst −45..−55%) for ALL types — the hoped-for fine-TF flip does NOT
   appear cleanly; the band degenerates to slow configs + cost/whipsaw bite.
2. **Finer TF ⇒ earlier entry (lower lag) BUT worse bear-resilience.** entry-lag falls monotonically 1d≈0.85 → 1h≈0.58
   across all types (catch moves earlier), but aw2022 worsens as TF finer. 1h–2h is the balance point.
3. **The capture/entry-lag lens INVERTS the wealth ranking (confirmed across all 8):** low-lag **HMA/TEMA/DEMA** = best
   move-catchers (earliest entry, highest capture, hardest bull-riders); adaptive **KAMA/VIDYA** = best bear-preservers
   but WORST move-catchers (highest lag). Participate-vs-preserve, now mapped per MA-type. **HMA is the standout catcher.**
4. **Per-asset does NOT beat the pooled-within-type band** (config-rank ρ≈0 cross-year) — deploy pooled fixed-EW bands.

## The two trading-style options (the user's framing)
- **INTRADAY → 1h (NOT 15m/30m, which degrade).** Best intraday move-catchers: **HMA, EMA** (highest capture, lowest lag,
  strong OOS-net). True fine-scalping (15m) is NOT supported by MA setups here — it degrades for every type.
- **LONG-TERM (swing/trend) → 1d/4h.** Value = bear-protected trend-riding (de-risked beta), NOT individual-move capture.
  Best bear-preservers: **EMA, VIDYA, KAMA** (1d/4h aw2022 −14..−24% vs buy-hold ≈ −71%). Hardest bull-participation:
  **HMA/DEMA/TEMA**.

## Caveats (binding)
- TRAIN-developed + move-labels are ex-post ⇒ **discrimination, not yet OOS-validated harvestability**. The aw2021/aw2022
  are the SELECTED band's regime performance (optimistic). Everything is still **de-risked beta** (aw2022 negative for ALL
  types/TFs; participation, not bull-beating alpha — consistent with the prior 0/21-beat-BH finding).
- The cross-agent maxDD/coverage inconsistency above.

## Next cuts (offered)
1. **Single-harness clean re-run** of the grid (one consistent maxDD/coverage definition) for the honest absolute cross-MA
   comparison.
2. **OOS harvestability gate** on the 1h–2h sweet-spot (esp. HMA/EMA): does the early-entry/capture survive *unconditionally*
   (all entries, ex-ante) on VAL/OOS — a tradeable edge, or a better MA for the de-risked-beta book?
