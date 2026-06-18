# Dynamic Capture Engine -- findings (2026-06-18/19, autonomous campaign)

Charter: `project-dynamic-capture-engine-charter-2026-06-18`. Engine: `src/strat/dynamic_capture_engine.py`
(extends `src/strat/ma_strat_builder.py`). Dev 2020 (6/3/3), iron/forward 2021, 2022 = no-touch bear.
Rank by WEALTH; TAKER floor (0.0024 RT); fixed-EW u10; selection on TRAIN+VAL (OOS held out). RWYB.

## Phase 0 -- STATIC floor (8 MA x 6 TF, ungated, taker)
- 48 cells: **45 B_preserve / 3 C_bull_only / 0 A_allweather**.
- Every TI x TF cell is de-risked beta that **bleeds the 2022 bear** (bear22 -10..-30 across the grid).
- Best robust (p05>0): `1h-WMA +21.8%` (p05 +1.9, bear -6.0), `1h-EMA +18.1%` (p05 +1.8). 1h is the robustness
  sweet-spot, consistent with the prior sub-daily finding.
- Artifact: `dynamic_capture_phase0_static_*.json`.

## Phase 2 -- Tier-1 SMA-200 position GATE (cash when close<=SMA200, per-asset, causal)
The proven lever (D33: regime GATING works, config-SWITCHING hurts). Result vs floor:
- **The universal bear-bleed is fixed**: bear22 across the top cells goes -10..-30 -> ~0
  (1d-TEMA +0.5, 2h-HMA -0.4, 1d-HMA +1.8, 2h-VIDYA 0.0, 1d-WMA -2.8, 1d-DEMA -0.7, 1d-KAMA -1.8).
- Cost: OOS participation dips (1h-WMA 21.8->12.4) and 6 fine-TF cells fall to D_weak (over-gating kills
  fine-TF bull capture + cost). p05 stays <0 (a 2020-bull-window bootstrap the gate cannot touch).
- Artifact: `dynamic_capture_phase2_gate_*.json`.

## IRON / forward 2021 (replay frozen 2020-selected gated specs on unseen 2021 + re-confirm 2022)
The engine TRANSLATES forward:
- **43/48 cells positive on 2021** (the unseen forward year).
- **15 cells ALL-WEATHER** = positive 2020-OOS AND positive 2021-fwd AND bear-preserve 2022 (>= -5%):

  | TF | MA | 2020-OOS | 2021-fwd | 2022-bear |
  |----|----|---------:|---------:|----------:|
  | 1d | HMA  | +17.2 | +290.9 | +1.8 |
  | 1d | TEMA | +20.6 | +220.0 | +0.5 |
  | 2h | HMA  | +18.1 | +190.9 | -0.4 |
  | 1d | DEMA | +16.4 | +163.9 | -0.7 |
  | 1d | WMA  | +20.9 |  +98.1 | -2.8 |
  | 1d | KAMA | +12.2 |  +73.5 | -1.8 |
  | 1h | DEMA | +10.9 |  +61.0 | -3.0 |
  | 1h | WMA  | +12.4 |  +46.9 | -4.7 |
  | 1d | SMA  | +18.0 |  +39.2 | -2.2 |
  | 1d | VIDYA| +6.5  |  +31.0 | -3.3 |
  | 2h | VIDYA| +14.8 |  +26.0 |  0.0 |
  | 4h | VIDYA| +5.7  |  +22.1 | -0.6 |
  | 30m| EMA  | +5.8  |  +16.0 | -3.6 |
  | 1h | VIDYA| +3.2  |   +6.5 | -4.5 |
  | 30m| VIDYA| +2.5  |   +1.9 | -0.0 |

## Honest verdict (so far)
- The gated engine is a **forward-validated de-risked-beta book with clean bear-preservation**, reproduced
  PER-TI and PER-TF. Best lane: low-lag MAs (HMA/TEMA/DEMA) at 1d/2h capture the 2021 bull strongly
  (+160..+290%) and preserve 2022 (~0); adaptive VIDYA = the cleanest bear-preserver across TFs.
- It is **NOT bull-beating alpha**: 2021 +291% is ~0.2x buy-hold (2021 BH ~+1400%). Participate-partially +
  preserve -- the daily_engine deliverable, now per-TI and forward-validated. Consistent with all prior work.
- **p05 (2020-bull-window bootstrap) negative everywhere** -- robustness flag, not an all-weather verdict;
  the forward 2021+2022 translation is the stronger, more relevant test and it passes for 15 cells.
- **Fine TFs (15m) fail** -- over-gated, cost-bound (negative 2021, deep 2022 bleed). Confirmed dead lane.

## Phase 1 -- non-MA TI expansion (gated, taker), coarse 1d/4h/2h
- 9 A_allweather / 27 B_preserve / 6 D_weak. ALL 9 all-weather cells at 1d, TREND-following:
  ADX (1d +28.1/+142.2/-3.2, new TOP, beats best MA), SUPERTREND, DONCHIAN, MACD, VORTEX, TSI, KELTNER, PSAR, ROC.
- Mean-reversion (RSI/STOCH/CCI/BBPCT/WILLR) does NOT translate to 2021 -> dead lane (consistent w/ prior "MR flat").
- Fine TFs (1h/30m/15m): 0 all-weather (over-gated + cost-bound). Volume TIs deferred. Artifacts ti_capture_phase1_*.

## THE KILLER NULL -- gate-only buy-hold DOMINATES the engine on WEALTH (the decisive finding)
Continuous full-cycle replay 2020-10-01 .. 2022-12-31 (held-out: OOS bull tail -> 2021 mixed -> 2022 bear), EW-u10, taker:

| strategy | full-cycle net% | maxDD% | return/maxDD |
|---|---:|---:|---:|
| GATE-ONLY (SMA200-timed buy-hold) | +937.5 | -54.1 | 17.3 |
| raw buy-hold | +548.6 | -79.4 | 6.9 |
| engine ADX/base 1d | +200.4 | -18.8 | 10.7 |
| engine SUPERTREND/base 1d | +194.4 | -17.4 | 11.2 |
| engine MACD/iron 1d | +186.2 | -15.3 | 12.2 |
| engine DONCHIAN/iron 1d | +184.0 | -13.3 | 13.8 |

- **On WEALTH the engine LOSES to a trivial SMA200-timed buy-hold by ~4.7x** (+937 vs +190). The move-capture TI
  signal OVER-de-risks: it gives up far more bull participation than its extra bear-preservation is worth.
- **Only the GATE beats raw buy-hold** (+937 vs +549) -- the simple market-timing overlay (= the daily_engine thesis).
  The TI move-capture signal adds NOTHING on wealth (it underperforms even raw BH).
- On risk-adjusted (return/maxDD) gate-only STILL wins (17.3 vs ~11-14).
- **The engine's ONLY genuine edge = lowest drawdown** (-13..-19% vs gate -54% vs BH -79%). It wins ONLY under a
  HARD maxDD cap (<~20%) that disqualifies gate-only -- i.e. for a risk-bounded mandate that cannot tolerate >20% DD,
  the DONCHIAN/MACD cells (+185%, -13..-15% DD) are the best admissible option.

## HONEST VERDICT (campaign)
The dynamic capture engine WORKS as built (forward-validated, ~24 all-weather MA+trend-TI cells, regime-gated,
repeatable). But on the binding WEALTH metric it is **de-risked beta dominated by a trivial SMA200-timed buy-hold**;
its move-capture signal over-de-risks. The ONLY thing that beats buy-hold on wealth is the GATE alone (the daily_engine
overlay, already owned). The engine's unique value is **minimum-drawdown insurance**, realisable only under a tight
maxDD cap. This re-earns -- per-TI, forward-validated, cell-by-cell with the gate baseline as the arbiter -- the prior
"no internal long-only signal beats buy-hold on wealth; the harvestable value is drawdown-preservation" conclusion.
Trend-following TIs (MA + ADX/MACD/DONCHIAN/SUPERTREND/...) all collapse to the same de-risked-beta point; MR is dead;
fine TFs are dead. Move CAPTURE (discrimination) is real; harvestable WEALTH alpha over buy-hold is not.

## Next
- Phase 1: non-MA TI expansion (22 TIs via plug-in adapter) -- prior work says non-MA TIs preserve bears
  better; gated, do they add stronger all-weather cells?
- Phase 2 Tier-2: per-regime band-SUBSET (chop -> slower configs) as a refinement (low priority; dynamic
  switching is fragile -- 1/6 TFs had timing skill).
- Bull-participation ratio vs BH per cell (characterise the de-risked-beta level).
