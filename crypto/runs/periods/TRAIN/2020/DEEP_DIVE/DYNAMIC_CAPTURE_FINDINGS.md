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

## THE DECISIVE NULL -- CORRECTED 2026-06-19 (adversarial verification caught 2 real bugs)
**RETRACTION.** An earlier pass claimed the SMA200-gated buy-hold (+937.5%) DOMINATED buy-hold on wealth. That
+937.5% was an **SMA-200 INITIALISATION ARTIFACT**: `_sma` used `min_periods=1`, so the 3 late-listing assets
(SOL=17, DOGE=43, AVAX=9 pre-window daily bars) got a partial-window average that hugged their listing pump instead
of a true 200-bar trend filter. 100% of the +937->+324 gap localises to those 3 names (the 7 established assets are
bit-identical). FIXED: the gate now uses a STRICT SMA200 (`min_periods=N` -> young asset = cash). A second, orthogonal
leak (`ma_strat_builder.py:404` selected the static band on `oo>0` = OOS-in-band) inflated reported `net_oos` by
+0.6..+6pp; also FIXED (band is now TRAIN+VAL-only). Both verified bit-exactly; gate number reproduced independently.

CORRECTED continuous full-cycle 2020-10-01..2022-12-31 (EW-u10, taker 0.0024, long-only spot, NO leverage):

| strategy | full-cycle net% | maxDD% |
|---|---:|---:|
| **raw buy-hold** | **+548.6** | -79.4 |
| GATE-only STRICT SMA200 | +324.1 | -54.1 |
| engine cells (ADX/MACD/DONCHIAN/SUPERTREND 1d, strict) | +115..+179 | -12..-25 |

- **On WEALTH, raw buy-hold WINS.** No internal long-only signal beats it: BH +549% > strict gate +324% > engine
  +115..+179%. The gate gives up wealth to cut DD (-79->-54%); the engine gives up MORE wealth for MORE DD reduction.
- The TI move-capture signal adds **nothing on wealth** over a plain trend gate -- it over-de-risks (binary cash-out
  during continuing bull moves). The "gate dominates buy-hold" headline is RETRACTED.
- **The engine's only genuine, harvestable property = minimum DRAWDOWN** (-12..-25% vs gate -54% vs BH -79%),
  realisable only under a hard maxDD / Sharpe / leverage-budget objective, never under unconstrained wealth.
- Held-out evidence to trust = the NEGATIVE `p05_oos_bootstrap` everywhere + the forward 2021/2022 translation SIGN
  (not the in-sample-inflated `net_oos`). Fixed-EW u10 inflates ALL wealth figures via the 2021 small-cap melt-up;
  cap-weighting ~halves every number (state weighting explicitly).

## RESCUE-B -- conviction-scaled exposure (does the TI have within-bull SELECTION skill?)
The verdict above used the TI as a BINARY cash-out. RESCUE-B instead uses it as a CONVICTION TILT on the strict
gate: per-asset weight = 0 (gate-off) / 0.5 (gate-on, TI-off) / 1.0 (gate-on, TI-on); MACD-iron(12,26,9) as the TI.
Continuous 2020-10..2022-12, EW-u10, taker, strict gate, positions lagged (causal):

| book | net% | maxDD% |
|---|---:|---:|
| strict gate-only | +330.8 | -54.1 |
| conviction-RAW (0/.5/1, reduced gross) | +364.8 | -35.4 |
| conviction-NORM (matched gross = pure SELECTION) | +370.5 | -54.8 |
| raw buy-hold (ref) | +548.6 | -79.4 |

FINDING: the TI move-capture signal DOES have a real (if modest) WITHIN-BULL SELECTION skill -- at MATCHED gross
the conviction tilt beats gate-only by +40pp (+370 vs +331); and conviction-RAW PARETO-IMPROVES gate-only (MORE
wealth +365 AND less drawdown -35 vs -54). So the TI is NOT "nothing": as a conviction/selection overlay (not a
binary timer) it makes a strictly BETTER de-risked-beta book. CAVEAT: one un-tuned MACD config on the held-out
cycle (suggestive, not dev-selected + forward-confirmed). STILL < raw buy-hold on wealth (+549% > +370%).

VALIDATION (matched-gross conviction vs a same-gross SHUFFLE NULL = random re-assignment of the conviction
weights among gate-on assets; 4 TIs x 3 periods, 1d): the selection edge is REAL + forward-robust for the
strongest trend TIs and survives the null OUT-OF-SAMPLE:
- MACD: conv beats shuffle by 2-3 sigma on held-out 2020-OOS (+57.8 vs 46.8+-3.3) AND forward 2021 (+293.6 vs
  259.7+-15.3); edge +8.6 / +18.4.
- DONCHIAN: same (edge +12.2 / +4.9, beats null dev + forward).
- SUPERTREND: noisy, within the null band (no reliable edge). KELTNER: positive on dev but FLIPS NEGATIVE on
  2021 forward (overfit, doesn't translate). 2022 bear: ~zero edge for all (too few gate-on assets to select).
CONCLUSION: a genuine within-bull asset-SELECTION skill exists for MACD/DONCHIAN (survives shuffle null OOS),
but it is TI-dependent, modest (~6% relative), bull-only, and does NOT approach buy-hold on wealth. It is the one
genuinely-new positive the campaign found beyond the known daily_engine gate -- a conviction-weighting refinement
of the de-risked-beta book, not wealth alpha.

## HONEST VERDICT (campaign, corrected + adversarially verified)
The dynamic capture engine WORKS as built (forward-validated, regime-gated, repeatable, ~24 all-weather trend cells).
But on the binding WEALTH metric, **no internal long-only signal beats raw buy-hold** (BH +549% > strict gate +324% >
engine +115..+179% > conviction-tilt +370%). As a BINARY cash-out the move-capture signal over-de-risks and adds
nothing on wealth over a plain gate; but as a CONVICTION TILT (rescue-B) it has a real within-bull SELECTION skill
that PARETO-improves the gate (+365% / -35% DD vs +331% / -54%) -- a strictly better de-risked-beta book, still < BH
on wealth. The engine's genuine harvestable value is therefore a **minimum-drawdown de-risked-beta allocator with a
modest TI selection edge**, valuable only under a hard maxDD-constrained mandate. This RE-EARNS -- per-TI, forward-validated, cell-by-cell, with TWO real bugs caught
and fixed by adversarial verification -- the standing "no internal long-only wealth alpha; harvestable value =
drawdown-preservation" conclusion. Trend-following TIs (MA + ADX/MACD/DONCHIAN/SUPERTREND/...) all collapse to the same
de-risked-beta point; mean-reversion is dead; fine TFs are dead. **Move CAPTURE (discrimination) is real; harvestable
WEALTH alpha over buy-hold is not.** If the user's objective is drawdown-constrained return, the engine is a real
minimum-DD book; if it is compound wealth, buy-hold wins and the open frontier remains EXTERNAL data.

## Next
- Phase 1: non-MA TI expansion (22 TIs via plug-in adapter) -- prior work says non-MA TIs preserve bears
  better; gated, do they add stronger all-weather cells?
- Phase 2 Tier-2: per-regime band-SUBSET (chop -> slower configs) as a refinement (low priority; dynamic
  switching is fragile -- 1/6 TFs had timing skill).
- Bull-participation ratio vs BH per cell (characterise the de-risked-beta level).

## FIX PASS (2026-06-19) -- solve the fixable weaknesses without overfitting
ANTI-OVERFIT GATE FIRST: asset-DNA traits do NOT persist TRAIN->2021 (trend-propensity/ER rho +0.25 then -0.54
INVERTS; vol only weakly/recently rho +0.55; autocorr unstable). => a STATIC asset-DNA label is overfit by
construction (why "per-asset was refuted"). The defensible per-asset lever is DYNAMIC asset-STATE (trailing vol,
current regime), not a frozen tag. Fixes tested as dynamic-state overlays (forward-validated, EW-u10, taker, net%/maxDD%):

| book | 2020-OOS | 2021 | 2022 |
|---|---|---|---|
| strict gate-only (base) | +46.8 / -13 | +277 / -54 | -23.5 / -25 |
| + conviction tilt | +36.7 / -12 | +309 / -35 | -17.8 / -19 |
| + conviction + vol-target | +35.3 / -10 | +177 / -20 | -14.7 / -16 |
| + conv + vol-target + chop | +19.4 / -10 | +154 / -11 | -10.5 / -11 |

- CONVICTION TILT = real Pareto win (2021 +309 vs +277 AND DD -35 vs -54; 2022 -18 vs -24). The keeper (validated
  shuffle-null + forward). VOL-TARGET + CHOP = clean de-risking frontier (cycle DD -54 -> -11) at a wealth cost;
  knobs for a DD-constrained mandate, causal/not overfit.
- FUNDAMENTAL (verticals): no free lunch -- vol-target makes vertical-capture WORSE (down-weights the high-vol
  moonshots). Cannot capture 30-100x runs without carrying their vol risk. Structural, not internally fixable.
- ASSET-DNA: static label overfit; brought back ONLY as dynamic asset-state (= the conviction/vol-target/chop machinery).
