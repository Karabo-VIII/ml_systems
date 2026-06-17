# PHASE 2 -- the DYNAMIC / ML ALLOCATION ENGINE (2020 runway, all 6 TFs)

The HEADLINE deliverable: the user's "dynamic" + "ML engine" ask, built as a **regime-conditional allocator
that dynamically weights the two COMPLEMENTARY families (trend MA + MR oscillator) per timeframe** so "when
one is out, the other is on." Construction + honest, two-sided, held-out validation.

- Tool: `src/strat/dynamic_allocation_engine.py` (RWYB: `python -m strat.dynamic_allocation_engine`).
- Window: **2020 BAND ONLY** (runway fenced to 2020-01-07..2021-01-01; data past that never scored). No
  2026/other data touched. Synthetic used only for the selftest nulls.
- Universe u10, maker cost, causal/lag-1, the two long-only sleeves from `deep2020_complementarity` (NOT
  reinvented): trend MA-family (KAMA at 1d per `ma_type_tf_research` winner; EMA elsewhere per the
  "missing-TF -> EMA" fallback) + MR oscillator family.
- Runway: **TRAIN 2020-01..07 (fit) / VAL 2020-07..10 (select the tier) / OOS 2020-10..2021-01 (confirm
  once)** -- the tier is NEVER selected on OOS. OOS == the complementarity-matrix OOS (apples-to-apples).
- Repro: git_sha `1756867`, n_shuffle 200 (1000-draw stress on the one hit), seed 13.

---

## THE HONEST FRAME (binding -- inherited from COMPLEMENTARITY.md, respected throughout)

Both sleeves are **LONG-ONLY**, so they cannot rescue each other's down days -- on a market-wide down day
both bleed. The complementarity value the prior phase measured is therefore **DRAWDOWN-DAMPENING /
variance-reduction, NOT positive-return rescue**. The dynamic engine's realistic objective is consequently
**BETTER RISK-ADJUSTED return (Sharpe / maxDD / p05) via regime-timed weighting -- NOT magic alpha.** This
report does **not** claim alpha the engine does not have. The bar the engine must beat is the **best STATIC
complementary blend** on a risk-adjusted axis AND the engine's own **weight-shuffle** (timing skill). If
the dynamic engine does not beat the static blend, the static blend is the answer -- reported as such.

---

## HEADLINE  [CLAIM: in-sample 2020-OOS, single bull regime, held-out tier-selection]

**STATIC-BLEND-WINS at 5 of 6 cadences; the dynamic engine adds genuine risk-adjusted value at ONLY 1 of 6
(30m).** Across {1d, 4h, 2h, 1h, 15m} the best static complementary blend matches or beats the dynamic
engine on Sharpe / maxDD / p05 AND the engine shows **no timing skill** (its OOS net does not sit in the
right tail of re-timings of its own weights; shuffle p = 0.18-0.83). At **30m** the dynamic engine beats the
static blend on Sharpe (5.21 vs 4.71), maxDD (-6.7 vs -8.1) AND p05 (15.4 vs 10.9) AND shows real timing
skill (p = 0.0 -- engine net beats all 1000 weight re-timings). **Read honestly: 1 win in 6 cadences at
p<0.10 is at/near the multiple-comparisons false-positive rate -- this is a SINGLE-CADENCE signal to verify,
not a portfolio-wide edge.**

The deployable conclusion for 5/6 TFs is therefore: **ship the STATIC complementary blend; dynamic regime-
timing adds nothing reliable on the 2020-bull OOS.** This is a real, valuable finding -- it tells us the
cheap, robust, interpretable static blend is the right product everywhere except (possibly) 30m.

---

## THE PER-TF RESULT TABLE  [CLAIM: in-sample 2020-OOS, bull-only]

OOS = Oct-Dec 2020. DYN = the dynamic engine (chosen tier). STATIC = the best per-TF complementary blend
(from `complementarity_matrix.json`). VOLTGT = vol-targeted buy-hold. shuf-p = one-sided p of the engine's
OOS net vs its own weight-shuffle distribution (timing-skill test). Tier chosen on VAL.

| TF  | DYN net / Sh / DD / p05 | STATIC net / Sh / DD / p05 | VOLTGT net | shuf-med / p | timing-skill | tier | PBO flip |
|-----|-------------------------|----------------------------|-----------|--------------|--------------|------|----------|
| 1d  | 18.5 / 2.83 / -8.6 / +0.3 | 15.2 / 2.89 / -5.4 / +0.1 | 42.6 | 14.2 / 0.19 | no | B | 0.0 |
| 4h  | 16.7 / 2.54 / -7.7 / -4.4 | 19.4 / 2.68 / -7.5 / -4.0 | 47.8 | 18.9 / 0.83 | no | B | 0.0 |
| 2h  | 35.9 / 2.97 / -13.2 / -4.7 | 32.2 / 3.06 / -12.1 / -1.5 | 55.5 | 31.1 / 0.23 | no | B | 1.0 |
| 1h  | 29.6 / 3.53 / -8.8 / +2.7 | 35.3 / 3.83 / -7.3 / +2.8 | 70.8 | 33.4 / 0.80 | no | B | 1.0 |
| **30m** | **49.1 / 5.21 / -6.7 / +15.4** | 43.9 / 4.71 / -8.1 / +10.9 | 93.6 | 39.5 / **0.0** | **YES** | B | 0.0 |
| 15m | 45.5 / 4.37 / -7.1 / +9.0 | 64.9 / 6.11 / -7.3 / +26.7 | 106.5 | 51.3 / 0.70 | no | B | 1.0 |

(net %, ann Sharpe, OOS maxDD %, block-bootstrap p05 % [n_boot=600 block=5].)

**Reading the table, two-sided:**
- **VOLTGT_BH beats both DYN and STATIC on NET at EVERY TF.** Consistent with the prior runway verdict:
  vol-targeted buy-hold is the 2020-bull WEALTH winner. The complementary book's value was NEVER raw net --
  it is the DD/p05 protection (e.g. STATIC p05 is positive at 1d/1h/30m/15m; VOLTGT carries full bull beta
  and the deepest drawdowns). The engine is not built to beat VOLTGT on net and does not.
- **DYN > STATIC on net at 1d (+3.3) and 2h (+3.7)** -- BUT both are **no-skill**: the engine's net does not
  beat its own weight-shuffle (p 0.19 / 0.23). That extra net is **average exposure**, not timing: in a bull
  the engine ran a slightly higher mean trend-weight (1d w~0.53 vs static 0.50; 2h w~0.58 vs static 0.48),
  and higher trend exposure simply earns more in a bull. The shuffle preserves the weights, so beating it
  would be timing; failing to beat it proves the gain is the level, not the timing.
- **STATIC > DYN outright at 4h, 1h, 15m** (Sharpe AND net) -- the engine actively underperformed the fixed
  optimal blend there. At 15m especially (STATIC Sharpe 6.11 vs DYN 4.37) the engine HURT.
- **30m is the lone clean win** -- see below.

---

## THE ONE HIT -- 30m  [CLAIM: in-sample 2020-OOS, single cadence, stress-checked]

At 30m the dynamic engine genuinely added risk-adjusted value:
- **Beats STATIC on all three risk-adjusted axes**: Sharpe 5.21 vs 4.71, maxDD -6.7 vs -8.1, p05 +15.4 vs
  +10.9 -- AND higher net (49.1 vs 43.9).
- **Real timing skill, stress-confirmed**: at 1000 weight-shuffle draws the engine's OOS net (49.1%) beats
  **every single re-timing** (percentile 1.0, p = 0.0); even the p95 shuffle is only 48.8%. The win is not a
  lucky single draw.
- **The weight genuinely tracks regime**: `corr(engine_weight, trend_strength_feature) = +0.41` over the 12
  OOS windows -- the engine tilts to trend when trend-strength is high, the mechanism the user asked for.
- PBO rank-flip = 0.0 (the VAL-best tier stayed best OOS), Tier-B (HistGB) chosen on VAL (Sharpe 3.02 > A's
  1.74).

**But the honest caveat on this one hit (binding):**
1. **Multiple comparisons.** 1 hit in 6 cadences at a 10% per-test threshold is roughly what chance delivers.
   Treat 30m as a *single-cadence candidate to verify*, NOT as a confirmed portfolio edge. (Quant referee
   recommended before any ship: family-wise correction / a 30m-only pre-registered re-test on a fresh band.)
2. **Part of 30m's net edge is also a LEVEL effect.** The engine ran mean trend-weight 0.429 vs static's
   min-var 0.333 -- a higher trend tilt that helps in a bull independent of timing. The shuffle controls for
   timing (same weights, re-timed) but not for "the engine simply chose a higher average trend weight than
   the min-var static." So 30m's edge = (real timing skill) + (a higher trend level that happens to help in
   a bull). Only the first survives a regime flip.
3. **2020-bull-only.** The trend tilt that helped at 30m is exactly what a bear would punish.

---

## THE ENGINE (what was built)

Two tiers, both consuming the SAME past-only causal features per rolling 7-day window (the SETUP horizon,
not a candle): `trend_strength` (ADX-like |drift|/churn of the buy-hold proxy), `vol_level`, `vol_regime`,
`chop_vs_trend`, `recent_trend` / `recent_mr` / `perf_spread` (sleeve persistence), `trend_breadth` /
`mr_breadth`. Standardized on TRAIN only.
- **Tier A (interpretable REGIME RULE)**: `w_trend = clip(trend_strength + 0.15*tanh(perf_spread/5), 0.2,
  0.8)`. High trend-strength -> weight trend; chop -> weight MR; never fully drop a sleeve (preserves the
  gap-fill diversification). No fitting -> no OOS leak by construction.
- **Tier B (ML)**: two HistGradientBoosting regressors (sklearn; ridge fallback) predict each sleeve's
  next-window net from the features; the predicted-net SPREAD -> a logistic weight, clipped [0.2, 0.8].
  Small-sample-robust.
- **Selection**: Tier-B won VAL Sharpe at all 6 TFs and was chosen (NEVER on OOS). The interpretable Tier-A
  is shipped alongside as the transparent baseline.

The weight applied to a window is decided ONLY from windows strictly before it (causal lag-1).

---

## CONTROLS (all four ran; the engine only "works" if it beats ALL)  [CLAIM: in-sample 2020-OOS]

| control | what it tests | result |
|---------|---------------|--------|
| **STATIC blend** (B's optimal per-TF weight) | does dynamic timing beat a FIXED optimal blend? | beaten at 1/6 (30m) on risk-adjusted axes |
| **VOLTGT_BH** on net | the bull-beta wealth bar | beats DYN on net at 6/6 (as expected; engine not built for this) |
| **weight-SHUFFLE** (block-shuffle the weight sequence, N draws) | is it TIMING or just average exposure? | timing skill at 1/6 (30m) only |
| **RANDOM** allocator (random weight/window) | trivial floor | DYN ~= random at 1d/4h/2h (no-skill), DYN > random at 30m |

Plus **PBO rank-flip** (did the VAL-best tier stay best OOS): 0.0 at 1d/4h/30m (clean), 1.0 at 2h/1h/15m
(the VAL-best tier flipped OOS -- a partial-overfit flag at those three, reinforcing the "no reliable edge"
read there). **Block-bootstrap p05** reported per row.

The static-blend reconstruction reconciles cleanly vs `complementarity_matrix.json` (Sharpe/maxDD match
within rounding at every TF; net a hair lower due to rolling-window boundary trimming) -- the bar is the
genuine optimal blend.

---

## SELFTEST (two-sided soundness -- runs with NO market data)

`python -m strat.dynamic_allocation_engine --selftest` -> **PASS** on all three legs:
- **POSITIVE**: a planted regime->sleeve map -> a correct causal weight shows TIMING SKILL (book net 147%
  vs shuffle-median 67%, p = 0.03). The apparatus can ACCEPT a genuine signal.
- **NEGATIVE**: i.i.d. noise sleeves -> false-timing-skill rate 4/40 = 0.10 (at the nominal 10%). The
  apparatus does NOT manufacture skill from noise.
- **RULE**: Tier-A gives trend-strong w=0.80 > choppy w=0.20, both clipped [0.2, 0.8].

This two-sided gate is what lets us trust the 30m PASS and the 5x NULL: the engine is proven to both detect
genuine timing AND reject ghosts.

---

## HONEST CAVEATS (binding)

1. **2020-BULL-ONLY, in-sample.** The whole OOS is one bull regime. The trend sleeve and any trend-tilt are
   structurally favored; the DD-dampening value of complementarity is exactly what should generalize
   worse-known until a bear/chop regime is tested. **Synthetic regime-stress is a LATER phase (flagged, not
   done here).** Do NOT carry the magnitudes (esp. fine-TF) forward.
2. **The 30m hit needs multiple-comparisons discipline.** 1/6 at p<0.10 is near the chance rate. It is a
   verify-candidate, not a shipped edge. Recommend `expert-quant` for a family-wise correction + a 30m-only
   pre-registered re-test before any capital.
3. **Long-only -> the objective is risk-adjusted, not alpha.** We measured Sharpe/maxDD/p05, not net-beats-
   VOLTGT (which it never will in a bull). Reported accordingly.
4. **Fine-TF MR magnitudes are overfit** (per COMPLEMENTARITY.md #2) -- the 15m/30m absolute numbers are
   in-sample bull artifacts; what may transfer is the STRUCTURE (the timing relationship), not the levels.
5. **UNSEEN N/A** (2020 band only by mandate). No 2026/other data was touched.

---

## WHAT THIS HANDS THE NEXT PHASE

- A working, two-sided-validated **dynamic/ML allocation engine** (`dynamic_allocation_engine.py`) with two
  tiers, the four mandated controls, a passing two-sided selftest, and per-TF held-out results.
- The honest verdict: **the STATIC complementary blend is the deployable product at 5/6 TFs**; dynamic
  regime-timing adds genuine risk-adjusted value only at 30m, and that single hit needs multiple-comparisons
  verification before it earns capital. **Do not over-engineer a dynamic layer where a fixed blend wins.**
- The open thread for a later phase: **regime-stress the 30m engine on a bear/chop band** (synthetic or a
  different real band) -- if the +0.41 weight-regime correlation and the timing skill survive a regime flip,
  30m is a real edge; if they evaporate, it was the bull-trend-tilt level effect and the static blend wins
  everywhere.

JSON: `runs/periods/TRAIN/2020/DEEP_DIVE/dynamic_engine.json`.
Charts: `charts/dynamic_weights_timeline.png`, `charts/dynamic_vs_static_vs_voltgt_equity.png`,
`charts/dynamic_engine_skill_bars.png`.
Tool: `src/strat/dynamic_allocation_engine.py`.
