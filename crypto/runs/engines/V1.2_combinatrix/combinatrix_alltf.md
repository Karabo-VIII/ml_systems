# V1.2 COMBINATRIX -- ALL COMBO-CADENCES (the all-TF-earned null)

The structured combinatrix had silently defaulted ONE combo-cadence (1d). The charter requires ALL timeframes;
a null is earned only if the full surface (incl. all TFs) was attempted. `run_structured_campaign` is now
parametrized on `combo_cad` (CLI `--combo-cad`), and the full structured search (singles -> pairs ->
beam-triples -> a1a2, full discipline: 2020-VAL-select / 2020-OOS-report / 2021-IRON-confirm + block-bootstrap
p05 + CSCV-PBO + deflated-Sharpe + White's Reality Check) was RUN at **4h, 2h, 1h** (1d already done).
Sequential, in-process, no LLM fan-out; 27 point-in-time u100 names; 2022+ asserted never loaded.

## Per-cadence table

| combo_cad | n_combos | n_names | PBO | White's RC p | #survivors | best combo (non-redundant) | OOS xBH | IRON xBH | p05 vs BH (OOS) | verdict |
|---|--:|--:|--:|--:|--:|---|--:|--:|--:|---|
| **1d** | 275 | 27 | 0.706 | 0.893 | **0** | `OR(KELTNER@4h+ULCER@1d+BBSQZ@1d)` | 0.815 | 0.746 | -43.1 | NULL |
| **4h** | 269 | 27 | 0.214 | 0.705 | **0** | `OR(CKSP@1d+BBSQZ@1d)` | 1.202 | 1.363 | -34.7 | NULL* |
| **2h** | 285 | 27 | 0.250 | 0.468 | **0** | `OR(KELTNER@4h+ADX@4h)` | 1.589 | 1.789 | -5.3 | NULL* |
| **1h** | 285 | 27 | 0.075 | 0.245 | **0** | `OR(KELTNER@4h+ADX@4h)` | 2.051 | 2.697 | +13.0 | NULL* |

`*` xBH > 1.0 (and the lone +ve p05 at 1h) are a **GRID-SCORING ARTIFACT**, not edge -- see the anomaly section.
Buy-hold OOS net per grid: 1d 32.5 / 4h 35.2 / 2h 38.0 / 1h 39.4. IRON-2021 H2 BH: ~62-64 each.

## Overall verdict -- the disciplined null holds across ALL combo-cadences

**0 survivors at every combo-cadence {1d, 4h, 2h, 1h}. The all-TF-earned null is CONFIRMED.** A survivor must
beat its best constituent single AND buy-hold on OOS+IRON net, with block-bootstrap p05 > 0 (vs both), CSCV-PBO
< 0.5, AND White's Reality Check p < 0.05. At every cadence: **White's RC FAILS** (p = 0.89 / 0.70 / 0.47 / 0.25,
never < 0.05 -> the best combo does not survive the multiple-comparisons deflation), **p05 vs best-single stays
NEGATIVE** for every contender, and **IRON-2021 never beats the bull**. The de-risked-beta wall holds under
combination at every combo grid -- combining de-risked-beta TI engines does not manufacture alpha. Confirms
dead-list **D82** across the full combo-cadence surface, not just 1d.

## CADENCE ANOMALY (FLAGGED) -- a grid-scoring artifact, not a survivor

As the combo grid fines (1d -> 4h -> 2h -> 1h), **PBO DROPS** (0.71 -> 0.21 -> 0.25 -> 0.08) and **raw xBH RISES**
above 1.0; at 1h the best non-redundant combo even shows a **positive** p05 vs BH (+13.0). This is the OPPOSITE of
the prior expectation (finer = more churn = PBO stays HIGH). It needed the full red-team -- and it is a **measurement
artifact, NOT edge**:

- **Mechanism.** `_combo_book` applies the lag-1 at the *combo* cadence (`combinatrix.py:222`), so a coarse-native
  signal (e.g. CKSP@1d) ffilled onto a fine combo grid silently reacts ~23h FASTER at every flip -- a hidden change
  of execution latency, plus a within-day Jensen term on partial-exposure days. The strategy's TRIGGERS are
  identical across grids (same n_trades = 4.7, same time_in ~ 0.31), only its FILLS are re-clocked finer.
- **Proof 1 (same strategy inflates).** CKSP@1d single, IDENTICAL trades/exposure on every grid: OOS net
  20.1 (1d) / 43.3 (4h) / 51.3 (1h); Sharpe 2.39 / 4.11 / 4.62. GROSS (no vol-target, no cost): 24.2 / 52.2 / 59.9
  -- so the inflation is NOT cost and NOT vol-target.
- **Proof 2 (falsifier restores invariance).** Apply the lag-1 at the NATIVE 1d cadence (ffill the already-lagged
  position) -> gross OOS is **GRID-INVARIANT: 24.2 / 24.1 / 24.9 / 25.6** across 1d / 4h / 2h / 1h. That isolates the
  artifact to the combo-cadence lag exactly.
- **Proof 3 (the anomaly combo is the artifact).** `OR(KELTNER@4h+ADX@4h)` (both 4h-native) across grids:
  xBH 0.295 / 0.685 / 1.589 / 2.051; p05 vs BH -63.1 / -45.5 / -5.3 / +13.0. At its HONEST 4h native cadence it
  LOSES to BH by a mile (p05 = -45.5). p05 vs best-single stays NEGATIVE (-7.7) even at 1h; robust = False at 1h.
- **Differential favoring.** The artifact flatters in/out strategies (flips + partial exposure) and is ZERO for
  always-in buy-hold (pos == 1 -> daily compounding is grid-invariant): BH inflates only +21% (32.5 -> 39.4, pure
  Jensen) while the trend single inflates +155% (20.1 -> 51.3). So **xBH is not apples-to-apples across cadences**;
  the > 1.0 crossover is a metric artifact, not a real beat.

**Implications.** (1) Do NOT report "combo beats BH at finer cadence" as real edge. (2) PBO = 0.08 at 1h is
CONTAMINATED (the grid-stable artifact dilutes the IS->OOS rank-instability PBO measures) -- do not cite it as
robustness. (3) The **0-survivor verdict is unaffected and honest** -- White's RC (built on the same grid, deflated
by M) + the negative p05-vs-single + IRON-confirm catch the artifact at every cadence (a two-sided-sound gate).
(4) For honest cross-cadence comparison of a coarse-native input, the lag must be applied at the NATIVE TF BEFORE
ffill, not at the combo cadence after (`combinatrix.py:222`) -- the one code note worth a future fix; it does not
change THIS null.

## Artifacts
`combinatrix_alltf_{4h,2h,1h}.json` + `combinatrix_alltf_{4h,2h,1h}.md` (per-cadence leaderboards, n_names on every
row) + `combinatrix_results.json` (1d, reproduced bit-identical = RWYB identity) + this summary +
`combinatrix_alltf_summary.json`. NOT committed.
