# THE DECISIVE LONG-ONLY PARTICIPATE-AND-PRESERVE FRONTIER TEST

**Module:** `src/strat/participate_preserve_frontier.py` (selftest PASS; RWYB)
**Run:** `python -m strat.participate_preserve_frontier` -> `runs/strat/participate_preserve_frontier_20260616_015759.json`
**Charts:** `runs/strat/participation_preservation_frontier.png` + `runs/strat/best_pp_vs_book_equity.png`
**Discipline:** STRICT long-only + spot (zero short logic), select/tune on 2020-2024+2025-OOS, UNSEEN 2025-12-31..2026-06-01
read ONCE at the end, fixed-EW, PIT survivorship-clean, maker cost, causal/lag-1, multiple-comparisons-deflated.

---

## THE QUESTION

The family-ensemble "cash-going book" preserves bears by going to CASH (= non-participation). Every prior
construction is one point on a participation<->preservation frontier; none has a deployable wealth edge.
**Is there ANY long-only construction NORTHEAST of the book on the (bull-capture% vs bear-preservation%)
plane -- MORE participation AND MORE preservation, beating it on full-cycle held-out WEALTH with ship-gate
p05>0 -- or is participation<->preservation a FUNDAMENTAL tradeoff (you can only move along the line)?**

---

## WARM-UP: vol-brake STRENGTH cannot reach the book's bear preservation [VERIFIED]

A more aggressive vol-brake on always-on EW-beta does NOT reach the book's 2022 preservation, and pays a
crippling bull tax -- confirming the user's mechanism note (**vol-target is symmetric to VOLATILITY, not
DRAWDOWN**). The 2022 bear maxDD / 2021 bull net by brake strength on plain beta:

| brake | 2022 bear maxDD | 2021 bull net |
|-------|-----------------|----------------|
| raw beta (none) | -73.4% | +208.4% |
| light | -66.2% | +162.8% |
| medium | -54.4% | +112.4% |
| **heavy** | **-38.3%** | **+63.9%** |
| **cash-going book (light)** | **-19.8%** | **+23.1%** |

Even HEAVY (the brake that already halves the bull) gets the bear only to -38.3% -- still ~18pp worse than
the book's -19.8%. To brake the bear hard enough, a symmetric vol-brake must brake the bull just as hard.
**Preservation requires a DIRECTIONAL (trend/drawdown) gate that goes to cash on the way DOWN and reinvests
on the way UP -- not a symmetric vol-brake.** This motivates the 4 constructions.

---

## THE FRONTIER (held-out DEV grade, 2020-2025) [VERIFIED-HELDOUT]

Four long-only directional gates, each a causal per-asset {0,1}->vol-scaled position on EW-beta, tuned on
DEV only (UNSEEN sealed). The plane: **bull-capture %** = construction bull-year compounded wealth / raw-beta
bull-year wealth (participation); **bear-preservation %** = 1 - construction worst-bear-DD / raw-beta
worst-bear-DD (preservation). Full-cycle DEV = the 2020-2025 calendar chain.

| construction | bull-capture % | bear-preservation % | full-cycle DEV net | DEV maxDD | NE of book? |
|--------------|----------------|---------------------|--------------------|-----------|-------------|
| raw beta (buy-hold) | 100.0 | 0.0 | +50.4% | -78.9% | (benchmark) |
| family-free vol-gate | 59.4 | 9.8 | +39.6% | -72.9% | no |
| bear_rally | 55.1 | 32.2 | +141.3% | -55.5% | no |
| terminal_leg | 34.5 | 42.4 | +126.8% | -49.1% | no |
| asymmetric | 42.9 | 45.0 | +253.9% | -49.4% | no |
| **cash-going book (incumbent)** | **9.5** | **64.9** | **+53.3%** | **-25.8%** | (incumbent) |
| **drawdown_aware** | **18.0** | **79.0** | **+151.7%** | **-28.0%** | **YES (NE)** |

**Read of the plane (see `participation_preservation_frontier.png`):** raw-beta (SE corner: full capture, zero
preservation) and the cash-book (NW corner: 9.5% capture, 64.9% preservation) bracket a roughly straight,
downward-sloping frontier line on which `bear_rally`, `terminal_leg`, `asymmetric` and `family_free` all sit
-- a clean participation<->preservation tradeoff with **no free lunch**. **Exactly one construction breaks NE
of the book: `drawdown_aware`** (more bull-capture: 18.0 > 9.5, AND more preservation: 79.0 > 64.9). So on the
DEV plane, participation<->preservation is **NOT** a single fundamental line -- a dominating point exists.

**Why the dominators out-compound raw beta on DEV wealth (legitimate, not a bug) [VERIFIED]:** the chain math
reconciles exactly. raw-beta makes +208% in 2021 then gives back -71.9% (2022) and -61.9% (2025), netting only
+50.4%. `drawdown_aware` captures less of each bull (+64% not +208% in 2021) but loses only -12.6%/-14.0% in
the two bears -> +151.9% chained. **Avoiding the -70% years is worth more to compound wealth than capturing
the full bull.** This is the real "preservation compounds" mechanism, not double-counting.

---

## THE DECISIVE VERDICT: `DEV_DOMINATES_BUT_NOT_SHIPPABLE` -> CLOSED FOR DEPLOYMENT

The single NE point (`drawdown_aware`) dominates on the DEV plane and out-compounds the book on DEV wealth.
But it **fails BOTH adversarial held-out gates** -- and the failures are STABLE across block sizes and seeds
(K=3 independent re-derivations), so this is not a knife-edge call.

### Gate 1 -- held-out ship-gate p05 (block-bootstrap, OOS+UNSEEN held-out) [VERIFIED]

| block | held-out p05 | held-out p50 | held-out p95 |
|-------|--------------|--------------|--------------|
| 5 | **-28.6%** | -9.1% | +14.2% |
| 10 | **-27.8%** | -9.4% | +14.0% |
| 20 | **-23.5%** | -8.2% | +9.9% |

Held-out p05 is **robustly negative** (and held-out p50 is negative too). The full-stream p05 is positive
(+16 to +19%) ONLY because it includes the 2021 mega-bull selection window; once you isolate the held-out
(2025-03-15 onward, bear-heavy) portion the left tail is deeply negative. **Ship-gate FAILS.**

### Gate 2 -- multiple-comparisons deflation (best-of-4 must beat the max-statistic null) [VERIFIED]

Permutation max-statistic null (sign-flipped block-bootstrap of each construction's excess-over-book stream,
taking the MAX edge across all 4 constructions each draw -> the null already embeds the best-of-K selection):

| block | seed | observed best edge | null max-edge p95 | p-value |
|-------|------|--------------------|-------------------|---------|
| 5 | 7 | +200.6pp | +219.7pp | **0.065** |
| 10 | 11 | +200.6pp | +220.1pp | **0.065** |
| 20 | 23 | +200.6pp | +246.3pp | **0.084** |

**PBO (CSCV) across the construction family: 0.671** (PBO ~ 0.5 = skill-less construction-selection; 0.67 =
the in-sample-best construction tends to under-perform out-of-sample). The +200pp dev edge over the book is
**within the best-of-4 noise band** (p consistently > 0.05). **Deflation FAILS.**

### Synthesis

```
GATES: dominates_plane = True     <- a NE point exists on the DEV plane (drawdown_aware)
       beats_book_wealth = True   <- +151.7% vs +53.3% DEV (the preservation-compounds mechanism is real)
       unseen_pos = True          <- drawdown_aware UNSEEN +2.0% (book -0.9%), preservation kept (-7.1% DD)
       heldout_p05>0 = FALSE      <- held-out p05 robustly -23% to -29% (FAILS ship-gate)
       deflation_survives = FALSE <- permutation p = 0.065-0.084, PBO = 0.67 (best-of-4 within noise)
```

**VERDICT: the dev-plane dominance is a SELECTION-WINDOW + best-of-several MIRAGE, not a deployable edge.**
The one construction that breaks NE of the frontier owes its dev win to (a) the 2021 bull captured in the
selection window and (b) being the luckiest of 4 tries. On the held-out portion it has a negative p05 left
tail and does not survive multiple-comparisons deflation. **For real capital, long-only participate-and-
preserve is effectively a fundamental tradeoff: you can MOVE participation<->preservation but you cannot
robustly BEAT the frontier.** The honest doors remain **SHORT (OFF -- the user's shortcut)** or **CARRY**
(the funding-dispersion dollar-neutral sleeve, the project's one held-out-positive edge).

> Nuance the referee will NOT inflate away: the dominance is not pure noise -- `drawdown_aware` is genuinely a
> better-shaped point than the rest of the family (it really did beat the book on dev wealth AND was UNSEEN-
> positive). The honest statement is **AMBIGUOUS-leaning-NEGATIVE / NOT-SHIPPABLE**, not "definitively zero".
> The verdict is "not deployable on the current evidence", which for asymmetric-loss real-capital is a NO.

---

## THE SINGLE CHEAPEST FALSIFIER

> **One construction that lands NE of the cash-going book AND clears held-out p05 > 0 AND survives the
> max-statistic permutation (p < 0.05) would re-open long-only participate-and-preserve.** None of the 4
> directional gates did. The next-cheapest probe before declaring the door permanently shut is a
> **regime-conditioned ENSEMBLE of the 4 gates** (does blending them shift the held-out p05 positive without
> re-inflating the multiple-comparisons count?). If that also fails the held-out p05 + deflation gates, the
> door is shut and the remaining alpha must come from SHORT (OFF) or CARRY -- not from re-shaping long-only beta.

A second, even cheaper falsifier of the *negative* conclusion: if a single re-run on a different block size
or seed flipped the permutation p below 0.05 or the held-out p05 above 0, the "not-shippable" call would be
fragile. **K=3 re-derivations confirm both failures are stable** (p in {0.065,0.065,0.084}; held-out p05 in
{-28.6,-27.8,-23.5}) -- the negative conclusion is robust, not a seed artifact.

---

## CLAIM-TAG LEDGER

| claim | tag | basis |
|-------|-----|-------|
| vol-brake (even heavy) cannot reach the book's bear preservation | VERIFIED | warm-up table, RWYB |
| a NE point exists on the DEV plane (`drawdown_aware`) | VERIFIED-HELDOUT (DEV) | frontier table; dev 2020-2025 |
| dominators out-compound raw beta via preservation-compounds, not double-count | VERIFIED | per-year chain reconciliation |
| held-out p05 robustly negative (-23% to -29%) | VERIFIED | block-bootstrap K=3 (blocks 5/10/20) |
| best-of-4 edge within multiple-comparisons noise (p 0.065-0.084) | VERIFIED | permutation max-stat K=3; PBO 0.67 |
| long-only participate-and-preserve is CLOSED FOR DEPLOYMENT | VERIFIED (decision) | both held-out gates fail, stably |
| the honest open doors are SHORT (OFF) or CARRY | INHERITED | MEMORY funding-dispersion finding; charter |

No emoji (cp1252). Does NOT git commit. UNSEEN touched exactly once.
