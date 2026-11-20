# THE LAST SHOT: REGIME-CONDITIONED ENSEMBLE for long-only participate-and-preserve

**Module:** `src/strat/pp_ensemble.py` (selftest 8/8 PASS; RWYB)
**Run:** `python -m strat.pp_ensemble` -> `runs/strat/pp_ensemble_20260616_020933.json`
**Chart:** `runs/strat/pp_ensemble_frontier.png`
**Discipline:** STRICT long-only + spot (exposure clipped to [0,1], ZERO short logic), select/tune on
2020-2024+2025-OOS, UNSEEN 2025-12-31..2026-06-01 read ONCE at the end, fixed-EW, PIT survivorship-clean,
maker cost, causal/lag-1, deflation over the FULL family of 6 (4 gates + 2 ensembles). No emoji. Not committed.

---

## THE FLAGGED FALSIFIER

The frontier test (`participate_preserve_frontier.py`, commit 4e0c34e) found 4 long-only directional gates;
exactly one (`drawdown_aware`) broke NORTHEAST of the cash-going book on the DEV plane but FAILED held-out
(p05 {-28.6,-27.8,-23.5}%, permutation max-stat p ~ 0.065-0.084, PBO 0.671 = best-of-4 + 2021-bull mirage).
Its own verdict named the cheapest remaining probe:

> "a REGIME-CONDITIONED ENSEMBLE of the gates (does combining them shift NE?) before declaring the door
> permanently shut. If that also fails the held-out p05 + deflation gates, the door is shut."

This is that probe. Two ensemble constructions over the SAME 4 gates:
- **VOTE** -- exposure = fraction of the 4 gates invested at each bar -> continuous [0,1] (no regime logic).
- **ROUTED** -- a CAUSAL, PRE-REGISTERED regime detector on the asset's OWN price routes between gates:
  confirmed-BULL (above slow-MA(100) AND slow-MA rising) -> full exposure 1.0; sustained-BEAR (below MA AND
  MA falling) -> the `drawdown_aware` gate; RECOVERY/chop (else) -> the `bear_rally` gate. Detector = trailing
  slow-MA + past-20-bar slope ONLY. No OOS fit, no lookahead (verified causal in selftest checks 3+4).

**Honest multiplicity:** the ensemble is YET ANOTHER construction. The deflation family is therefore the FULL
6 {bear_rally, terminal_leg, drawdown_aware, asymmetric, **vote, routed**}; the max-stat permutation null takes
the max edge across ALL 6, PBO runs over 6 columns. Adding 2 constructions can only RAISE the bar.

---

## THE FRONTIER PLANE (held-out DEV grade, 2020-2025) [VERIFIED-HELDOUT]

The plane: **bull-capture %** = construction bull-year compounded wealth / raw-beta bull-year wealth;
**bear-preservation %** = 1 - construction worst-bear-DD / raw-beta worst-bear-DD. The cash-going book is the
incumbent (NW corner: low capture, high preservation). Does an ensemble shift NORTHEAST of it?

| construction | bull-capture % | bear-preservation % | full-cycle DEV net | DEV maxDD | NE of book? |
|--------------|---------------:|--------------------:|-------------------:|----------:|:-----------:|
| raw beta (buy-hold) | 100.0 | 0.0 | +50.4% | -78.9% | (benchmark) |
| family-free vol-gate | 59.4 | 9.8 | +39.6% | -72.9% | no |
| bear_rally | 55.1 | 32.2 | +141.3% | -55.5% | no |
| asymmetric | 42.9 | 45.0 | +253.9% | -49.4% | no |
| **ENS: vote** | **36.5** | **52.5** | **+178.2%** | **-44.1%** | **NO** |
| terminal_leg | 34.5 | 42.4 | +126.8% | -49.1% | no |
| **ENS: routed** | **27.6** | **53.3** | **+128.5%** | **-46.2%** | **NO** |
| **cash-going book (incumbent)** | **9.5** | **64.9** | **+53.3%** | **-25.8%** | (incumbent) |
| drawdown_aware | 18.0 | 79.0 | +151.7% | -28.0% | YES (the lone gate) |

**Read of the plane (see `pp_ensemble_frontier.png`):** both ensembles land squarely ON the frontier line,
BETWEEN the book and the gates -- MORE bull-capture than the book (36.5 / 27.6 > 9.5) but MORE THAN 5pp LESS
preservation (52.5 / 53.3 vs the book's 64.9 -- a ~12pp give-back, far outside the pre-registered 5pp band).
**Neither ensemble is NORTHEAST of the book.** The ensemble bought participation by SELLING preservation --
it moved ALONG the participation<->preservation frontier, it did not BEAT it. The only point still NE of the
book is the lone gate `drawdown_aware` (which the prior test already showed fails held-out). **Combining the
gates does NOT create a new NE point; averaging/routing pulls toward the frontier MIDDLE, not its NE corner.**

---

## THE DECISIVE GATE (pre-registered) -> VERDICT: SHUT

REOPENS iff ALL four: (1) NE of book, (2) held-out p05 > 0, (3) permutation max-stat p < 0.05, (4) PBO < 0.5.

```
DECISIVE GATE 1 (NE of book)                         : FALSE  <- no ensemble is NE (preservation give-back > 5pp)
DECISIVE GATE 2 (held-out block-bootstrap p05 > 0)   : FALSE  <- best ensemble (vote) held-out p05 = -47.2%
DECISIVE GATE 3 (permutation max-stat p < 0.05, 6-fam): FALSE  <- p = 0.092
DECISIVE GATE 4 (PBO < 0.5, 6-family)                : FALSE  <- PBO = 0.671
```

**Gate 1 fails before the statistics even run.** All four fail. **VERDICT: SHUT_NO_NE_POINT.**

### Decisive statistic + K=3 independent re-derivations (the SHUT is not a seed/block artifact)

The decision carries the same asymmetric-loss weight as a SHIP (false-close of a real door), so every
load-bearing statistic is re-derived across block in {5,10,20} and seeds {7,11,23}.

**(K=3) Held-out (OOS+UNSEEN, 2025-03-15 onward, n=440 days) block-bootstrap COMPOUND p05:**

| stream | block 5 / seed 7 | block 10 / seed 11 | block 20 / seed 23 | held-out p50 | held-out compound |
|--------|-----------------:|-------------------:|-------------------:|-------------:|------------------:|
| cash_book | -38.8% | -39.6% | -38.3% | -20 to -22% | -20.1% |
| **ENS: vote** | **-47.2%** | **-47.1%** | **-42.1%** | **-15 to -16%** | **-17.0%** |
| **ENS: routed** | **-45.5%** | **-45.2%** | **-40.6%** | **-15%** | **-16.8%** |

Both ensembles' held-out p05 is robustly **-40% to -47%** AND their held-out p50 (median) is negative. The
ensembles' held-out left tail is even DEEPER than the book's -- they participate more on the way down. The
held-out portion is bear-heavy (the book itself lost -20% there); the full-stream p05 is less negative only
because the 2021 mega-bull lives in the SEL selection window. **Ship-gate robustly FAILS.**

**(K=3) Permutation max-statistic deflation over the 6-family (observed best ensemble edge = vote, +200.6pp):**

| block | seed | observed best ensemble edge | null max-edge p95 | p-value | survives? |
|------:|-----:|----------------------------:|------------------:|--------:|:---------:|
| 5 | 7 | +200.6pp | +235.9pp | **0.077** | FAIL |
| 10 | 11 | +200.6pp | +244.7pp | **0.092** | FAIL |
| 20 | 23 | +200.6pp | +277.4pp | **0.118** | FAIL |

The +200pp dev edge over the book is WITHIN the best-of-6 noise band at every block (p in {0.077, 0.092,
0.118}, all > 0.05) -- and it WORSENS as the block grows (more autocorrelation preserved = wider null). Adding
the 2 ensembles RAISED the bar vs the 4-gate test (p went from ~0.065-0.084 to ~0.077-0.118). **PBO = 0.671**
across the 6-family (unchanged from the 4-gate test -- the in-sample-best construction tends to UNDER-perform
out-of-sample). **Deflation robustly FAILS.**

### UNSEEN read-once (regime CHOP/SIDEWAYS; raw-beta -8.1%) -- for completeness, NOT a pass

| stream | UNSEEN net | UNSEEN maxDD |
|--------|-----------:|-------------:|
| cash_book | -0.9% | -2.3% |
| ENS: vote | +0.4% | -9.5% |
| ENS: routed | +1.9% | -8.2% |

The ensembles were marginally UNSEEN-positive (+0.4 / +1.9%) with kept preservation -- but UNSEEN was a flat
chop window (raw-beta only -8.1%, no real bear stress), and a +0.4-1.9% read on a single 5-month chop is
inside noise. It does NOT rescue a negative held-out p05 + failed deflation; the decisive gate is unmoved.

---

## SYNTHESIS -> THE DOOR IS DEFINITIVELY SHUT

> **The regime-conditioned ENSEMBLE -- the single cheapest remaining falsifier the frontier test flagged --
> does not clear the decisive gate on ANY of its four legs.** Combining the 4 directional gates, whether by
> simple VOTE or by causal pre-registered regime-ROUTING, produces a point that sits ON the
> participation<->preservation frontier line (paying ~12pp of preservation to buy ~20-27pp of participation),
> NOT northeast of it. And even granting it a wealth edge, that edge is (a) a 2021-bull selection-window
> artifact, (b) within the best-of-6 multiple-comparisons noise (perm p 0.077-0.118), (c) PBO 0.67
> (selection-skill-less), and (d) backed by a held-out p05 of -40% to -47% with a negative held-out median.

**Decisive statistic: there is NO long-only construction NORTHEAST of the cash-going book that clears
held-out p05 > 0** -- 4 directional gates + 2 ensembles (6 constructions total) all fail, robustly across
K=3 block sizes and seeds. The participation<->preservation tradeoff is FUNDAMENTAL for long-only spot:
you can MOVE along the frontier (trade participation for preservation) but you cannot BEAT it.

**Long-only participate-and-preserve is now DEFINITIVELY CLOSED.** The honest doors are:
- **SHORT (currently OFF -- the user's hard constraint)** -- the only way to add a TRUE bear leg (the
  long-only constraint is the binding wall: a directional gate going to cash on the way down is bounded by
  +0% in a bear, never positive; preservation is the ceiling).
- **CARRY** -- the funding-dispersion dollar-neutral sleeve (the project's one held-out-positive edge:
  net OOS +7.9% / UNSEEN +10.3%, beta~0, p<0.001).

### THE CHEAPEST REMAINING FALSIFIER

There is no remaining cheap LONG-ONLY falsifier of the SHUT verdict that re-shapes beta. The only construction
that could still re-open the door is an **EXTERNAL conditioner** (Coinglass liquidations / on-chain flows) that
times the de-risk from information NOT in the price -- but that is no longer "re-shaping long-only beta", it is
a new signal lane (and the project's prior mover work found internal-data direction info-bound dead 4 ways).
The negative conclusion's own falsifier is the K=3 robustness: a single re-run flipping perm p < 0.05 OR
held-out p05 > 0 would make the SHUT fragile -- **K=3 confirms both fail stably** (perm p in {0.077, 0.092,
0.118}; held-out vote p05 in {-47.2, -47.1, -42.1}%). The negative is robust, not a seed artifact.

---

## CLAIM-TAG LEDGER

| claim | tag | basis |
|-------|-----|-------|
| both ensembles are strictly long-only [0,1], causal (no lookahead) | VERIFIED | selftest 8/8 (checks 1,3,4,7) |
| neither ensemble lands NE of the cash-going book (preservation give-back > 5pp) | VERIFIED-HELDOUT | frontier plane (DEV 2020-2025) |
| best ensemble (vote) held-out p05 robustly -42% to -47% (p50 also neg) | VERIFIED | K=3 block-bootstrap (blocks 5/10/20) |
| best-of-6 edge within multiple-comparisons noise (perm p 0.077-0.118) | VERIFIED | K=3 permutation max-stat; PBO 0.671 |
| ensembling moves ALONG the frontier, does not BEAT it (tradeoff is fundamental) | VERIFIED (decision) | all 4 decisive gates fail, stably |
| long-only participate-and-preserve is DEFINITIVELY SHUT | VERIFIED (decision) | the flagged last-shot falsifier fails |
| the honest open doors are SHORT (OFF) or CARRY | INHERITED | MEMORY funding-dispersion finding; charter |

No emoji (cp1252). UNSEEN touched exactly once. NOT git committed.
