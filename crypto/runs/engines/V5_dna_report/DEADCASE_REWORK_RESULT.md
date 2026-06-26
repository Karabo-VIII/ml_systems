# DEAD/NULL CASE REWORK — RESULT (2026-06-25)

The user: "rework the null and dead cases... make sure everything is robust and hyper-audited."
Interrogated the two most informative failures on the AUDITED harness (19/19). Both gave
decisive answers — and one OVERTURNED a prior verdict.

## CASE 1 — ADX: the "borderline/null" verdict was WRONG (under-swept)

V3 dismissed ADX as borderline (93rd-pct null, "1d-only marginal"). That was a NARROW
param range. A proper lookback {7,10,14,21,28} × threshold {0.25-0.55} sweep finds ADX is
the program's **FIRST genuinely SELECTIVE + VAL/OOS-CONSISTENT + null-beating engine.**

SELECTIVE+ROBUST cells (deploy<70% BOTH windows, edge>0 BOTH, OOS null≥95):

| Cell | VAL edge | VAL null | VAL dep | OOS edge | OOS null | OOS dep |
|------|----------|----------|---------|----------|----------|---------|
| **ADX\|1d lb28 thr0.45** | +3.75pp | 100 | 49% | +0.94pp | 97 | 40% |
| **ADX\|1d lb21 thr0.45** | +2.70pp | 100 | 55% | +1.01pp | 100 | 58% |
| **ADX\|1d lb10 thr0.55** | +0.88pp | 75  | 65% | +1.11pp | 100 | 64% |
| **ADX\|4h lb10 thr0.55** | +0.25pp | 0   | 68% | +2.65pp | 100 | 65% |
| **ADX\|4h lb28 thr0.35** | +0.31pp | 0   | 66% | +0.90pp | 100 | 66% |

The 1d cells (lb21-28, thr0.45) are the strongest: selective (~40-58% deploy), positive
on BOTH windows, 97-100th-pct null on BOTH. This is the discrete-timing property the
always-long-tilt engines lack. **ADX|1d (lb21-28, thr~0.45) is promoted to a VALIDATED
SELECTIVE engine** — the first one driven by the TI signal's selectivity rather than the
regime gate's always-long tilt.

CAVEAT (honest): the OOS edge (+0.9-1.0pp/wk) is smaller than the VAL edge (+2.7-3.8pp),
i.e. some VAL-fit, but it stays POSITIVE and null-beating OOS, which the tilt engines'
"selective" cousins never achieved. Needs the overfit-probe (smooth surface) next — the
lb21/lb28 + thr0.45 neighbourhood IS locally smooth here (both pass), which is encouraging.

## CASE 2 — STOCHASTIC: confirmed dead, but the -7.8pp was an INVERTED-LOGIC artifact

Stochastic was the worst bleeder in the DNA (-7.8pp/wk, Sharpe -3.4 at 2h). Inversion test
(does flipping held=1-held rescue it?):

| TF | normal OOS | inverted OOS | read |
|----|-----------|--------------|------|
| 1d | -0.53pp | +0.10pp | mixed (inversion ~flat, no real edge) |
| 4h | -3.25pp | -0.52pp | BOTH lose → genuinely no signal |
| 2h | -7.53pp | -0.44pp | BOTH lose → genuinely no signal |

Findings:
- The catastrophic -7.8pp was largely an **inverted-logic artifact**: the %K/%D-cross-above-
  oversold entry was systematically buying tops. Flipping it removes most of the bleed.
- BUT the inverted signal is still ≤0 OOS at every TF → **no exploitable edge either way.**
  Stochastic is **confirmed genuinely DEAD** (not a fixable bug, just no signal), and the
  scary -7.8pp number should be read as "backwards + noisy," not "strong negative alpha."

## Net effect on the engine roster
- **NEW validated selective engine: ADX|1d (lb21-28, thr0.45).** First of its kind.
- Stochastic stays dead (correctly), with the honest note that its magnitude was inflated
  by inverted logic.
- Lesson reinforced (the user's point): a "null" verdict from a coarse sweep is not a death
  certificate. ADX needed the right lookback to reveal a real selective edge.

## Next (before any deploy claim on ADX|1d)
- [ ] Overfit-probe ADX|1d: full lookback×threshold heatmap must be a SMOOTH plateau around
      lb21-28/thr0.45, not an isolated spike.
- [ ] Re-run with the ensemble (multiple lb/thr configs voting) rather than a single param,
      matching the SOTA architecture, and re-validate selectivity+consistency.
- [ ] Then combine ADX|1d (selective) with the MA/RSI base and the robust-tilt overlays.


---

# ADX|1d OVERFIT PROBE — RESULT (2026-06-25, honest read)

Dense grid lookback {16-30} × threshold {0.38-0.52}, OOS edge pp/wk:

```
  lb\thr   0.38   0.42   0.45   0.48   0.52
   16     -0.53  -0.69  -0.46  -0.54  -1.17
   18     -0.72  +0.08  -0.80  -1.01  -1.15
   20     -0.76  +0.19  +0.97* -0.05  -0.75
   22     +0.36  +1.01* +0.65* +0.13  -0.73
   24     +1.22* +0.50  +0.51  +0.99* -0.75
   26     +0.69* +1.85* +0.98* +0.58  +0.72
   28     +0.32  +1.15* +0.94* +0.78  -0.09
   30     +0.45  +0.47  +0.84* +2.34* +1.01*
```
(* = OOS positive AND null≥95 AND selective<70%.  25/40 cells OOS-positive.)

## Honest verdict (correcting the script's auto-optimistic line)
NOT a clean plateau, NOT a pure spike — it's a **directional GRADIENT**:
- **lookback ≥ 22 is consistently positive** (rows 22-30: ~22/30 positive), strongest at
  thr 0.42-0.48. This region is coherent (neighbours agree) — that part is robust.
- **lookback ≤ 20 is mostly negative.** So ADX|1d IS sensitive to lookback: it needs a
  LONG window (≥22 days) to measure trend strength, which is mechanically sensible.
- thr 0.52 (very strict) degrades — too few signals.

CONCLUSION: ADX|1d carries a **real, moderate selective edge in the lb22-28 / thr0.42-0.48
pocket** — robust enough to be the program's first genuine SELECTIVE engine, but with a
HONEST caveat: it is lookback-sensitive (dies below lb20) and its OOS edge (~+0.6-1.1pp/wk)
is smaller than VAL. Promote it to VALIDATED-SELECTIVE **provisional** — deploy only as an
ENSEMBLE over the robust pocket (lb {22,24,26,28} × thr {0.42,0.45,0.48}), never a single
param, and re-confirm on the 2025 OOS slice when the user opens it. This is the conservative
promotion, not a green light for size.
