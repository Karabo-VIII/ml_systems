# Breakout archetype through the same gauntlet -- does the MA pattern generalize?

Founding mandate (MEMORY.md): "explore ALL indicators broadly+freshly; treat 'exhausted' as a hypothesis
to re-test." The fixed-approach arc tested ONE archetype (MA-cross). This runs a structurally DIFFERENT
one -- **Donchian breakout** (long when close breaks the prior N-bar high; exit on the prior M-bar low) --
through the same gauntlet (NAIVE -> FIXED family -> FULL stack -> OOS + scorecard). Tool:
`src/strat/breakout_arc.py`. 4h, equal-weight u10 book, causal MtM. UNSEEN sealed (series ends 2025-12-31).

## Variant x period (ROI% / maxDD%), 4h
| variant | Jun2022 bear | VAL | OOS | OOS %posCells |
|---|---|---|---|---|
| NAIVE (26 cfg, taker) | -5.9 / -5.6 | 36.1 / -17.6 | 5.0 / -17.7 | 59% |
| FIXED (slow-N[50,150], taker) | -5.2 / -5.7 | 32.8 / -18.0 | 8.0 / -16.3 | 64% |
| **FULL (+trail+minhold+maker)** | **-3.1 / -2.9** | 19.1 / -10.7 | **+11.0 / -15.4** | **72%** |

## Generalization findings
1. **The PATTERN generalizes.** Same signatures as MA: (a) the FAMILY fix (slow-N) helps OOS (+3.0pp)
   and the bear (+0.7), slightly hurts VAL; (b) the FULL stack is a drawdown controller -- it HURTS the
   VAL bull (-17pp: the trail cuts breakout's big VAL run) but HELPS the bear (+2.8) and OOS (+6.0).
2. **Breakout is a MATERIALLY BETTER archetype than MA-cross out of sample** (FULL stack, OOS):
   | metric | MA-cross | Donchian breakout |
   |---|---|---|
   | OOS compound | +2.0% | **+11.0%** |
   | OOS Sharpe | 0.23 | **0.72** |
   | OOS maxDD | -23.6% | **-14.4%** |
   | OOS breadth | 5/10 | **72% (~7/10)** |
   | OOS-heldout block-bootstrap p05 | -33% | **-21%** |
   On the held-out OOS, breakout wins on EVERY axis -- higher return, ~3x Sharpe, ~40% shallower drawdown,
   broader, and a less-negative tail. The entry ARCHETYPE is a bigger lever than the MA-arc treated it.
3. **BUT the absolute-robustness ceiling persists.** Breakout FULL OOS-heldout p05 = -21.2% -- still
   NEGATIVE (fails the North Star p05>0 bar), bear still -3.1%. Breakout is CLOSER to robust than MA but
   does not clear it. Same ceiling, less far from it.

## Verdict
The founding "explore freshly" mandate paid off: **switching the entry archetype (breakout vs MA-cross)
buys more held-out quality than any structural overlay did** (OOS Sharpe 0.23 -> 0.72, breadth 50% ->
72%). The MA-centric framing was too narrow. The drawdown-control + family-fix pattern is archetype-general
(good -- it's a real structural truth, not an MA artifact). The honest ceiling still stands (p05<0, bear
negative) -- but breakout is the better base, and it is close enough that the right next question is
whether breakout + ONE more orthogonal lever (NOT another internal-price variant -> over-mining; rather a
DIFFERENT instrument/timeframe/an external signal) can clear p05>0. Caveat: one archetype, one OOS window,
TRAIN-era design; confirm before belief. UNSEEN never touched.

json: `breakout_arc.json`. RWYB: `python -m strat.breakout_arc`.

## Timeframe sweep (HARD RULE: never default one cadence) -- `breakout_tf_sweep.py`
4h was validated as breakout's genuine sweet spot (not an arbitrary default):
| cadence | OOS roi/maxDD | OOS breadth | OOS Sharpe | OOS-heldout p05 |
|---|---|---|---|---|
| 1d | -6.7 / -14.1 | 32% | -0.66 | -21.14 |
| **4h** | **+11.0 / -15.4** | **72%** | **0.72** | -21.21 |
| 1h | +2.0 / -18.6 | 52% | 0.22 | -28.04 |
| 30m | +1.6 / -19.1 | 50% | 0.16 | -28.42 |
4h dominates the held-out OOS on every axis (compound, Sharpe, breadth). 1d is too coarse (OOS-negative,
32% breadth); 1h/30m pay more cost and degrade (Sharpe ~0.2). **The p05 ceiling holds at EVERY cadence**
(best -21) -- the timeframe axis cannot clear robustness either. Cadence is material (Sharpe 0.72 -> -0.66
across cadences) so the sweep mattered; the answer is 4h, principled.
