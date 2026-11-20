# The complete stack — full keeper + flicker-fixed BTC market gate

Assembles the full keeper stack (FIXED 2MA-slow + TRAIL 10% + min_hold 12 + MAKER) and resolves the BTC
market gate's one defect (the 1h bull flicker) via hysteresis / a slower BTC SMA. Tool:
`src/strat/complete_stack.py`. Equal-weight u10 book, causal MtM, MAKER, all TRAIN-era. ROI% / maxDD%.

Consistency: GATE_NONE reproduces fixed_stack's full-stack numbers exactly (4h comb 22.1, bear -6.1,
bull 23.3).

## Gate forms
| gate | rule |
|---|---|
| GATE_NONE | full keeper stack, no market gate |
| GATE_BTC100 | BTC > BTC.SMA100 (raw — flickers at fine cadence) |
| **GATE_BTC100_H** | BTC vs SMA100 with a 3% HYSTERESIS band (latched) |
| GATE_BTC200 | BTC > BTC.SMA200 (slower market SMA) |

## Two findings

### 1. Hysteresis FIXES the bull flicker at every cadence (defect closed)
The raw BTC100 gate cut the 2024 bull at 1h (23.1 -> 14.1), 30m (15.8 -> 10.2), 15m (7.1 -> 3.8) by
flickering in/out as BTC chopped around its SMA100. **The 3% hysteresis band restores the bull to the
no-gate value at EVERY cadence** (1h 23.1, 30m 15.8, 15m 7.1 — all back to GATE_NONE). The latch (turn ON
only above SMA*1.03, OFF only below SMA*0.97) stops the whipsaw. So the gate's headline defect is solved.

### 2. The market gate's BEAR-softening is a 4h phenomenon (not a blanket win)
| cadence | GATE_NONE bear | BTC100_H bear | verdict |
|---|---|---|---|
| **4h** | -6.1 / -6.0 | **-2.6 / -2.9** | clean win: -57% bear damage, halved maxDD, rally/bull/reversal all preserved-or-better (reversal 7.2 -> 8.3) |
| 1h | -11.5 / -14.1 | -11.6 / -14.8 | flicker-safe NO-OP (gate neither helps nor hurts the bear) |
| 30m | -11.5 / -16.3 | -14.5 / -17.4 | HURTS the bear (latch holds through a fine-cadence bounce that resumes down) |
| 15m | -9.9 / -18.3 | -9.0 / -15.4 | small help on bear, helps reversal (1.9 -> 7.0) + comb (18.4 -> 21.3) |

At 4h, BTC's 100-bar hysteresis regime cleanly identifies the 2022 bear (BTC well below SMA100 -> latched
OFF -> assets sit out -> ~half the bear damage), at ZERO cost to rally/bull/reversal. At finer cadences
the month-long bear window has BTC oscillating enough that the latch stays partly ON, and the assets'
own trail/min-hold has often already exited — so the gate adds little (1h) or mistimes the sit-out (30m).

## The deployable complete spec (TRAIN-era)
**4h: FIXED 2MA-slow(60-150) + TRAIL(10%) + MAKER + BTC100-hysteresis gate.** (min_hold is a no-op at
4h.) Per-period: rally 2.7 / reversal 8.3 / combined 22.1 / **bear -2.6** / bull 23.3 — the bear softened
~57% with everything else preserved or improved. The hysteresis gate is the right form; raw BTC100 and
BTC200 both cost the bull or the rally.

## Honest limits (RWYB)
- The gate is validated as a **4h** instrument. It is flicker-safe (hysteresis) but NOT a bear-help at
  1h, and can HURT the bear at 30m. Do not apply it blanket across cadences.
- The bear is still NEGATIVE (-2.6). The market gate halves the long-only bear damage; it does not erase
  it. Money-making in a bear needs shorting (out of scope).
- All TRAIN-era / in-sample structural design. The market-gate edge in particular (one bear window) is
  the LEAST-sampled claim here -> it MUST be confirmed on VAL/OOS before any belief.

## Next
- The out-of-sample test: run the assembled 4h complete spec on VAL (2024-05-15..2025-03-15) and OOS
  (2025-03-15..2025-12-31) — does the structural stack (and the 4h market gate) transfer? UNSEEN stays
  sealed.

Chart: `../charts/complete_stack.png` (left: bear by gate x cadence — green hysteresis lifts the 4h bear,
mixed at finer; right: bull by gate x cadence — green restores the bull everywhere = flicker fixed).
