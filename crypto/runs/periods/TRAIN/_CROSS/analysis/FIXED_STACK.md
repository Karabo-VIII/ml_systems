# Keeper stack, isolated, across ALL cadences (4h / 1h / 30m / 15m)

Continuation of the fixed-approach ladder. The ladder showed the keepers are **FIXED (2MA-slow family)
+ TRAIL(10%)**, that min-hold is a no-op at coarse cadence, and that the SMA200 regime gate is harmful.
This isolates the keepers WITHOUT the harmful regime gate and EXTENDS to the FINE cadences (30m/15m)
where min-hold + maker were predicted to pay. Tool: `src/strat/fixed_stack.py`. Equal-weight u10 book,
causal MtM net daily returns, all TRAIN-era. ROI% / maxDD% / Sharpe.

**Optimization validated:** a warmup-bounded slice ([start-600bars, end] instead of full history) makes
fine cadences tractable AND reproduces the committed full-history L1_FIXED numbers bit-for-bit
(FIXED@4h: combined 17.3, bull 24.5, bear -9.9, Feb 2.0 -- identical). The optimization is free.

## Combined Jan+Feb 2020 ROI% by cadence x variant
| variant | 4h | 1h | 30m | 15m |
|---|---|---|---|---|
| NAIVE (all 120 cfg) | 11.9 | 16.3 | 7.1 | **-7.4** |
| FIXED (2MA-slow) | 17.3 | 29.0 | 17.8 | 11.2 |
| FIXED+TRAIL(10%) | 21.9 | 29.6 | 17.5 | 10.8 |
| FIXED+TRAIL+HOLD(12) | 21.8 | 29.5 | 19.2 | 12.6 |
| **FIXED+TRAIL+HOLD+MAKER** | **22.1** | **31.1** | **22.5** | **18.4** |

## The fine-cadence recovery (the key result)
**15m goes from a -7.4% NAIVE loser to a +18.4% winner under the full stack.** Decomposing the 15m
combined ROI gain:
- FIXED family choice: -7.4 -> +11.2  (**+18.6pp** -- the family does most of the work)
- + min_hold(12):      +10.8 -> +12.6 (**+1.8pp** -- min-hold finally BINDS at fine cadence, as predicted;
  it was a no-op at 4h/1h)
- + maker fees:        +12.6 -> +18.4 (**+5.8pp** -- maker is a fine-cadence lever, ~0 at 4h)

30m tells the same story (NAIVE 7.1 -> stack 22.5; min-hold +1.7pp, maker +3.3pp). At 4h/1h min-hold
and maker add little (the slow family already holds ~1 week; few trades to re-price) -- they are
explicitly FINE-cadence levers. This is exactly the building-block prediction realized at book level.

## The long-only floor (the bear stays negative)
| variant | 4h | 1h | 30m | 15m |
|---|---|---|---|---|
| NAIVE | -7.1 | -11.7 | -15.4 | -20.3 |
| FIXED | -9.9 | -13.0 | -13.8 | -14.3 |
| FIXED+TRAIL+HOLD+MAKER | **-6.1** | **-11.5** | **-11.5** | **-9.9** |

The full stack is the least-bad in the Jun-2022 bear at every cadence, and the trail + maker meaningfully
cut the fine-cadence bleed (15m -20.3 -> -9.9). **But every cell is still NEGATIVE.** A long-only book
cannot make a one-way downtrend positive by tuning entries/exits/costs -- it can only lose less. Making
the bear non-negative requires either staying in cash (a WORKING regime gate -- the raw SMA200 one
failed) or shorting (out of scope: long-only + spot + lev=1). This is a structural ceiling, not a miss.

## Verdict
- **The winning stack is FIXED + TRAIL(10%) + HOLD(12) + MAKER**, and it is best/near-best in nearly
  every (cadence, period) cell. With the full stack the cadences converge (4h 22.1 / 1h 31.1 / 30m 22.5
  / 15m 18.4 combined) -- the fine-cadence cost-drag gap is CLOSED by family + hold + maker. 1h is the
  sweet spot.
- **At TAKER (no maker assumption), the stack still rescues the fine cadences** (15m FIXED+TRAIL+HOLD
  +12.6 combined, vs NAIVE -7.4) -- so the recovery is NOT purely a maker-fill bet; family+hold do the
  heavy lifting, maker is the cherry (and a CEILING, since real p_fill is 0.21-0.40).

## Honest caveats (RWYB)
- All five periods are TRAIN-era -> in-sample structural design. Confirm on VAL/OOS before belief.
- Maker assumes fills; at 15m/30m the +5.8/+3.3pp maker lift is a ceiling, not a floor.
- The bear floor is the open door -> next: a REGIME-INSTRUMENT search (can a better gate than raw SMA200
  open the sit-out door without killing the rally?).

Chart: `../charts/fixed_stack.png` (left: combined ROI by cadence -- NAIVE worst, stack lifts 15m out of
the hole; right: the bear by cadence -- stack least-bad but every cell negative = the long-only floor).
