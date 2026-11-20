# Regime-instrument search — can a better gate open the long-only sit-out door?

The keeper stack (FIXED 2MA-slow + TRAIL 10%) leaves the Jun-2022 bear NEGATIVE. For a long-only book the
only lever is a WORKING regime gate (short is out of scope). The raw SMA200 sit-out FAILED (too laggy at
4h, killed the young rally). So we searched the regime-instrument space, two-sided: does the gate cut the
BEAR/REVERSAL damage WITHOUT killing the RALLY/BULL? Tool: `src/strat/regime_gate_search.py`. Keeper
stack + gate, equal-weight u10 book, TAKER (isolate the gate), all TRAIN-era. ROI% / maxDD%.

## Gates tested
| gate | rule |
|---|---|
| G0_NONE | no gate (baseline = FIXED+TRAIL) |
| G_SMA200 | long only when close > SMA200 (the known failure) |
| G_SMA100 / G_SMA50 | faster self-regime |
| G_SLOPE100 | long only when SMA100 is RISING (10-bar slope) |
| G_HALF200 | HALF size below SMA200 (de-risk, not full sit-out) |
| **G_BTC100** | long only when **BTC > BTC.SMA100** (MARKET-wide regime) |

## The headline: gate on the MARKET (BTC), not the asset's own price
**Every SELF-referential gate failed** (G_SMA200/100/50, G_HALF200): they sit out the asset's OWN rally
(price below its own long SMA early in an up-move) and cost more rally upside than they save in the bear.
This is the same reason self-conditioned ML config-selection failed — a coin's own level is not a regime.

**G_BTC100 (the market-wide gate) is the only instrument that opens the door.** In the pooled two-sided
scatter it is the ONLY gate top-right of the no-gate baseline (better rally AND less-negative bear):

| period | G0_NONE 4h | **G_BTC100 4h** | G0_NONE 1h | G_BTC100 1h |
|---|---|---|---|---|
| rally | 1.5 / -0.7 | **2.8 / -0.4** | 10.3 / -2.5 | 10.9 / -1.9 |
| reversal | 7.3 / -13.3 | 6.4 / -13.3 | 6.8 / -15.0 | **8.7 / -8.3** |
| bear | -6.0 / -5.7 | **-2.7 / -2.8** | -12.5 / -14.8 | -12.5 / -15.0 |
| bull | 22.7 / -4.6 | **22.7 / -4.6** | 21.4 / -4.6 | **10.1 / -9.3** |

**At 4h, G_BTC100 cleanly opens the door:** the bear goes -6.0 -> **-2.7** (maxDD -5.7 -> -2.8, halved),
the rally IMPROVES (1.5 -> 2.8), and the bull is UNTOUCHED (22.7). That is the two-sided win the SMA gates
could not deliver — cut the bear damage ~55% with zero rally/bull cost.

## The one real cost: the 1h bull flicker
G_BTC100's only failure is the **2024 bull at 1h** (21.4 -> 10.1). In early 2024 BTC chopped right around
its 100-bar SMA, so the 1h gate flickered in/out and cut the bull gains. At 4h the SAME regime is smooth
(bull untouched). So the BTC gate wants a SMOOTH cadence (4h) or hysteresis -- a raw 1h close-vs-SMA100
crossing whipsaws the gate in a choppy bull. (This is the pooled "d_bull -5.6" that flagged it HARMFUL in
the strict per-pooled verdict -- the pooling hid that the cost is entirely the 1h-flicker cell.)

## Honest verdict
- **G_BTC100 at 4h is the first instrument to genuinely soften the long-only bear floor** (bear -6.0 ->
  -2.7) without a rally/bull tax. The regime signal that works is the MARKET's (BTC), not the asset's own.
- **It does NOT make the bear positive** (-2.7 is less-bad, not money-making). The long-only floor stands;
  the market gate halves the damage, it doesn't erase it. Positive-in-bear needs shorting (out of scope).
- **The 1h flicker is a real defect** -> the gate needs a smoother form (4h cadence, BTC.SMA200, or a
  hysteresis band) before it is trustworthy at fine cadence.
- TRAIN-era / in-sample. Confirm on VAL/OOS before belief.

## Next
- Add G_BTC100 (4h, smoothed) to the FULL keeper stack (+HOLD+MAKER) and confirm it survives + a
  hysteresis variant to kill the 1h flicker.
- This reframes the ML "trade-vs-sit-out" door: condition on the MARKET regime (BTC), not self-DNA --
  the one selection signal that showed two-sided value here.

Chart: `../charts/regime_gate_search.png` (left: two-sided scatter, G_BTC100 alone is top-right of
no-gate; right: gate x period bars -- BTC gate lifts rally/reversal/bear, only the bull bar dips from the
1h flicker).
