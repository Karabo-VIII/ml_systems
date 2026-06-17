# Combined Jan+Feb 2020 — the two-month scenario (strengths & weaknesses)

One continuous run, 2020-01-07 → 2020-03-07 (equity compounds through the Jan **rally**, the ~Feb-19
**top**, and the **reversal**/COVID-onset). Split at 2020-02-07 to decompose each cell's ROI into its
JAN-half and FEB-half. Tool: `src/strat/combined_analysis.py` (24 cfg/fam × 7 assets × 4 cadences ×
4 exits = 5,376 cells, taker). Numbers below are CROSS-ASSET MEANS.

## Combined ROI by cadence (signal-flip)
| cadence | combined | jan-half | feb-half | maxDD | %pos |
|---|---|---|---|---|---|
| 4h | +11.5% | +10.2 | **+1.0** | −17.6% | 60% |
| 1h | +15.8% | +12.6 | **+2.8** | −19.7% | 78% |
| 30m | +7.5% | +7.6 | **−1.2** | −23.4% | 66% |
| 15m | **−7.4%** | −0.0 | **−9.8** | −30.0% | 42% |

## STRENGTHS
1. **Slow 2MA cross is robust over the full round-trip.** 2MA·slow(60-150): **+19.1% combined, 82%
   positive, and the ONLY bucket clearly positive in BOTH halves** (jan +15.2%, **feb +3.5%**) — it held
   the rally gains AND made more in the reversal. Best individual configs compounded +32-36% cross-asset
   (up to +80-89% on the best asset) over the two months.
2. **Coarse cadences survive the reversal.** 4h and 1h stayed net-positive *through Feb* (feb-half
   +1.0% / +2.8%). They keep what they make.
3. **min-hold is the best discipline over the span** — best pooled combined ROI (+9.4%) and the least
   give-back in Feb (−1.4%), by trading less and dodging the Feb chop's cost bleed.
4. **2MA beats 3MA** over the combined span too (2MA-slow +19.1% vs 3MA-slow +8.7%).

## WEAKNESSES
1. **15m does not survive the round-trip** — combined −3% to −9%; the reversal compounds the cost bleed.
   Even min-hold only reaches −3.1%. Fine cadences are a 2-month loser at taker.
2. **The reversal gives back gains at 30m/15m** (feb-half negative). They can't hold what they earn.
3. **SURPRISE — trailing stops did NOT protect gains; the tight trail-5% was the WORST.** trail5 pooled
   combined +3.0% (vs signalflip +6.9%, minhold +9.4%) and gave back −3.5% in Feb. In a **choppy top**
   (not a clean one-way crash) the trail whipsaws out, pays the cost, and misses the continuation.
   **Cutting fast LOST to trading less.** (CAVEAT: a clean sustained crash would likely favor the trail;
   this is the rally→choppy-top regime — re-test the exit ranking in a true bear.)
4. **Drawdown is material over a multi-month span** — combined maxDD −17% to −30%; even the best family
   draws −21%. Multi-month = where position-level risk control earns its keep.
5. **Fast MAs are unsalvageable** over the span (deeply negative both halves, −10% to −17%).

## VERDICT
Over a realistic two-month span WITH a regime change, the robust configuration is **slow 2MA cross +
coarse cadence (4h/1h) + min-hold discipline** — it compounds through the rally and holds through the
reversal (+19% combined, positive both halves, 82% breadth). Fine cadences and fast MAs do not survive
the round-trip. The key exit lesson: **in a choppy top, trading less (min-hold) beats cutting fast
(trail)** — the opposite of the usual "use a stop to protect gains," because the chop whipsaws the stop.

Next: a true 2022 bear month to test whether the exit ranking flips (does the trail finally win in a
clean downtrend?) and whether 2MA-slow holds.
