# Old vs new + residual weaknesses + family consistency — TRAIN/2020/JAN

Tool: `src/strat/ma_compare.py` (40 distinct 2MA + 40 3MA × 7 assets × {baseline, min-hold12} × 4
cadences, rich metrics: net, maxDD, %-bars-in-profit, trades; oldest month, taker).

## [A] OLD (baseline) vs NEW (min-hold 12) on the WEAK cells
The fix rescues exactly where the weakness was worst:

| cadence | speed | OLD net | NEW net | Δ | OLD %pos | NEW %pos |
|---|---|---|---|---|---|---|
| 30m | fast(<20) | −11.6% | −1.1% | **+10.5** | 22% | 40% |
| 30m | mid(20-60) | +6.6% | **+9.5%** | +2.9 | 73% | 84% |
| 15m | fast(<20) | −23.0% | −9.9% | **+13.1** | 5% | 18% |
| 15m | mid(20-60) | −2.2% | **+0.5%** | +2.7 | 44% | 55% |

min-hold lifts the failing cells the most (15m-fast +13pp), and turns the mid-MA cells net-positive.

## [B] Residual / OTHER weaknesses
- **B1 — the residual killer: fast MA on fine cadence is STILL a loser.** 15m-fast −9.9%, 30m-fast
  −1.1% even *after* min-hold. The fix rescues them ~halfway but cannot make the *fastest* MAs profitable
  at the *finest* cadence — those need a slower MA, not just a discipline overlay. (The fastest MA on the
  finest cadence is the one structural combination the overlay can't save.)
- **B2 — min-hold did NOT hurt any bucket** (no negative delta) in this period — it's a free improvement here.
- **B3 — no drawdown side-effect** (this regime): avg maxDD −11.4% → −10.6% (slightly *better*). In a
  rally, holding longer didn't deepen DD. (REGIME CAVEAT: in chop/bear, forcing the hold could deepen DD —
  re-test per period.)
- **B4 — "sustained" is only moderate WITHIN the month:** even the best family sits in profit only ~47% of
  bars. The month dipped early then rallied, so configs were underwater early and made the gains late —
  consistent at the *end* (91% positive), back-loaded *through* the month.
- **B5 — XRP is the weakest instrument** (+6.6%, 64% pos after fix) — a weak-mover effect, not a killer
  (everything else is +5% to +19%).

## [C] FAMILY CONSISTENCY — 2MA beats 3MA; SLOW is the sweet spot
mean net / std / %positive / avg-maxDD over the month (baseline):

| family · speed | mean | std | %pos | maxDD |
|---|---|---|---|---|
| **2MA · slow(60-150)** | **+14.3%** | 11.4 | **91%** | −10.0% |
| 2MA · mid(20-60) | +12.7% | 11.2 | 85% | −12.1% |
| 2MA · vslow(≥150) | +13.9% | 12.9 | 76% | −7.2% |
| 3MA · slow(60-150) | +7.2% | 10.2 | 75% | −10.2% |
| 3MA · mid(20-60) | +3.7% | 13.4 | 63% | −13.6% |
| 2MA/3MA · fast(<20) | −2 / −9% | ~19 | 49/37% | −19/−21% |

- **The sustained + consistent family = 2MA · slow(60-150): 91% of configs positive, +14.3% mean,
  shallow −10% drawdown.** It's the up-and-left point on the consistency scatter.
- **2MA beats 3MA at every speed** (2MA-slow 91% pos vs 3MA-slow 75%; 2MA-mid 85% vs 3MA-mid 63%).
  Reason: 3MA needs three MAs to ALIGN (fast>mid>slow) — a stricter, more fragile condition that breaks on
  noise; the simple 2-MA cross is more robust.
- **fast MA is the killer for BOTH families** (49/37% pos, deep DD) — the consistent loser.

**Takeaway:** the reliable strat family this month is the **2-MA cross with a slow (60-150 bar) lookback**,
cadence-matched, optionally with the min-hold overlay to clean up the fine-cadence residual. 3MA adds
fragility, not consistency. Re-test the family ranking + the min-hold fix on a bear month next.

Repro: `python -m strat.ma_compare`.
