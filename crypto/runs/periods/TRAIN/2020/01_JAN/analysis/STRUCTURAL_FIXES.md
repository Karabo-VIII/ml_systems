# Structural fixes per timeframe — TRAIN/2020/JAN

**Question (user):** identify structural gaps, solve them early, then see if performance improves —
mine structural solutions to the per-timeframe MA failures (the killer = over-trading via MA/cadence
mismatch → cost drag; whipsaw ~10-16%). Tool: `src/strat/structural_fixes.py` (4,480 cells: 32 configs
× 7 assets × 5 overlays, oldest month, taker).

## The overlays tested (discipline layers on the raw MA signal)
- **cooldown(N)** — after an exit, block re-entry for N bars (the user's idea)
- **min_hold(M)** — after an entry, hold ≥ M bars before any exit
- **confirm(K)** — only enter if the signal held K consecutive bars (debounce the cross)
- **cool+conf** — the combination

## Result: MIN-HOLD wins at every cadence, most where it's needed

mean net % (baseline → best overlay), Δ vs baseline:

| cadence | baseline | cooldown6 | **min_hold12** | confirm3 | best Δ |
|---|---|---|---|---|---|
| 4h | 9.6 | 9.6 | **10.8** | 8.6 | min_hold **+1.2** |
| 1h | 12.9 | 14.1 | **15.7** | 13.3 | min_hold **+2.7** |
| 30m | 7.8 | 8.8 | **12.0** | 8.2 | min_hold **+4.2** |
| 15m | 0.7 | 2.2 | **5.4** | 2.2 | min_hold **+4.7** |

- **min_hold(12) is the dominant structural fix** — it lifts net at *every* cadence and the lift GROWS
  toward the failing cadences (15m +4.7pp: 0.7 → 5.4%).
- It works by **eliminating whipsaw**: whipsaw count 12.5 → 0.1 at 15m (a 12-bar floor makes a ≤2-bar
  whipsaw impossible), and trades drop 46 → 30.
- **The cooldown (user's idea) is the runner-up** (+1.4pp at 15m): directionally right (it halves
  whipsaw 12.5 → 6.1), but weaker than min_hold because it only blocks *re-entry*, not the early *exit*.
- **confirm/debounce** barely helps and even hurts at 4h (it delays good entries).

## The big one: min_hold ≈ maker, for free
min_hold(12) at 15m **taker** = +5.4% — essentially the same as what **maker fees** gave (+5.4%). So a
*structural overlay you control* recovers the fine-cadence edge **without needing maker fills** (whose
p_fill 0.21-0.40 is unproven). The whipsaw is the leak; min_hold plugs it directly.

## Honest caveat (why the period store matters)
This is the TRAIN/2020/JAN **rally** regime. A minimum-hold helps in a trend (whipsaw-prevention >
the cost of holding through a small reversal). In a **chop or bear** month, forcing a 12-bar hold could
*hurt* (you hold losers longer). The structural fix must be re-tested per period — exactly what the
period-keyed storage is for. Next: run the same overlay test on a 2022 bear month before adopting it.

Repro: `python -m strat.structural_fixes`.
