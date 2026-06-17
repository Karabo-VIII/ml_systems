# FINDING — the daily-mover problem, decomposed into 4 and rigorously solved (2026-06-13)

> A 4-hour sprint on the user's framing: "capture the daily movers" is FOUR problems the project had
> conflated — (1) SELECTION (predict which asset moves ex-ante), (2) TIMING (catch the start),
> (3) CONTINUATION (once moving, will it persist), (4) CAPTURE-RATE (of a move you're in, what fraction).
> A dedicated rigorous agent attacked each with FRESH angles the dead-list never tried (asymmetric/convex
> exits, dollar-neutral hedge, onset-flow state, fizzle-filter, tail-precision, the magnitude lens).
> All held-out, RWYB, two-sided, leak-audited. Scripts: `src/mining/{daily_movers_profile,
> mover_capture_rate, mover_continuation, magnitude_selection, mover_burst_timing}.py`.

## The convergent answer (NOT "all 4 are dead")
**Every DIRECTIONAL attack on the movers is information-bound dead — confirmed four ways with the
freshest angles. But the orthogonal MAGNITUDE dimension is ALIVE, with TWO independent confirmed
signals, and a single clear monetization (a both-sided / straddle structure, which needs options).**
The user's intuition was right: the four DIRECTIONAL framings are dead; the MAGNITUDE axis is not.

## The 4 lanes (held-out numbers)
| # | Problem | Directional verdict | The magnitude / orthogonal truth |
|---|---|---|---|
| 1 | SELECTION (which name, ex-ante) | weak: ~95% a STATIC vol-tier ranking (small/young coins move more, already in vol-targeting); dynamic edge ~+0.05 rank-IC = vol-persistence | ex-ante magnitude-pick is NOT a strong cross-sectional alpha |
| 2 | TIMING (catch the start) | burst onset DETECTABLE (AUC 0.885) but COINCIDENT not leading (median lead 20/30min — the burst is already underway); fizzle-filter lifts a directional ride only to BREAK-EVEN gross, still sub-cost | **fizzle-filter WORKS: OOS AUC 0.70, conversion 31%->50%(top30)->58%(top10), breadth 10/10 — it identifies big-\|move\| names; it predicts MAGNITUDE not sign** |
| 3 | CONTINUATION (will it persist) | DIRECTIONAL dead: OOS AUC 0.506 = D72's 0.52 exactly; fresh onset-flow/OI-build-vs-liq-wick features refuted; tail-precision economically negative | **MAGNITUDE-continuation ALIVE: OOS AUC 0.737, robust 10/10, mech-only 0.66 (not a vol-tautology), beats shuffled null** |
| 4 | CAPTURE-RATE (fraction realized) | entry-bound: only 31% of triggers convert; ride net -0.09% OOS; EVERY asymmetric/convex/neutral exit monotonically WORSE than dumb hold; bleed is ENTRY not EXIT | causal capture ~25-30% gross / ~0 net; the "67% capture" is a hindsight mirage (conditions on runup>=5%, a day-end fact) |

## Two load-bearing methodology catches (the sprint's apparatus value)
1. **The hindsight-cohort trap:** any "mover capture %" measured on the `runup>=5%` cohort silently
   conditions on the future and manufactures a fake 50-80% capture. The ONLY honest cohort is the
   CAUSAL all-trigger set (the trigger can't see the day's outcome). `mover_capture_rate.py` reports both.
2. **The magnitude-vs-direction separation:** signals that "clear AUC 0.58" repeatedly turn out to
   select BIGGER moves regardless of sign (a magnitude/vol-cluster property), with the selected cohort's
   directional return a coin-flip. The fizzle-filter and the A2 continuation variant both do this.

## The monetization (the open go/no-go)
The magnitude edge (continuation 0.73 + fizzle-filter conversion 58%) implies BUYING magnitude on the
filtered cohort -- i.e. a STRADDLE. Perp-straddles are a proven trap (D64/D65: no gamma, cost). So the
vehicle is OPTIONS, which the project lacks (Deribit ingest ~$700/mo). The DECISIVE data-in-hand test
(running now, `mover_straddle_ev.py`): does the filtered cohort's REALIZED magnitude exceed a fair
option premium (VRP usually favors SELLERS -- IV>RV ~71% of days), i.e. is BUYING magnitude +EV on the
movers, or only SELLING VRP on the calm names? That answers whether the options ingest is justified.

## Honest ceiling
Directional daily-mover capture is dead (now exhaustively, with the asymmetric/neutral/filter space
mapped). The movers' MAGNITUDE is capturable in principle and the SELECTION of big-|move| names works
(AUC 0.70) -- the realizable path runs through options, gated on the straddle-EV test + a data decision.
This is a DATA/INSTRUMENT decision, not a research-effort one.
