# Mover-capture INDEPENDENT VERIFICATION (2026-06-09) — NULL on held-out

**Question (user):** *"Do you have a strat that captures the 25% of daily movers?"* The other instance built
`src/strat/mover_capture.py` and declared done but persisted **no held-out result**. This is the independent
re-run (overseer, parallel instance) on real `u100`, H=3, K=5, 42 assets, long-only, net taker 0.24%.

## The verdict: NULL — no selection score captures the movers on unseen data
The ORACLE ceiling is huge and real (top-K by *forward* return = **7.6% TRAIN → 15.1%/3d UNSEEN**). But every
realizable past-only selection score captures **~0%** of it and is a coin flip vs random selection:

| score | UNSEEN SEL% | UNSEEN RND% | edge_pp | capt-vs-oracle | >rnd (days) | OOS edge_pp |
|---|---|---|---|---|---|---|
| vol_expansion | −0.034 | −0.310 | +0.276 | −0.002 | 0.48 | **−0.083** |
| breakout | −0.462 | −0.310 | −0.151 | −0.031 | 0.46 | −0.444 |
| momentum | −0.263 | −0.310 | +0.048 | −0.017 | 0.47 | −0.437 |
| range_surge | +0.091 | −0.310 | +0.401 | +0.006 | 0.51 | −0.213 |
| accel | −0.284 | −0.310 | +0.026 | −0.019 | 0.44 | −0.491 |

**Reading:** (1) `capt-vs-oracle ≈ 0` everywhere — the signals capture essentially none of the 15%/3d available
move. (2) `>rnd ≈ 0.44–0.51` — beating random selection is a coin flip. (3) the edge-vs-random **flips sign**
between OOS (all negative) and UNSEEN (tiny positive) = noise, not signal. (4) SEL% itself is mostly **negative**
on held-out — the picked "movers" *lose money* after cost.

## Conclusion
**You cannot capture the 25% daily movers by PICKING them ahead of time** at daily resolution with these scores —
selection power is null (consistent with D55 direction-unpredictable + "naive harvest = coin-flip"). The movers are
real and large, but **uncapturable by past-only cross-sectional selection**. This confirms the user's skepticism
that the other instance's "done" was not a profitable solution.

This converges with the parallel entry-timing finding (entry/selection is fungible/null) → the honest capture, if
any, comes from the **EXIT (looseness/convexity) + regime/trend participation** (managed-futures), NOT from
selecting the movers. Open neighbors (not yet refuted): momentum-CONTINUATION-with-loose-exit (ride, don't pick),
finer-than-daily cadence, maker execution (won't create selection skill — edge is coin-flip — but shifts the level).

Raw output: `runs/strat/mover_capture_u100_h3_VERIFY_2026-06-09.txt`. Evaluator soundness: `mover_capture.py
--selftest` PASS (detects a synthetic cross-sectional edge), so the NULL is a real-data result, not a broken gate.
