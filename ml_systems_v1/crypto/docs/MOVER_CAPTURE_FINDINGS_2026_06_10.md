# Mover-Capture — Findings (2026-06-09/10)

**Question (user):** *"Do you have a strat that captures the 25% of daily movers?"* then reframed: *"solve for 25% of
ORACLE moves (price-swing oracle + indicator oracle) — any move, any time, any cadence — not just the 25% of movers
on a particular day."* Tool: [`src/strat/mover_capture.py`](../src/strat/mover_capture.py). All RWYB on u50/u100,
taker 0.24% RT, LONG-ONLY, 50/20/20/10 split, UNSEEN once.

> **All figures VERIFIED (RWYB — measured from `mover_capture.py`).**

## The opportunity is real; the signal is not (the research's open question, now tested)
Market research: ~7.6 long-only up-movers/day ≥5%, broad, bursty, durable — but **raw moves ≠ edge** (random entry =
47% net-positive = coin-flip). The research left "is there a capturing signal?" UNTESTED. This lab tested it.

## 1. Cross-sectional SELECTION (predict which asset moves) — NULL at daily, regime-dependent intraday
SELECTED top-K-by-score vs RANDOM (x-sec mean) vs ORACLE top-K-by-forward, per window:
- **Daily**: all price scores (vol-expansion/breakout/momentum/range/accel), in BOTH continuation and reversal, are
  **null-to-negative on OOS** (extreme recent-movers underperform the calm middle = falling knives / blow-off tops).
- **Structural** (liq/funding/whale/basis): genuine *in-sample* power (capture 0.33, win 0.60) that **decays OOS**
  (win-rate inverts to 0.43). The liq-cascade-bounce is the only held-out-positive, but **sub-cost** (~0.1-0.2pp).
- **Learned ranker** (ridge + GBM on the full feature set): strong TRAIN edge (GBM +1.57pp), **~0 OOS** (overfit).
  DNA = forced-flow/positioning (liq_long_z30, fund_rate_z30, vol, oi_divergence) — real in-sample, non-stationary.
- **Cadence gradient (the one positive signal):** the selection edge GROWS finer — momentum OOS edge daily-negative →
  4h +1.48pp → **1h +5.59pp** — i.e. mover-capture is structurally an *intraday* problem (you must enter *during* the
  move). **But it does NOT hold on UNSEEN** (regime-dependent, not robust).

## 2. The reframe (capture % of the oracle move) — faithfully tested
- **The perfect-swing oracle is unrealizable.** A perfect swing-trader catching *every* ≥2% wiggle compounds to
  **trillions %**; "25% of that" is impossible (capture ≈ 0). (It also exposed a real bug: `_zigzag_pivots` was
  collapsing to [start,end] = buy&hold — FIXED 2026-06-10.) The right metric is **per-move capture**, not
  compound-vs-perfect-swing.
- **Per-move capture = NEGATIVE at every cadence and entry.** For each individual 2-50% move, fraction the strat
  banked (missed move = 0):

  | entry | cadence | coverage | per-move capture (when in a move) | moves captured ≥25% |
  |---|---|---|---|---|
  | breakout | 1d | 9% | −13% to −27% | 1-3% |
  | breakout | 4h | 10% | −8% to −24% | 2-3% |
  | breakout | 1h | 9% | −10% to −24% | 1.6-2.8% |
  | dip-buy-in-uptrend | 1d/4h | 4-9% | **−71% to −86%** | 1-3% |

  The realizable strat enters <1/10 of moves and **loses on the ones it enters.** Structural reason: a causal entry
  is either **too late** (breakout — by confirmation the move is mostly done, you bank the give-back) or **fails**
  (dip-buy — the uptrend ends and the trailing stop knifes out). Capturing a move needs entry at its *start* = a
  prediction = null (§1). The user's "price near the long-term MA" zone tested as the dip-buy = the *worst* (it's a
  low-vol equilibrium, not a launchpad — matches the earlier `entry_zones` finding).

## 3. The only realizable edge is capital-PRESERVATION, not move-capture
vs buy&hold, the ignition+convex strat **beats hold on held-out** (OOS +28-31pp, 77% of assets) — but by *avoiding
the alt crashes* (trend-follower cuts losers), which is the **same risk-managed-beta** finding as
[ENTRY_SIGNAL_CAPTURE_FINDINGS](ENTRY_SIGNAL_CAPTURE_FINDINGS_2026_06_09.md), restated. It is risk-management, not
mover-capture.

## Verdict + the genuine path
**No — there is no causal mover-capture strat at 1d/4h/1h.** Selection is null/regime-dependent; per-move capture is
negative for every entry/cadence tested; the only realizable edge is capital-preservation. The reframe was genuinely
better — it gave a faithful, bounded metric that proved the point cleanly — and the metric's verdict is robust.

**Where a profitable mover-capture could still live (all require relaxing a constraint — the A/B/C fork):**
1. **Sub-hour / tick** — the cadence gradient (selection edge grows finer) says the breakout is "early" enough only at
   very fine resolution; but there taker cost is fatal → needs **maker execution** (Fork B).
2. **Cascade event-clock + leading data** (Coinglass liq heatmap) — the structural DNA (forced-flow) had real
   in-sample power; the research says it needs a sub-4h event-clock + pre-event data (Fork B).
3. **The indicator-oracle** (best-TI-config per move) — untested here; prior config-selection work found
   selection ≈ random OOS, so likely null, but it is the unbuilt half of the reframe.

*Provenance: /orc autonomous run, anchor 2026-06-09 21:44 SAST. RWYB from `mover_capture.py`
(`--selftest --learned --capture --permove --entry-kind`).*
