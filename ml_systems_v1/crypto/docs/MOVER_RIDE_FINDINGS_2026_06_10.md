# Mover-Ride at 1m + trigger-time meta-labeler — FINDINGS (2026-06-10, evening)

**User mandate:** find a tradeable u50 strategy (2x/yr per instrument), or solve capturing the
top daily movers, or dynamic weekly MA adaptation — "ride the trend, don't predict it; oracle
and TI frameworks side by side." This session ran the complete evidence chain on the one
genuinely open cell (D68's untested sub-hour pointer) using the new 1m data layer.

## The chain (three pre-registered studies, each RED-teamed or probe-verified)

### 1. The meat EXISTS at 1m — the bar-level "too late" mechanism (D67) is resolution-bound
`src/mining/mover_ride.py` decomposition (u10, 2021–2026, 16,369 asset-days analyzed):
- **24.4% of TRAIN asset-days have ≥5% open→high run-up** (42.9% have ≥3%) — the opportunity
  premise re-confirmed at scale.
- On those mover days, after a CAUSAL +1.5% close-cross trigger (fires at median minute 119 of
  the day), the post-fill oracle ceiling is **+5.7% median / +8.1% mean TRAIN (n=2,662); +5.3%
  median OOS (n=1,043)**. At bar-level the move was gone by confirmation; at 1m it is NOT.
- Prior P(mover5 | trigger fired) ≈ 0.38 — the trigger is information-rich as a *day filter*.

### 2. Unconditional riding bleeds — false positives, not entry-timing, kill it
Pre-registered 9-cell grid (T ∈ {1.0,1.5,2.5}% × trail ∈ {1,2,3}%), ALL cells, BOTH splits,
**negative absolute**: −35 to −73%/yr TRAIN, −36 to −88%/yr OOS at 24bps RT; win 35–41%;
breadth ≤2/10. Implementation brute-force verified (0/3,000 window mismatches; no same-bar
fills; splits clean). The RED-team caught my r1 null as hindsight-conditioned (~0.5pp/event
biased toward false KILL on a zero-edge martingale) — fixed to a post-trigger timing null;
the ABSOLUTE losses stand regardless. Artifact: `runs/mining/mover_ride_u10_20260610_193845.json`.

### 3. The discrimination test (the last internal-data angle) — NO held-out information
`src/mining/mover_metalabel.py`: ML as META-LABELER on the fixed trigger (the one
framework-endorsed ML use). 16 causal trigger-time features (run-up speed, overshoot,
prev-day, pre-vol, day-vol ratio, volume surge, aggressor imbalance, OI-deltas ×3, funding,
liq-ratio, regime, BTC-rel, BTC-24h, hour). HGB + logistic, TRAIN-fit (7,054 events),
single pre-registered operating point, OOS-once (3,369 events):
- **AUC: HGB 1.000 TRAIN → 0.521 OOS; logit 0.543 → 0.510.** Memorization, then coin-flip.
- OOS selected-third: −0.293%/event vs rejected −0.320%/event — no separation; breadth 3/10;
  all gates FALSE. Even with to-close exits (no trail whipsaw) the selected subset is
  −0.014%/event ≈ breakeven *before* any edge.
- u50 expansion NOT run, with reason: an information-less discriminator does not improve with
  more assets; expansion is justified only for a passing u10.
Artifact: `runs/mining/mover_metalabel_u10_20260610_195259.json`.

## VERDICT (D72, SCOPED): intraday mover-riding from internal data is closed
Whether a +1.5% intraday run-up continues to a ≥5% mover is **unpredictable with everything in
our data** — price/volume/flow/OI/funding/liq/regime/BTC-context, rule-based or learned, at 1m
resolution. The constraint is **INFORMATION, not resolution, cost, or execution**: D55
(direction unpredictability) extends to intraday continuation-given-onset. The capture target
the user named (25% of a 5% mover ≈ +1.25%/event; meat available at the trigger ≈ +5% median)
is arithmetically reachable ONLY with a discriminator; no internal feature set provides one.

## The 3 user asks — where each lands on the evidence
1. **2x/yr per instrument (u50, LO+spot+lev=1, timing):** not reachable on anything tested —
   bar-level exhausted (FIND_LO_WINNER 2026-06-10), sub-bar event-clock now tested and
   information-bound (this doc + D71). Honest per-instrument ceiling on internal data remains
   the regime-managed-beta class (~13–26% CAGR market fact, thread 22).
2. **Capture 25% of top movers:** oracle side QUANTIFIED (meat exists, trigger is early
   enough); causal side fails on discrimination — see D72. The arithmetic now pinpoints
   exactly what a solution must supply: a trigger-time signal with OOS AUC ≳ 0.58–0.60 on
   continuation. Internal data tops out at 0.52. → **external/leading data** (Coinglass
   pre-event heatmap, on-chain netflow, news/social momentum) is the only identified route;
   this is the A/B/C fork's Fork-B residual, now with a concrete spec + a ready harness +
   sealed UNSEEN sets to spend once.
3. **Weekly adaptive MA config:** the concurrent instance's weekly MA-oracle (in flight,
   `runs/oracle/jan_slices/`) already shows the modal best config CHANGING every week
   (SMA(5,50)→EMA(10,20)→SMA(5,100)→SMA(5,50), modal agreement 2–6/15 within-week) —
   consistent with the four prior nulls (D45; vol→config NULL; regime-switch NULL;
   secret-sauce NULL). Preliminary read: nothing stable to adapt to; await its completion
   before re-registering.

## What survives (durable)
- The 1m data layer + three audited harnesses (`liq_subbar`, `cascade_oracle`, `mover_ride`,
  `mover_metalabel`) — any future leading-data candidate plugs into a ready, RED-teamed
  evaluation rig with pre-registered gates.
- The meat-curve numbers (the capture TARGET spec for any external-data signal).
- The apparatus lessons: hindsight-conditioned nulls bias toward false KILL (r1 null bug);
  oracle ceilings inflate with post-event vol (D71 lesson); both now encoded.

Repro: `python -m mining.mover_ride --universe u10` ; `python -m mining.mover_metalabel
--universe u10` (seed 7; lineage in JSONs).
