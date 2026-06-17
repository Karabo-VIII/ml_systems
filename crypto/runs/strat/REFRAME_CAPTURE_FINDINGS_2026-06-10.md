# The capture reframe, tested — convergent NULL at 1d→15m (2026-06-10)

**User reframe (2026-06-09):** stop solving "predict the 25% of daily MOVERS" (verified null); instead solve
"capture ≥25% of the ORACLE move" — any 2-10% slice, any cadence (incl. intraday), with two oracles: PRICE-oracle
(perfect entry/exit ceiling) and INDICATOR-oracle (best causal-TI-family ceiling). Unit = a SETUP across a MOVE.

## Is the reframe better? YES (lens) — but it does NOT change the answer at 1d→15m (empirics)
The reframe is the *right lens*: capture-rate of moves, not daily selection; the project's founding unit made
concrete. I adopted it and tested it properly (`src/strat/oracle_capture_lab.py`, selftest PASS — a genuine
continuation signal captures 46% of a synthetic oracle move, beats random 28%). The result, on real data:

**Capture-after-cost of a causal momentum-continuation signal, UNSEEN, by cadence (price-oracle):**
| cadence | taker | maker | moves | trades | clears 25%? |
|---|---|---|---|---|---|
| 1d | −0.052 | −0.049 | 22–25 | 1–3 | no |
| 4h | −0.068 | −0.042 | 72–84 | 28–36 | no |
| 1h | +0.013 | +0.051 | 116–153 | 93–115 | no |
| 15m | −0.009 | +0.056 | 108–156 | 140–192 | no |
| dollar | +0.418 | +0.687 | **1–3** | 3–9 | **VOID (small-sample artifact)** |

**The cost-cliff dominates.** At every cadence with a real sample (1d/4h/1h/15m), capture-after-cost is pinned near
zero (−7% to +5.6%) — nowhere near 25%. As cadence gets finer the oracle move shrinks but cost stays fixed, so net
capture stays ~0. The only "clearing" cells are dollar bars, and they are statistically void: 1–3 moves on a
~3-week UNSEEN, a mis-scaled 64-bar window. TRAIN+VAL capture looks healthy (0.37–0.73) but UNSEEN collapses to ~0
— the standard overfit-collapse the discipline catches.

## Four convergent experiments — all NULL for active capture at 1d→15m spot/LO
1. **Cross-sectional daily mover-SELECTION** (`mover_capture`, u100, 5 scores): coin-flip vs random, capt-vs-oracle ~0, net-negative after cost.
2. **Multi-MA-alignment time-SELECTION** (`selection_signal_lab`, u10 1d): beats selection-null 0/10 on UNSEEN; "beats buy&hold 10/10" is **bear-abstention** (UNSEEN was a −20 to −51% drawdown; the signal "wins" by sitting flat) → **beats BOTH = 0/10** (the wealth-add verdict).
3. **Entry-TIMING within a move** (parallel instance): fungible (random-within-move ≈ the trigger).
4. **Oracle capture-rate across cadences** (this run): ~0 after cost, 1d→15m; cost-cliff dominates.

**The honest convergent finding:** the moves are real and large (oracle 15%/3d; 2–10% slices everywhere), but a
causal signal captures **~0% of them after cost at every cadence from 1d to 15m, spot/LO/fixed-size.** The apparent
"wins" are all bear-abstention / regime-timing = drawdown-avoidance (beta-timing), NOT capture-alpha. This is not
defeatism — it is four independent, sound-evaluator (selftests PASS), held-out experiments converging.

## Sub-bar done PROPERLY (2026-06-10 closure) — the null EXTENDS to sub-bar
Re-ran dollar + dib with a dollar-appropriate move-window (360 ≈ 4h of action) on the FULL series, **222–253 UNSEEN
moves** (statistically firm):
- **Dollar = clean NULL**: taker capture **−0.43 to −0.47**; maker ~0 (−0.02 to −0.06), all 3 assets. The cost-cliff
  dominates at ~40s bars too — the 1d→15m null extends to sub-bar.
- **Dib = BTC-only flicker**: BTC dib clears 25% (taker +0.265 / maker +0.362, 99 moves) but **ETH dib is null
  (+0.063 taker)** and SOL dib has no data. A single-asset positive that does NOT replicate = a candidate to probe
  (get SOL dib + seed-robustness), **not an edge**. Most likely a BTC bar-frequency artifact.

**So the Fork-B (sub-bar) frontier does NOT overturn the null for momentum-continuation.** Five convergent
experiments now. The one remaining flicker (BTC-dib) is single-asset + non-replicating.

## What's genuinely still open (not refuted)
1. **The BTC-dib flicker** — probe with SOL-dib data + seed-robustness + block-bootstrap; expected to be noise.
2. **The convexity/exit path** (ride trends with loose exits) — the other instance's lever; gives beta+trend+convexity
   (compound return), NOT entry-capture alpha; OOS-robustness (regime-dependent exit) is the open battle.
2. **Maker execution properly** — the 1h maker cell (+0.05 to +0.13) is the least-null; maker is the cost lever, but
   p_fill 0.21–0.40 makes real maker worse than the 0.06% RT assumed. Won't create capture skill (the edge is ~0),
   but shifts the level.

## Implication
The reframe sharpens — does not move — the **A/B/C fork**: daily-through-15m spot/LO is null for active capture
(now 4-ways confirmed); the only unrefuted ground is sub-bar/tick (Fork B), which is a real data-engineering + cost
commitment. The honest expectation there remains guarded (the cost-cliff is the same wall, just finer).

Artifacts: `oracle_capture_BTC_ETH_SOL__1d_4h_1h_15m_dollar.json`, `selection_lab_u10_1d_2026-06-10.txt`,
`MOVER_CAPTURE_VERIFY_2026-06-09.md`. All evaluators selftest-PASS (the nulls are real, not broken gates).
