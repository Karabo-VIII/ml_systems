# Plain rig — ER-gated FIXED-MA baseline (4h, strictly 3 DOF) — REPORT

**Date:** 2026-06-05 · **Cadence:** 4h · **Universe:** u100 (77 assets, 0 skipped) · **Cost:** taker 0.0024 (src/strat/fill_model) · **No commit.**

## What was built (per RESEARCHER_REPORT_1 redirect)
The minimal honest null the 6-cell adaptive map must beat *before* any cell is added. Strictly **3 DOF**:

| DOF | Choice |
|---|---|
| 1 — ER hard-gate | `ER > 0.40` (raw Kaufman ER, **not** the per-asset 252-bar percentile — that boundary was the flagged DOF risk). Trade only when trending; SKIP chop. |
| 2 — fixed MA | one config: `SMA(10)/SMA(30)` |
| 3 — exit | one policy: ATR-trailing stop (`3×ATR14`) + time-stop (`42 bars = 7d`) |

**Entry (setup, confirmed at close of bar t):** `ER>0.40 AND close>max(prior-20 highs) AND fast>slow` — all past-only; SetupHarness fills at `opens[t+1]`.
Structural constants fixed up-front (not tuned, not free DOF): `ER_WIN=20`, `ATR_WIN=14`, `BREAKOUT_N=20`.

**Files (runnable):**
- `ergated_fixed_ma_4h.py` — 3-DOF core (causal ER + breakout + SMA + ATR; reuses `wealth_bot.harness.sma_past_only`; mirrors the expert rig's Kaufman ER). `--selftest` (synthetic two-sided gate check, PASS) and BTC RWYB (+ leak guard + causality proof).
- `run_u100_4h.py` — u100 runner → `RESULTS_4h.md`, `results_u100_4h.json`. Exit via `src/strat/setup_harness.SetupHarness` + `ExitPolicy`.

## Result — REFUTED on held-out (OOS+UNSEEN)
Produced by `python run_u100_4h.py` (RWYB):

| metric | value |
|---|---|
| **Pooled held-out per-trade NET expectancy** | **−2.22%** (n=1098, winrate 27%, median −3.14%, p05 −12.78%, p95 +11.42%) |
| **Pooled UNSEEN-only per-trade NET expectancy** | **−1.81%** (n=347, winrate 30.5%) |
| **Assets with positive held-out expectancy** | **11 / 77** (median asset −2.08%) |
| BTC 4h (all windows) | TRAIN −47.8% / VAL −4.4% / OOS −15.1% / UNSEEN −6.0% compound |

The expectancy is **negative**, nowhere near the 2–5%/trade target. The 11 positive assets are the thin upper tail of a left-skewed distribution (median asset −2.08%), consistent with noise, not a robust per-asset edge. Adding map cells on top of this is fitting noise — **the timing premise fails on held-out at this minimal config**.

## Look-ahead audit (two findings; verified against code)
1. **`SetupHarness.leak_guard()` fires `LEAK_SUSPECT` on the positive-edge assets** (WLD/ICP/RENDER/SUI, ratio 0.07–0.82) and `INSUFFICIENT_EDGE` on the negative bulk.
2. **The strategy is provably past-only — no leak.** `causality_selfcheck` (prefix-truncation): `entry[t]` and `atr[t]` recomputed from `df[:t+1]` *exactly* equal the full-series values at `t` — **0 entry mismatches, 0.0 ATR diff** across all sampled points on WLD/ICP/BTC. Construction uses only backward `rolling`/`shift` (no `.shift(-k)`); fill is `opens[t+1]`.

**Reconciliation:** the `LEAK_SUSPECT` label is a **false positive** of the relative lead/lag *heuristic*, not a real leak. On these assets the positive held-out compound comes from a handful of trades (noise), so injecting one bar of the future barely improves the result — which the ratio test misreads as "the entry already encodes the future." The hard prefix-truncation proof overrides the heuristic. The corroborating read: leak_guard's low ratio is actually *consistent with* "no robust exploitable structure here" — i.e. it agrees the positive tail is not a real edge.

## Verdict
**REFUTED.** ER-gated fixed-MA breakout at strictly 3 DOF, 4h, u100, taker-net, held-out: **−2.22% per-trade expectancy (winrate 27%, 11/77 assets positive).** Provably leak-free. This is the honest null; it does not clear the target and does not justify adding adaptive-map cells. (Negative held-out is a valid, valuable outcome — it redirects effort away from MA-cross timing as the alpha source.)
