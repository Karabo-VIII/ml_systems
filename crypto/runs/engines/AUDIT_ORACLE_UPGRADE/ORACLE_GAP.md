# LONG-ONLY ORACLE + the realizable engine's GAP to it

**Date:** 2026-06-28 (Claude). **Window:** OOS-2024 (2024-05-15..2024-12-31), **2025 SEALED** (asserted max bar < 2025-01-01).
**Universe:** u100 (n=63 @ 1d, n=64 @ 4h). **Cost:** PROD (taker+slip 0.0028 RT). **Regime: DE-LEAKED (lag1)** — the
V2 numbers here are post the same-day-regime leak fix (leakcheck.json; ~15% haircut already applied).
**Binding:** long-only spot lev=1 (no market-neutral even as framing); rank by WEALTH; MtM/lag-1; reused verified
`ma_slice_engine` / `ma_wealth_driver` / `canonical_exits` path.
**Script:** `oracle_gap_study.py` -> `oracle_gap.json`.

The oracle here is the **best-achievable in the ALLOWED space (long-only spot)**, NOT an unconstrained perfect
predictor. Three nested levels per non-overlapping 7d week + cumulative.

---

## THE 3 LEVELS (%/wk, slice-profit-rate)

| Level | What it is | 1d (%/wk) | slice-prof | 4h (%/wk) | slice-prof |
|---|---|---:|---:|---:|---:|
| **L1 perfect-cash (per-slice)** | deploy each name iff its forward-7d return > cash, EW over deployed. The mandate-headline absolute long-only ceiling. | **+12.04** | **94%** | **+11.84** | **94%** |
| L1 perfect-cash (per-bar) | same but per-BAR foresight (accounting-consistent ceiling for the DNA arithmetic). **Carries rebalancing premium — do NOT headline.** | +24.14 | 100% | +43.72 | 100% |
| **L2a best-realizable type** | realizable causal band signal, HINDSIGHT-best MA type for the window (VIDYA@1d, SMA@4h), de-leaked regime, EW. The achievable ceiling at EW concentration. | +3.52 | 38% | +3.41 | 31% |
| **L2b best-realizable + concentrate** | L2a + concentrate on top-K held names by trailing momentum (best K=3 in-window). Realizable concentration dial. **Per-bar-rebalanced — concentration-premium inflated.** | +14.04 | 41% | +12.30 | 41% |
| **L3 V2 engine (frozen band, de-leaked, EW)** | the deployed frozen TRAIN+VAL band (SMA), de-leaked regime, EW. The current %/wk (uncapped). | **+3.29** | **34%** | **+3.41** | **31%** |

Cross-check: L3 V2 SMA|1d (+3.286) and L2a SMA|4h (+3.408) match `leakcheck.json` lag1 SMA|1d (+3.286) / SMA|4h
(+3.408) **exactly** — the harness reproduces the verified de-leaked engine. (At 4h the hindsight-best type IS SMA,
so L2a == L3 there.)

> **Capped deployable (the number to actually quote):** L3 EW de-leaked is +3.29%/wk (1d) / +3.41 (4h). The 15%
> per-name cap dials this to **~+1.2-1.4%/wk** (the cap multiplier ~0.42, from CURRENT.md's EW3.4->cap1.4 dial,
> now on de-leaked figures). That capped ~1.2-1.4%/wk is the honest deployable headline — NOT any compound.

---

## THE TWO GAPS (the headroom split)

The total V2 -> absolute-ceiling gap splits into a **recoverable** part (better tuning/sizing WITHIN the rules) and
an **un-capturable** part (needs a genuinely predictive signal — knowing which names rise — which long-only price-TI
does not have).

| | 1d (pp/wk) | 4h (pp/wk) |
|---|---:|---:|
| **UPGRADE headroom** — V2 -> realizable ceiling (recoverable by tuning/sizing) | **+10.76** | **+8.89** |
| **Un-capturable info limit** — realizable ceiling -> perfect cash (needs prediction) | **+10.10** | **+31.42** |
| Total V2 -> per-bar perfect | +20.85 | +40.31 |

Read: even with HINDSIGHT-optimal band type + HINDSIGHT-optimal concentration, the realizable causal signal leaves
**+10 to +31 pp/wk on the table vs perfect cash-timing** — that residual is the part you can only get by *predicting*
which names go up, which is exactly the directional edge the project has refuted (D17/D55/D77, North-Star 34% hit-
rate). It is the structural long-only ceiling, not an engineering miss.

---

## GAP DNA (per-bar basis — the only basis where components sum exactly)

| Component | 1d (pp/wk) | 4h (pp/wk) | Nature |
|---|---:|---:|---|
| **late entry / early exit** | +0.23 | +0.00 | frozen-type -> hindsight-best-type per-bar capture. **Tiny** — the frozen SMA band already captures essentially all the per-bar move the best type would; entry/exit timing is NOT where the money leaks. |
| **under-concentration** | **+10.52** | **+8.89** | EW -> concentrate on the winners (top-K). The single biggest *recoverable* lever at 1d. This is the per-name CAP dial taken the other way (concentrate, don't dilute). **But: per-bar-rebalanced -> carries rebalancing premium; the realizable slice-frequency lift is smaller.** |
| **wrong cash calls** | +10.10 | **+31.42** | realizable ceiling -> perfect cash (deployed in a down-bar / cash in an up-bar). The **un-capturable info limit** — needs directional prediction. Dominates at 4h (more bars = more cash-timing decisions to get right). |

**Biggest recoverable headroom: UNDER-CONCENTRATION at 1d (+10.5 pp/wk), CASH-TIMING dominates at 4h (but that
piece is un-capturable).** Late/early timing is a non-issue (~0). So the realizable upgrade path is **concentration,
not better entries/exits**.

---

## VERDICT / WHAT THIS MEANS FOR THE UPGRADE

1. **Entry/exit timing is NOT the leak** (+0.0-0.2 pp/wk). The frozen SMA band already captures the per-bar move a
   hindsight-best type would. Do not spend the upgrade budget on exit/entry tuning.
2. **Concentration IS the recoverable lever** (+8.9-10.5 pp/wk on the per-bar basis; the slice-frequency realizable
   version is the known cap dial: 15%->+1.4, 33%->+1.6, 50%->+2.1, EW->+3.4 %/wk). The honest upgrade = **raise the
   per-name cap / concentrate into the strongest held names**, accepting more single-name DD. This is a risk dial the
   user controls, not new alpha. L2b (top-3 concentration) lifts Sharpe to ~3.9-5.0 and HALVES maxDD (-24% vs -39%)
   vs EW — concentration into momentum winners is both higher-return AND lower-DD in this trending OOS window
   (caveat: that DD benefit is window-specific; concentration generally RAISES single-name DD risk).
3. **The rest of the gap is un-capturable long-only** (+10 to +31 pp/wk = wrong-cash-calls): it requires knowing
   which names rise (directional prediction), the refuted vein. This is the same structural ceiling the North-Star
   reckoning found (34% of 7d windows profitable; trend = low-hit-rate). The perfect-cash ceiling deploys only ~44%
   of names per week and still hits 94% slice-profitability — proving the ceiling is real and the engine's 34% is
   the price of NOT having the cash-call oracle.
4. **The mandate-headline ceiling is +12%/wk (perfect cash-timing, per-slice), 94% slice-profitable.** The realizable
   engine at EW is +3.3-3.4%/wk, 31-34% slice-profitable. The capped deployable is ~+1.2-1.4%/wk. The gap from
   deployable to the +12%/wk ceiling is mostly the un-capturable directional-prediction piece + the concentration
   dial the user can turn.

**CAVEATS (anti-fabrication):** (a) per-bar L1 and L2b carry rebalancing premium — never headline the per-bar
compound (1e8% at 4h is meaningless); the per-slice L1 (+12%/wk) and capped L3 (~+1.3%/wk) are the honest anchors.
(b) L2a/L2b use IN-WINDOW hindsight (best type, best K) — they are CEILINGS, not deployable; the deployable is L3
frozen. (c) The L2b DD/Sharpe benefit is OOS-2024-window-specific (a clean trending window). (d) Regime is DE-LEAKED
(lag1) throughout, so all figures are ~15% below the previously-reported leaked headlines.
