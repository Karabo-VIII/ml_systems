# Final candidate slate (2026-07-01) — 4 deployable candidates, honestly graded

All: u100 cross-sectional basket (NOT BTC), 1d, 14d-lookback / 7d-slice, inv-vol cap33, next-open fills, prod cost
0.0028 RT, long-only spot lev=1. Graded on the full OOS 2024-05-15 → 2026-05-28 ($10k start). DSR deflated for the
**~40 configs tried across this session** (honest multiple-comparisons, not the raw best).

| Candidate | $10k → | comp | maxDD | Calmar | %wk+ | p05 | **DSR₄₀** | sub-periods (pos/4) | tier |
|---|---|---|---|---|---|---|---|---|---|
| **MOM8_ER** | **$14,645** | **+46%** | **−21%** | **2.23** | 8% | −28% | 0.12 | +22/+5/+14/+0 (3/4) | **LEAD — promising** |
| **CONV** | $13,647 | +36% | −33% | 1.10 | 16% | −44% | 0.07 | +65/−26/+10/+1 (3/4) | **PROVEN de-risked beta** |
| **SMA_ER** | $11,128 | +11% | −21% | 0.55 | 10% | −38% | 0.04 | +13/+2/−6/+2 (3/4) | capital-preservation |
| **PARTIAL** | $12,242 | +22% | −36% | 0.62 | **49%** | −46% | 0.05 | +63/−25/−3/+3 (2/4) | consistency |
| *BH_btc* | $11,095 | +11% | −50% | 0.22 | 54% | −61% | — | (2/4) | benchmark |
| *BH_alt (EW-u100)* | $4,455 | −55% | −81% | −0.68 | 45% | −93% | — | (1/4) | benchmark |

## VAL (2023-07→2024-05, the untuned window) — the out-of-sample-for-the-DESIGN test

Gate params (breadth 0.5/0.4, ER>0.35, cap33, K8) were chosen looking at OOS → OOS is *in-sample for the gate design*.
VAL was **never tuned on** (in-sample only for the denoised MA band ensemble; MOM8_ER is param-free → **fully** OOS on VAL).

| Candidate | $10k → | comp | maxDD | Calmar | NW_p | p05 | VAL→OOS |
|---|---|---|---|---|---|---|---|
| **MOM8_ER** | **$35,717** | **+257%** | −21% | **12.42** | 0.07 | **+13%** | **HOLDS** |
| CONV | $22,210 | +122% | −31% | 3.89 | 0.07 | −16% | HOLDS |
| **SMA_ER** | $23,187 | +132% | −14% | 9.54 | **0.03** | **+28%** | HOLDS |
| PARTIAL | $21,494 | +115% | −29% | 3.90 | 0.07 | −12% | HOLDS |
| *BH_btc* | $20,138 | +101% | −20% | 5.05 | 0.05 | +3% | — |
| *BH_alt* | $23,838 | +138% | −35% | 3.91 | 0.07 | −20% | — |

**All 4 HOLD** (positive on both windows) → H0 "overfit to OOS" **REJECTED**. On VAL, **SMA_ER (NW p 0.03, p05 +28%)**
and **MOM8_ER (p05 +13%)** clear the project robustness bar (p05>0) for the first time in the arc — but *only in the bull
regime*. VAL magnitude (+257%) ≫ OOS (+46%) is **regime, not overfit** (BH_alt also +138% on VAL). Full-cycle p05 stays
negative (the bear is the un-earnable part for long-only). Artifact: `strat_val_oos.{py,json}`.

## Final prod checks (paper-trade-safe gate) — `strat_prod_deploy.py`

| Check | CONV | SMA_ER | PARTIAL | MOM8_ER |
|---|---|---|---|---|
| **Causality** (perturb close[t0] → P&L changes only t>t0) | PASS | PASS | PASS | PASS |
| **Determinism** (rebuild → bit-identical) | PASS | PASS | PASS | PASS |
| **Cost applied** (net < gross on turnover) | PASS | PASS | PASS | PASS |
| **PROD-GATE** | PAPER ✓ | PAPER ✓ | PAPER ✓ | PAPER ✓ |

The causality perturbation test **caught a real off-by-one cost-timing bug** in the shared `net_of`: it charged the
rebalance cost `|W[t]−W[t−1]|` at bar t, but W[t] uses close[t] → the *cost* term (not the return) had a same-bar
dependency. Fixed to charge `|W[t−1]−W[t−2]|` (the rebalance that actually fills at t) → fully causal. Compound
unchanged (same total turnover, shifted one bar). **All 4 are paper-trade-safe; none is real-capital-ready** (every
DSR₄₀ < 0.15, full-cycle p05 < 0).

## The candidates

- **MOM8_ER (lead)** — momentum(14d) top-8, gated by **breadth AND trend-efficiency** (deploy only clean-trend weeks).
  Best on every risk-adjusted axis: highest return (+46%), lowest drawdown tie (−21%), **Calmar 2.23 (2× the next best)**,
  positive in 3/4 sub-periods. **Grid-robust:** across ER thr {0.25–0.45} × K {5,8,12}, *every* config is positive
  (+23% to +100%), most 2-4/4 sub-periods — it is **not a lucky single config**. Mechanism: momentum only works *inside*
  clean-trend regimes; the ER filter finds them, momentum picks the movers within them. Raw momentum (no gate) is −50%.
- **CONV (proven de-risked beta)** — the MA cross-type ensemble, breadth-gated. +36% / −33%, Calmar 1.10, 3/4 sub-periods.
  The canonical, most-battle-tested build; the safe wealth deployable.
- **SMA_ER (capital-preservation)** — lowest drawdown (−21%), +11%, 3/4 sub-periods. For a minimal-drawdown budget.
- **PARTIAL (consistency)** — always 15-100% invested; 49% of weeks positive (vs 8-16% for the gated engines), +22% / −36%.
  For "in the market every week, de-risked." Weakest sub-period record (2/4).

## Binding honest caveats (no false victory)

- **None is statistically-proven alpha.** Every DSR₄₀ < 0.15 (need >0.95), every p05 < 0, every NW p > 0.05. Deflated for
  the ~40-config search, none clears significance. These are **de-risked / regime-conditional beta**, not proven edges.
- **The tension on MOM8_ER:** it is *grid-robust* (any threshold/K works) yet *DSR-low* — because its return is
  **regime-concentrated** (the clean-trend weeks, mostly the late-2024 bull + mid-2025 rallies). The *strategy family* is
  robust; the *magnitude* rode a small number of favorable regimes. That's why it's LEAD-PROMISING, not PROVEN.
- **Survivorship:** today's u100 applied retroactively → headline returns overstated. The robust deliverable is the
  **drawdown control + the beat-vs-BH_alt (−55%)**, which is survivorship-neutral (same universe). Size off DD, not return.
- **Live state (2026-05-28, breadth 0.38):** CONV / SMA_ER / MOM8_ER = **CASH** (gate shut); PARTIAL = DEPLOY (65 names).

## Pre-registered live-forward gate (converts promising → shippable)

Freeze each candidate's exact spec (no re-tuning — re-tuning voids it). Track the weekly book from today. Escalate a
candidate to real capital only after **≥20 deployed weeks** with: deployed-week median > 0, compound > cash, compound >
BH_btc measured over the same deployed weeks, maxDD within its backtest floor (CONV −35% / MOM8_ER −25% / SMA_ER −25%).
Kill on: deployed-median ≤ 0 over 20+ weeks, compound < cash, or maxDD breach. Sizing: de-risked beta, hard cap per
deployed week; do not size off the backtest compound.

## Wired into the paper trader (live-forward track is now RUNNING)

`strat_prod_deploy.py` is the weekly emitter. Each run: (1) re-runs the 3 prod checks, (2) emits every candidate's
current book (DEPLOY names+weights or CASH), (3) appends to `live_books/journal.jsonl` (idempotent by date) and
regenerates `PAPER_TRADE_LEDGER.md`. **Re-run weekly** (`python runs/engines/STRAT_TIERS/strat_prod_deploy.py`) to
accumulate the pre-registered track. First entry logged (as-of latest data 2026-05-28): CONV/SMA_ER/MOM8_ER = CASH,
PARTIAL = DEPLOY (65 names, gross 0.29). Weekly decisions accumulate in the ledger until the promotion gate is met.

## Recommendation

Track all four live from today — the emitter is wired and the first week is logged. **MOM8_ER is the lead** (best
risk-adjusted, grid-robust, +257% on the untuned VAL window, cleared p05>0 there) and the one most worth proving.
**CONV is the proven fallback.** Ship real capital to none until the live-forward gate is met. Artifacts:
`strat_prod_deploy.{py,json}`, `strat_val_oos.{py,json}`, `strat_final_candidates.{py,json}`, `strat_momentum_capture.json`,
`strat_oracle_gap.json`; live track: `PAPER_TRADE_LEDGER.md` + `live_books/`.
