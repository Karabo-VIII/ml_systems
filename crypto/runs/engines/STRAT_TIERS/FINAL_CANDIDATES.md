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

## Recommendation

Track all four live from today. **MOM8_ER is the lead** (best risk-adjusted, grid-robust) and the one most worth proving.
**CONV is the proven fallback.** Ship real capital to none until the live-forward gate is met. Artifacts:
`strat_final_candidates.{py,json}`, `strat_momentum_capture.json`, `strat_oracle_gap.json`.
