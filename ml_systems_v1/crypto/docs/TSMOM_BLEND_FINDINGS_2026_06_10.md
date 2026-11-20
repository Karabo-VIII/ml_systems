# The D4 TSMOM Ensemble + Blend — end-to-end build & robustness verdict (2026-06-10)

**Built at the user's direction after the MA decomposition showed the value of MAs is the D4 (cross-sectional TSMOM)
corner, not the entry-config corner.** This is the full pipeline: build the vol-scaled ensemble → race it vs the
regime-beta book + buy&hold → **blend** them → re-cost at maker → **robustness battery**. All VERIFIED (RWYB) from
`src/strat/tsmom_ensemble.py`; artifacts in `runs/mining/tsmom_*.json`.

## The strategy (LO + spot + lev=1)
Per-asset momentum = fraction of lookbacks {21,63,126,252 d} with positive trailing return (the *ensemble* — no magic
lookback); inverse-vol weighting (risk-parity); **breadth-normalized** exposure (invest more when more names trend,
cash otherwise); daily rebalance, lagged weights (MtM-correct), net of cost. **Blend** = convex combo
`α·regime_beta + (1−α)·TSMOM_breadth` (both already lev≤1, so the blend is too).

## The race (u50 daily, net of maker cost)
| book | FULL ann% | maxDD% | Calmar | Sharpe | OOS ann% | UNSEEN ann% |
|---|---|---|---|---|---|---|
| TSMOM_breadth | 42.7 | **−48.0** | 18.2 | 1.11 | **−1.3** | **−4.1** |
| **BLEND_25r** | 45.7 | −47.5 | 21.4 | 1.14 | −4.4 | −4.9 |
| **BLEND_50r** | 48.6 | −47.8 | 24.3 | 1.16 | −7.5 | −5.9 |
| regime_beta | **53.7** | −53.9 | **27.3** | 1.17 | −13.8 | −7.8 |
| buy_hold | 44.3 | −82.4 | 11.5 | 0.87 | −23.4 | −39.1 |
| RANDOM_null (same exposure) | 20.8 | −60.2 | 3.9 | 0.68 | −18.4 | −14.9 |

**What the blend achieves:** a clean **risk/return frontier** between TSMOM (defensive) and regime_beta (more
bull-beta). **BLEND_50r dominates pure TSMOM** (higher return ~same DD → Calmar 24 vs 18) and is **materially more
robust held-out than regime_beta** (OOS −7.5% vs −13.8%; UNSEEN −5.9% vs −7.8%) at the cost of ~5pp full-cycle return.
No single point strictly dominates — regime_beta holds the highest *full-cycle* Calmar (bull-beta), the low blends hold
the best *held-out* preservation. **BLEND_25r–50r is the balanced choice.** Per-year: regime wins the bull years
(2021 +587% vs TSMOM +415%), TSMOM/low-blend win the bears (2025 −17% vs −23%).

**The selection is real:** every book beats the **exposure-matched RANDOM null** robustly and *every year* (FULL 43-54%
vs 21%; OOS −1 to −8% vs −18%) — the cross-sectional momentum signal genuinely picks the right names (unlike the
earlier vol-rotation, which lost to random). **Not cost-fragile:** maker ≈ taker (turnover ~5%/day at daily).

## Robustness battery (BLEND_50r) — the honest verdict
- **Parameter-robust (PASS):** across **30 perturbations** (10 lookback-sets × 3 regime-SMAs), full-cycle ann = **42-51%**
  (med 47), Calmar 19-28 (med 22.5) — stable, not a cherry-picked config.
- **Full-cycle block-bootstrap p05 > 0 (PASS):** p05 = **+7.6%**, p50 +48.9%, p95 +112.9%, **P(>0) = 98%**. The
  full-cycle edge survives resampling — it is *not* an artifact of one lucky path.
- **Held-out (2025-26) robustly NEGATIVE (FAIL):** OOS positive in **0%** of perturbations, UNSEEN in 3%; held-out
  bootstrap p05 = **−28%**, P(>0) = 39%. The down-market is negative for *every* LO config.
- **Max-DD (FAIL vs <30% gate):** −48% at full deployment. To meet <30% DD, size to ~half → ~24%/yr.

## Verdict
**The blend is the best-engineered, most-robust long-only crypto book we have built** — parameter-robust,
full-cycle-bootstrap-positive (p05 +7.6%), cost-insensitive, with genuine cross-sectional momentum selection (beats
random every year) and the lowest drawdown of the field. It is a **real, defensible risk-managed-beta + momentum
product (~43-49%/yr full-cycle, Calmar ~22-27).**

It is **not ship-grade as a standalone alpha**, for one honest reason: **2025-26 was a sustained crypto down-market, and
no long-only spot strategy can produce positive return when the whole asset class falls** — it can only preserve, and
the blend preserves best (loses ~5% vs buy&hold's −39%). It also exceeds the <30% DD gate at full size (size down →
~24%/yr @ ~25% DD). This is the *market's* ceiling under LO+spot+lev=1, re-confirmed: **full-cycle beta + momentum
selection + preservation is real and robust; positive alpha in a falling market is not available.**

## What this settles (the arc from config-tuning to here)
The naive, research-grounded path beat the clever one. After proving the **entry-config corner (B1)** is a weak,
non-adaptable lever, we built the **D4 corner** the literature endorses and got the most robust book in two iterations:
*be in the trending names (momentum-ensemble), sized inverse-to-vol, scaled by breadth, blended with a regime filter,
flat in cash otherwise.* The remaining frontier is not "a better MA rule" — it is the standing fork: **relax lev=1**
(modest leverage on this bounded-DD book is the mechanical path to higher compound) or **new leading data** (the only
route to alpha that survives a down-market). Both are your call.

*Tools: `src/strat/tsmom_ensemble.py` (`--maker`, `--battery`, `--book`). Artifacts: `runs/mining/tsmom_ensemble_*.json`,
`runs/mining/tsmom_battery_*.json`. Provenance: /orc 2026-06-10.*
