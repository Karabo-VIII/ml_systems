# MA config granularity — TRAIN walk-forward discovery (2026-06-11)

> User /orc task: per-asset config is dead (regime_dna_lab) — now test the next granularities
> **per-CLUSTER** (asset_dna BLUE/STEADY/VOLATILE/DEGEN) and **per-REGIME** (causal trend UP/DOWN), the
> honest way: walk-forward **within the training window only**. Val/OOS/Unseen NOT touched (sealed).
> 2-MA cross + 3-MA alignment grids overlaid across ALL assets. Tool:
> [`src/strat/train_ma_walkforward.py`](../src/strat/train_ma_walkforward.py). RED-teamed (2 CRITICAL
> fixed before any verdict was read — see below).

## The test
Walk-forward within TRAIN (ts < 2024-05-15): split into K chronological folds; on each fold select
the best config per group on all earlier folds, **test on the next fold** (out-of-fold, in-train).
A granularity "earns its degrees of freedom" iff it beats POOLED (one config for all assets) on a
**PAIRED** per-(asset,fold) basis — matched denominators, market-correlation cancelled. Power
reported (MDE at 80% power). Fold-boundary force-close prices at the entry fold's last close (no
look-ahead). Holdout sealed throughout.

## The RED-team caught (and we fixed) two CRITICAL methodology bugs first
1. **Different-sized books.** POOLED uses 1 config, CLUSTER 3, CLUSTER_REGIME 6 — each generates its
   own trades, so comparing per-trade *means* across them is the recurring more-trades artifact. →
   Fixed to a **paired per-(asset,fold)** comparison.
2. **No statistical power.** The first design's MDE was 30–46pp/trade vs observed edges <10pp — so
   "none beats pooled" was a *can't-tell*, not a refutation. → Now we report MDE and label each cell
   VALID / INCONCLUSIVE-underpowered / no-edge.
(The preliminary "none valid" read I gave before the audit was therefore overstated — corrected here.)

## Results (paired diff vs POOLED, per-trade pp; t-stat; verdict)
| | u10 2MA | u10 3MA | u50 2MA | u50 3MA |
|---|---|---|---|---|
| **CLUSTER** (asset_dna) | +1.6 (t 0.7) inconcl | **+5.0 (t 1.8) VALID** | **−4.1 (t −1.1)** | **−9.5 (t −1.5)** |
| **REGIME** (UP/DOWN) | −11.3 (t −1.8) | −14.9 (t −1.3) | −2.8 (t −0.4) | **−25.3 (t −2.7)** |
| **CLUSTER_REGIME** | +0.2 (t 0.0) | +6.3 (t 0.7) | **−18.1 (t −2.3)** | −11.8 (t −1.3) |

## Verdict
1. **Per-CLUSTER: NOT robust.** The one positive cell (u10 3MA, +5pp t=1.78 VALID) **did not
   replicate at u50** — it reversed to −9.5pp. A 10-asset/3-cluster small-sample artifact, not a real
   edge. Clustering by DNA does not reliably beat pooled.
2. **Per-REGIME: it HURTS.** Negative in all four cells, significantly so at u50 3MA (−25pp, t=−2.73).
   Switching the config by regime adds the long-only DOWN-regime configs, which are a drag — the same
   D58 bear-bounce dead vein the sibling `regime_dna_lab` found. Don't regime-switch the config;
   regime-GATE (trade UP, sit in cash in DOWN) instead.
3. **CLUSTER_REGIME: no benefit** (leans negative, u50 2MA t=−2.28) — too many DoF, no payoff.
4. **POOLED is the robust granularity** — one MA config applied across all assets. Now established on
   a fair, paired, power-aware test, and the finer groupings don't merely fail to beat it; several
   *actively hurt*.

This closes the granularity ladder consistently: **per-asset dead → per-cluster not robust →
per-regime hurts → POOLED wins.** The user's DNA hypothesis (different config per asset/cluster/
regime) is, on honest walk-forward, not a real lever in MA-config space.

## Scope + honesty caveats
- **TRAIN walk-forward only** (holdout sealed, as instructed). The pooled MA config shows a positive
  per-trade mean in-train (descriptive ~+30%/trade, win 37%) but this is IN-SAMPLE; `regime_dna_lab`
  showed such MA edges decay to ~0/negative on the true OOS/UNSEEN. This discovery settles the
  **architecture** (pooled, not per-group), NOT a holdout edge.
- 6 MA-family grids (2MA + 3MA), 1d, long-only spot. Other signal families / cadences untested here.
- Many cells remain underpowered (MDE 9–23pp); the *negative lean* of regime/cluster_regime at u50
  (t up to −2.7) is the firming signal, not the inconclusive positives.

Repro: `python -m strat.train_ma_walkforward --universe u50 --folds 6` (seed-free; git lineage in JSON).
