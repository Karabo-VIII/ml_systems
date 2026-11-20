# Literature-Applied Insights — 2026-05-23 12:55 SAST

Web research conducted while POVs were running. Findings to integrate into the deploy work.

## Key concepts found in 2024-2025 literature

### 1. Minimum Regime Performance (MRP) — strategy-decay risk metric

From Smith et al. 2026 (arxiv 2604.08356): MRP = minimum Sharpe across distinct market regimes. Higher MRP = more durable strategy. This is a NATURAL measure of robustness — captures the worst-realized Sharpe across structural environments.

**Apply to our work**:
- V7 is 100% bull → MRP undefined / effectively 0 in chop/bear → FAILS MRP test
- V1 has bull/chop/bear mix → has nonzero MRP per regime
- COMPOSITE explicitly routes per regime → MRP = min(bull V7 Sharpe, chop basket Sharpe, bear basket Sharpe)
- **MRP would penalize V7 heavily relative to COMPOSITE — supports our F24 climb-back action**

### 2. Survivorship-weighted council (dynamic ensemble)

From Turner 2025 evolutionary crypto bot: "survivorship-weighted council of the strongest models handles predictions, with winners promoted and weaker members pruned."

**This is exactly what POV-13 is building** — dynamic engine eligibility timeline that promotes engines whose trailing edge is positive AND prunes engines whose trailing edge collapses. Our approach is aligned with current literature.

**Implementation refinement**: literature suggests EWMA-weighted (not equal-weighted) rolling means for the eligibility score. POV-13 uses simple trailing-mean; consider upgrade to EWMA half-life ~14 days for better responsiveness.

### 3. Page-Hinkley drift detection (formal statistical drift)

Multiple methodologies for concept drift: ADDM, DDM, Page-Hinkley, structural break tests, Kalman filters.

**Apply**: per-engine, track Page-Hinkley statistic on rolling fire-pnl. When PH exceeds threshold, flag engine as "drift event" — exclude from basket until drift resolves.

**Cost**: 50 LOC; adds rigorous statistical drift detection on top of POV-13's rolling-mean eligibility.

### 4. Rolling 30-day Sharpe-maximizing weights vs equal-weight (Modern Portfolio Theory)

From Multi-agent crypto allocator (arxiv 2507.20468): compares static equal-weight vs rolling 30-day Sharpe-maximizing on top-10 cryptos. Dynamic rebalancing significantly improves Sharpe.

**Our analog**: our top-3 picks/day at 25% equal-weight is STATIC. A rolling-Sharpe-maximizing variant would weight picks by their TRAILING Sharpe (not just consensus count). This is the same idea as F21 consensus-weighted but with Sharpe-weighting instead of consensus-weighting.

### 5. Trend-following adaptive 150-pair benchmark

From arxiv 2602.11708: adaptive trend-following on 150+ crypto pairs (2022-2024): Sharpe 2.41, maxDD -12.7%.

**Comparison to our work**:
- V1 32-engine TRAIN: Sharpe 3.18 (point), 0.96 (5%-CI) — at the published-2.41 benchmark only when 70% of TRAIN edge holds
- V7 TRIPLE: Sharpe 3.65 (point), 1.64 (5%-CI) — exceeds 2.41 even at 5% CI
- COMPOSITE: Sharpe 3.69 — exceeds 2.41
- BUT: published benchmarks include OOS by definition; our numbers are TRAIN-only. Published 2.41 = our deflated 5%-CI → V7 + COMPOSITE pass the bar; V1 does not on its own

## Actions taken applying this research

1. **F26+** add MRP discussion to deploy candidates ladder — V7 fails MRP, COMPOSITE wins on MRP
2. **POV-13 informed** — current dynamic eligibility design is literature-aligned (survivorship-weighted council). EWMA + Page-Hinkley are next-iteration upgrades.
3. **Future iteration target**: implement Sharpe-weighted top-3 pick (refinement of F21 consensus-weighted) — should outperform consensus-weighted further.

## Provenance

Web searches conducted via WebSearch tool on 2026-05-23 at 12:55 SAST. Search queries: "adaptive engine ensemble dynamic basket weighting rolling window survivorship crypto quantitative 2024 2025"; "strategy regime decay detection per-rule eligibility tracker concept drift trading systems 2025". 10+ papers/articles surfaced; key references applied above.

**Honest caveat**: these are 2024-2025 cite-able findings but execution-level details (exact EWMA half-life, exact Page-Hinkley threshold) are paper-specific. Should be tuned to our catalog via grid-search in next iteration.
