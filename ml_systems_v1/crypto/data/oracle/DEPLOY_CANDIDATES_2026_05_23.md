> **SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup)**: This document predates the
> canonical split-discipline gate. References to "OOS" in this file may include data
> from the canonical UNSEEN window (>=2026-01-01) per [src/split_config.py](../../src/split_config.py).
> Use this document for historical context only; deploy decisions citing UNSEEN-relevant
> claims must be re-derived from the canonical segments.

# Deploy Candidates 2026-05-23 (post-data-bug audit + 6-POV mining + bootstrap CI)

**Status**: DELAY-NEEDS-V3-REPLAY (per project policy). All numbers are TRAIN-realized. OOS deflation expected at the bootstrap-derived factors shown below.

**Generated**: 2026-05-23 ~11:25 SAST.
**Inputs**: 
- Catalog: `data/oracle/engine_catalog_discovery.parquet` (234 catch-tier engines)
- Returns: **`close.pct_change()` computed directly from chimera_v51 close column** (NOT the corrupted `target_return_1_raw` — see `DATA_BUG_target_return_1_raw_2026_05_23.md`)
- Filter pipelines: lifecycle decay, anti-fragility audit, cluster-residual audit, red-team audit, synthetic-stress bootstrap

## 🟢 NEW PRIMARY DEPLOY CANDIDATE — F39 DECILE-3 BASKET (post-OOS validation, 13:40 SAST)

**15 of 18 engines SURVIVED full OOS (2024-05-16 to 2026-05-19) = 83% survival rate** (vs catalog baseline 17%).

| Asset | Class | Config | Regime | TRAIN comp | VAL comp | **OOS comp** | OOS ratio |
|---|---|---|---|---:|---:|---:|---:|
| **LINK** | RSI_threshold | p_5_lo_40_hi_60 | chop | +18.1% | +226.4% | **+418.8%** | **+23.14** |
| LINK | RSI_threshold | p_6_lo_40_hi_60 | chop | +18.1% | +213.5% | **+369.3%** | +20.40 |
| LINK | RSI_threshold | p_5_lo_35_hi_60 | chop | +18.1% | +204.2% | **+353.2%** | +19.52 |
| LINK | RSI_threshold | p_7_lo_40_hi_60 | chop | +18.1% | +212.0% | +313.5% | +17.32 |
| LINK | RSI_threshold | p_6_lo_35_hi_60 | chop | +18.1% | +189.0% | +252.4% | +13.94 |
| LINK | RSI_threshold | p_8_lo_40_hi_60 | chop | +18.1% | +258.4% | +238.8% | +13.19 |
| LINK | RSI_threshold | p_7_lo_35_hi_60 | chop | +18.1% | +177.0% | +190.4% | +10.52 |
| DASH | OBV_zscore | p_100_t_1.5 | chop | +10.0% | +45.0% | +65.1% | +6.52 |
| JST | RSI_threshold | p_6_lo_40_hi_80 | chop | +8.2% | +30.2% | +42.0% | +5.13 |
| JST | RSI_threshold | p_10_lo_40_hi_70 | chop | +10.9% | +26.8% | +48.0% | +4.40 |
| JST | RSI_threshold | p_9_lo_40_hi_70 | chop | +10.9% | +25.8% | +37.1% | +3.40 |
| APT | RSI_threshold | p_27_lo_40_hi_60 | bull | +10.3% | +13.4% | +20.8% | +2.02 |
| JST | RSI_threshold | p_6_lo_35_hi_75 | chop | +9.1% | +20.1% | +17.5% | +1.93 |
| APT | RSI_threshold | p_29_lo_40_hi_60 | bull | +11.7% | +23.3% | +21.9% | +1.88 |
| APT | RSI_threshold | p_28_lo_40_hi_60 | bull | +10.3% | +11.3% | +11.8% | +1.15 |

**3 INVERTED**: ETC norm_efficiency chop, DYDX OBV bull, ICP RSI bull. Drop these.

**Resulting 15-engine deploy basket**: 7 LINK + 5 JST + 3 APT + 1 DASH. RSI/OBV dominated. 12 of 15 chop + 3 bull (vs V7's 100% bull). True multi-regime.

**Recommended deploy stack**:
- **PRIMARY**: F39 15-engine decile-3 basket
- Backbone: 7 LINK RSI chop configs (deploy 2-3 of them to dedup near-identical configs; pick p_5, p_6, p_8 for sample variation)
- Diversifiers: DASH OBV chop (1) + JST RSI chop (2-3 distinct configs) + APT RSI bull (1-2)
- Final: ~7-9 unique engines, 4 assets (LINK/JST/APT/DASH), 2 regimes (chop + bull)

**Caveats**:
- DERIVED FROM VAL signal — strict pure-OOS test (post-2026-05-19) requires v3-paper-trade-replay later
- 7 LINK RSI configs are near-duplicates — dedup before deploy
- Re-run with multi-cutoff methodology to verify these survive on OTHER cutoff-derived deciles
- LINK ADV bottleneck per F25 capacity audit (LINK is in V1's diverse universe — capacity ceiling probably ~$50-100M at LINK concentration)

## 🚨 CATALOG-LEVEL INVALIDATION (post-F36/F36.b VAL Mining, 2026-05-23 13:30 SAST)

**POV-17 mined the VAL window using TRAIN methodology** (user's explicit request: "mine VAL the way we did TRAIN — tells us why we get the results"). Three findings make DEPLOY UNADVISABLE on the current catalog:

1. **TRAIN top-30 vs VAL top-30 Jaccard = 0.000** (ZERO engine overlap)
2. **Spearman(TRAIN compound, VAL compound) = -0.176 (p=0.009, N=222)** — TRAIN selection is STATISTICALLY WORSE than random for VAL
3. **VAL catch-tier survival rate 4.3%** — equal to what pure noise would produce
4. **234 catch-tier survivors out of 937 candidates lacks Bonferroni adjustment** — the catalog is statistically indistinguishable from a noise selection

**The catalog mining method is STRUCTURALLY INVALID.** Selecting engines by TRAIN compound is anti-predictive of OOS edge on this corpus.

**What this means for the deploy candidates below**: every basket numbered V1, V7, COMPOSITE, V7-SAFE, etc. shares the same flawed foundation. Their TRAIN performance does NOT predict OOS.

The 5-engine OOS-evidence-based deploy basket (F31) IS the post-hoc curation that produced positive OOS — but its 3 of 5 winners (FET wh_whale, PEPE Donchian, SUPER norm_deviation) are positive on VAL too (per F36), and 2 of 5 failures (FIL liq_long, WLD MA_SMA) bring the basket aggregate down. **This is the lower bound of survivable engines after 77-engine OOS validation + 16-lookalike validation**.

**Recommended methodology overhaul before any future deploy**:
1. Multi-cutoff catalog mining (rolling 12-month windows)
2. Engines must be top-30 on at least 2 of 3 non-overlapping cutoffs
3. CSCV PBO (Bailey-de Prado) for backtest-overfitting estimation
4. Bonferroni / BH multiple-comparisons correction at engine-selection
5. UNION of TRAIN-best + VAL-best as the production candidate set (LINK RSI chop family + the existing 5-engine survivors)

## 🟢 OOS-EVIDENCE-BASED DEPLOY SUBSET (post-F29 survivor profile, 2026-05-23 ~13:10 SAST)

**The only deploy-defensible engines after combined 61-engine OOS validation are**:

| Asset | Class | Config | Regime | TRAIN→OOS ratio | n_fires | Notes |
|---|---|---|---|---:|---:|---|
| FIL | measure_engines/liq_long_usd | op_gt_thr_1.0 | bull | **+0.37** | 10 | V7-member, SURVIVED |
| FIL | measure_engines/liq_long_usd | op_abs_gt_thr_1.0 | bull | **+0.34** | 10 | V7-member, SURVIVED (= same engine as above at threshold=1.0) |
| FET | measure_engines/wh_whale_net_usd | op_abs_gt_thr_1.5 | bull | **+0.79** | 13 | NOT in V7 (failed magnitude filter); SURVIVED |
| PEPE | Donchian_state_above_midline | period_55 | chop | **+1.14** | 10 | NOT in V7 (failed stability); SURVIVED (OOS > TRAIN) |
| WLD | YZ_vol_regime | t_0.5 | bull | **+0.34** | 10 | NOT in V7 (failed stability); SURVIVED |
| PEPE | MA_state_SMA_above | period_50 | chop | +0.26 | n/a | DECAYED_partial |
| XRP | measure_engines/hbr_eta_buy | op_gt_thr_1.0 | bull | +0.30 | 30 | DECAYED_partial |
| HBAR | measure_engines/norm_deviation | op_gt_thr_1.0 | chop | +0.05 | 34 | DECAYED_heavy but positive |
| SUI | measure_engines/bd_imbalance_l5 | op_lt_thr_1.0 | chop | +0.03 | 37 | DECAYED_heavy but positive |

**~4-5 unique engine signatures (dedup FIL × 2 = 1) across 5-6 assets (FIL/FET/PEPE/WLD/XRP/HBAR/SUI), regimes bull (4) + chop (4)**.

### F30 ADDITIONS (from lookalike OOS validation, 25% survival rate):

| Asset | Class | Config | Regime | TRAIN | OOS | Ratio | Notes |
|---|---|---|---|---:|---:|---:|---|
| **FET** | wh_whale_net_usd | op_abs_gt_thr_2.0 | bull | +45.7% | **+43.1%** | **+0.94** | STRONGEST survivor (94% TRAIN retention) |
| SUPER | norm_deviation | op_abs_gt_thr_1.0 | chop | +33.2% | +21.7% | +0.66 | New SURVIVED |
| WLD | MA_state_SMA_above | period_50 | chop | +37.6% | +18.6% | +0.49 | DECAYED_partial |
| HBAR | norm_deviation | op_abs_gt_thr_1.5 | chop | +32.4% | +15.8% | +0.49 | DECAYED_partial |

**Final OOS-evidence-based deploy candidates** (12 engines, 7 assets, 2 regimes):

| Asset | Class | Config | Regime | Ratio | Notes |
|---|---|---|---|---:|---|
| FET | wh_whale_net_usd | abs_2.0 | bull | +0.94 | Strongest |
| PEPE | Donchian | period_55 | chop | +1.14 | Strongest |
| FET | wh_whale_net_usd | abs_1.5 | bull | +0.79 | |
| SUPER | norm_deviation | abs_1.0 | chop | +0.66 | |
| WLD | MA_SMA | period_50 | chop | +0.49 | |
| HBAR | norm_deviation | abs_1.5 | chop | +0.49 | |
| FIL | liq_long_usd | gt_1.0 | bull | +0.37 | |
| FIL | liq_long_usd | abs_1.0 | bull | +0.34 | (= same as above at thr=1.0) |
| WLD | YZ_vol_regime | t_0.5 | bull | +0.34 | |
| PEPE | MA_SMA | period_50 | chop | +0.26 | DECAYED_partial |
| XRP | hbr_eta_buy | gt_1.0 | bull | +0.30 | |
| HBAR | norm_deviation | gt_1.0 | chop | +0.05 | DECAYED_heavy but positive |
| SUI | bd_imbalance_l5 | lt_1.0 | chop | +0.03 | DECAYED_heavy but positive |

**Recommended deploy basket** (top 5 by ratio + diversity):
- FET wh_whale_net_usd abs_2.0 bull (whale-signal, ratio 0.94)
- PEPE Donchian period_55 chop (chop trend, ratio 1.14)
- FIL liq_long_usd gt_1.0 bull (liquidation, ratio 0.37)
- SUPER norm_deviation abs_1.0 chop (chop mean-revert, ratio 0.66)
- WLD MA_SMA period_50 chop OR WLD YZ_vol_regime bull (chop trend / vol-regime, ratio ~0.4)

5 engines × 5 different classes × 5 different assets × 2 regimes (bull + chop) = TRUE diversification with OOS-validated edges.

**Counter-intuitively**: V7's filter logic EXCLUDED 3 of 5 best survivors (FET wh_whale, PEPE Donchian, WLD YZ_vol). The filter was anti-predictive.

**Pre-deploy actions**:
1. v3-paper-trade-replay this 9-engine set on UNSEEN 2026-Q1+Q2 window
2. Expect ~25-100% of TRAIN-realized compound (per ratio distribution above)
3. ENSURE no FET-cluster concentration — FET wh_whale + PEPE Donchian + WLD + FIL liq are reasonably uncorrelated
4. SIZE conservatively (1/4 Kelly) given small per-engine n_fires (~10-15)

## 🔴 OOS VALIDATION REALITY CHECK (2026-05-23 ~13:00 SAST, POV-14)

**Both V7 and V1 FAIL OOS validation** on consistent close-derived methodology:

| Basket | OOS classification | Inversion rate | Survivors |
|---|---|---:|---|
| V7 (12 engines) | 2 SURVIVED, 10 INVERTED | **83%** | FIL liq_long_usd × 2 (= 1 unique signature) |
| V1 (32 engines) | 2 SURVIVED, 1 DECAYED, 25 INVERTED, 4 DEAD | **78%** | FIL liq_long_usd × 2 + 1 other |

**Bootstrap CI (F23 p05 V7 Sharpe 1.64, V1 0.96) was OPTIMISTIC** — actual OOS deflation is ~83% sign-flip, not 5%-CI scaling.

**Implications**:
- V7 deploy is **NOT VIABLE** as stated. Only 1 unique engine (FIL liq_long_usd bull) survives OOS, with ratio +0.34-0.37.
- V1 deploy is **NOT VIABLE** as stated. Same survivor + 1 other engine; rest INVERTED.
- The "TRIPLE FILTER" (stable + not-concave + mag) does NOT predict OOS survival on this catalog.
- The 83% inversion rate matches the prior 30-engine baseline — our filter logic does NOT meaningfully reduce overfitting.

**What this means for the user's mandate**: deploy on TRAIN-selected baskets is HIGHLY RISKY. Need either:
(a) Use ONLY the surviving engine (FIL liq_long_usd bull) — single-engine deploy = capacity-limited + concentration risk
(b) Re-mine with a fundamentally different filter (POV-16's alternate filters all capture more OOS-positive engines than V7 — but TRAIN realized weaker)
(c) Accept that the daily-bar regime has insufficient signal to support stable engine deploy; pivot to V20 (tick-level) per CLAUDE.md mandate

## TL;DR ranked deploy ladder (final, post-bear/chop + capacity audit + OOS REALITY CHECK)

| Rank | Candidate | n_eng | Point %/d | Sharpe | 5%-CI floor %/d | 5%-CI Sharpe | NAV % | Capacity ceiling | Regime coverage | Honest verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **1** | **COMPOSITE (regime-routed: bull→V7 / chop→CHOP / bear→BEAR)** | 22 | **+0.501** | **3.69** | TBD | TBD | **NAV 4.14x / 304 days** | TBD | **bull+chop+bear** | **Satisfies "any regime" mandate; bear leg fragile but presence > absence** |
| **2** | **V7 TRIPLE FILTER** | 12 | +0.822 | 3.65 | +0.314 | 1.64 | +180% | **~$47M** | bull only | Best mean %/d; 6/12 high red-team-vuln; concentration on FET/APT |
| 3 | V1 + CONSENSUS-WEIGHTED sizing | 32 | +0.799 | 3.38 | TBD | TBD | **+908%** | **~$145M** | bull+chop+1 bear | Best NAV; TOP1@100% on consensus≥3 |
| 4 | V1 32-engine moderate-decoupling | 32 | +0.583 | 3.18 | +0.166 | **0.96** | +454% | **~$145M** | bull+chop+1 bear | Marginal p05 Sharpe; 18 assets diverse |
| 5 | V7-SAFE (red-team-pruned) | 6 | +0.406 | 3.40 | TBD | TBD | +62% | ~$20M | bull only | 50% alpha loss for adversarial robustness |

**Capacity ceilings** = max deploy USD where mean %/d ≥ 5bps (POV-11 Z2 audit). Above this, slippage/partial-fills push edge below noise floor.

**Recommended deploy stack** (per scale and regime):
- **Retail-Mid ($10K-$10M)**: V7 TRIPLE (best mean %/d) + COMPOSITE (regime-routed) as parallel sleeves
- **$10M-$47M**: V7 capacity still safe but approaching ceiling; consider V1+CONSENSUS-WEIGHTED to extend headroom
- **$47M-$145M**: V1 + CONSENSUS-WEIGHTED + COMPOSITE; V7 economic capacity exhausted
- **>$145M**: V1 capacity-bound; need 4h-cadence or alternative-data sleeves
- **Hard mandate**: NO V7-only deployment without COMPOSITE for "any regime"

## V7 TRIPLE FILTER (RECOMMENDED PRIMARY DEPLOY CANDIDATE)

**Definition**: catch-tier ∩ stable_flag (lifecycle decay) ∩ NOT-CONCAVE (anti-fragility) ∩ mean_pnl_pct ≥ 1.5%

**Sizing**: top-3 picks/day by n_engines_firing, 25% NAV each, 24bp RT taker cost, 1-day hold

**Results (TRAIN, FIXED close-derived returns)**:
- Mean %/d: **+0.822**
- Sharpe: **3.65**
- 95%-block-bootstrap 5th percentile: mean **+0.314%/d**, Sharpe **1.64**
- NAV over 136 active TRAIN days: +180%
- Hit rate: 55.9%

**12 basket members**:

| Asset | Class | Config | Regime | n_fires | mean_pnl% | hold_d |
|---|---|---|---|---:|---:|---:|
| APT | Distance_z_state | period_50_threshold_1.5 | bull | 10 | 5.16 | 3 |
| APT | MA_state_EMA_above | period_20 | bull | 13 | 2.53 | 3 |
| AR | MA_state_SMA_above | period_20 | bull | 10 | 4.45 | 1 |
| AR | measure_engines/te_in_btc | op_abs_gt_thr_1.0 | bull | 11 | 3.59 | 3 |
| DOT | Distance_z_state | period_50_threshold_1.5 | bull | 21 | 1.81 | 1 |
| FET | measure_engines/rv_bpv_5m | op_abs_gt_thr_1.0 | bull | 18 | 3.43 | 1 |
| FET | measure_engines/rv_bpv_5m | op_gt_thr_1.0 | bull | 10 | 5.59 | 1 |
| FET | measure_engines/rv_rv_5m | op_abs_gt_thr_1.0 | bull | 18 | 3.38 | 1 |
| FET | measure_engines/rv_rv_5m | op_gt_thr_1.0 | bull | 11 | 4.98 | 1 |
| FIL | measure_engines/liq_long_usd | op_abs_gt_thr_1.0 | bull | 10 | 3.56 | 1 |
| FIL | measure_engines/liq_long_usd | op_gt_thr_1.0 | bull | 10 | 3.56 | 1 |
| ICP | Distance_z_state | period_50_threshold_1.5 | bull | 22 | 5.03 | 1 |

**Composition concerns**:
- 6 assets only (FET ×4, APT ×2, AR ×2, FIL ×2, DOT, ICP)
- 100% BULL regime — NO bear or chop coverage
- Engine duplicates: FIL liq_long_usd op_abs and op_gt are functionally identical at threshold=1.0 (both produce same pnl on same fires)
- After dedup: ~9-10 unique engine signatures

**Red-team vulnerability (F22)**: 6/12 engines above-median vulnerability:
- DOT/ICP Distance_z_state, FIL liq_long_usd (×2), AR MA_state_SMA, [one more]
- The 6 SAFE V7 engines (APT × 2 + FET × 4) form a tighter sub-basket (V7-SAFE)

## V1 + CONSENSUS-WEIGHTED sizing (RECOMMENDED SECONDARY DEPLOY CANDIDATE)

**Definition**: Basket = 32-engine J<0.50 MIS (F1 RECONFIRMED). Sizing = TOP1 @ 100% on consensus-≥3 days, else TOP3 @ 25%.

**Results**:
- Mean %/d: **+0.799**
- Sharpe: **3.38**
- NAV: **+908%** (best NAV of all variants)
- Hit rate: 57.4%

**Operationalization rule**:
```
on each fire date:
  count n_engines firing per asset (asset cells with >=1 fire)
  if max consensus across all asset cells >= 3:
    pick the single asset with highest consensus, deploy 100% NAV (in lieu of cash)
  else:
    pick top-3 assets by n_engines, deploy 25% NAV each
```

**Mechanism**: F14 (corrected) showed 3-4 engine consensus = +1.025%/d mean fwd-ret (5x baseline). This sizing rule concentrates capital on high-conviction cells. F19 (Y5 crowding) confirms super-crowded days have +0.67%/fire (vs -0.05% normal), validating consensus as a SIZING signal.

## Stress-test budget (per F23 / POV-9 synthetic stress)

For ANY of the above deploy candidates, allocate capital with these stress-tolerance budgets:

| Scenario | V1 NAV impact | V7 NAV impact | maxDD V1 | maxDD V7 |
|---|---:|---:|---:|---:|
| BLACK_SWAN (-50% single day) | -209pp | -106pp | -46% | -45% |
| PROLONGED_BEAR (-2%/d × 30d) | -221pp | -112pp | -48% | -47% |
| HIGH_VOL (3σ × 60d) | +971pp | +676pp | -78% (p05) | -85% (p05) |

**Position sizing implications**:
- ~50% maximum drawdown is the realistic stress floor — only deploy capital that can absorb 50% drawdown
- Kelly fraction at point-mean Sharpe is OPTIMISTIC; use fractional Kelly (1/4 to 1/2)
- TRAIN's worst basket-day (-11.2% V1 / -9.3% V7) is a LOWER BOUND, NOT a representative tail — BLACK_SWAN scenarios at -37.8% basket-day are 3.4-4.1x worse

## STAY-OUT signal overlays (per F18 RE-VALIDATED + F19)

Apply these as overlays on top of any basket:

| Signal | Threshold | K-day stay-out | Lift | Applies to |
|---|---|---:|---:|---|
| UNI ac1_14 (autocorr) | > 0.4 | 10 days | 5.63x | UNI positions specifically |
| MOVR ac1_14 | > 0.4 | 10 days | 6.57x | MOVR positions |
| FLOKI ac1_14 | > 0.4 | 10 days | 4.19x | FLOKI positions |
| Funding-extreme | abs(funding z) > 2 | same-day | flips pnl +0.07 → -0.07 | All deploys |
| Whale-extreme | abs(wh_whale_net_usd z) > 2 | same-day | +0.78%/fire vs +0.11% normal | INCREASE sizing |

## Pre-deploy CHECKLIST (must pass ALL before capital)

- [ ] v3-paper-trade-replay on V7 + V1+consensus-weighted on (2024-05-16 to 2026-04-30) UNSEEN window
- [ ] Confirm OOS deflation matches the bootstrap CI (V7 expected +0.31%/d, V1 expected +0.17%/d)
- [ ] If V1 OOS p05 Sharpe < 1.0 — DO NOT deploy V1 standalone; use V7 or hybrid
- [ ] Validate engine duplicates: dedup V7's FIL liq_long_usd op_abs/op_gt and FET rv_bpv_5m op_abs/op_gt
- [ ] Add chop / bear regime coverage (current catalog catch-tier bear = 21 engines, 0 stable; explicit bear mining needed)
- [ ] Address 5 FAILED stages in catalog rebuild (chimera_legacy, s3, liq_features, etf_flows) before deploying
- [ ] Set per-asset exposure cap: V7 has FET at 4/12 = 33% — cap at 20% in deploy
- [ ] Set basket max_dd alert at -40% (slightly inside TRAIN -47% maxDD)
- [ ] Confirm cost model: 24bp taker cost assumed. Maker rebate (5bp) would lift returns 0.12pp/d but requires p_fill confidence (per MakerCostModel invariants: empirical p_fill 0.21-0.40 vs default 0.80 — be conservative)

## What this session NOT-YET-VALIDATED

- Y1 microstructure-stress per-fire (VPIN/HBR analysis) — framework-only
- Y6 causal-inference / DO-calculus — framework-only
- Z2 queuing-theory capacity per (asset, hour) — framework-only
- Z4 frontier-ML contrastive residual — needs GPU, framework-only
- Multi-horizon (R) — partial in sim variants D/E/F
- 4h cadence (G) — needs different data, not done
- Exogenous events (Q) — needs event calendar
- Anomaly detection (V) — not done
- Deploy-readiness operational (S) — partial via F22 red-team

## Provenance trail

This document supersedes prior `DEPLOY_CANDIDATES_2026_05_22.md` (which used pre-data-bug audit and named PEPE Donchian / different engines). The current candidates use the FIXED close-derived returns and the 6-POV mining expansion (Y2/Y3/Y4/Y5/Z1/Z3).

Source files:
- `runs/audit/MINING_FRAMEWORK_2026_05_23/EMERGENT_STORY_FINAL.md` (full synthesis, 1000+ lines)
- `runs/audit/MINING_FRAMEWORK_2026_05_23/HEADLINE_1PAGER.md` (executive summary)
- `data/oracle/sim_decoupling_audit_FIXED.md`, `sim_all_baskets_fixed.md`, `sim_v7_safe_basket.md`, `sim_consensus_weighted_basket.md`, `synthetic_stress_basket_report.md`
- `data/oracle/engine_lifecycle_decay.md`, `engine_antifragility_audit.md`, `beta_disguised_engines_FIXED.md`, `red_team_vulnerable_engines.md`
- `data/oracle/critical_phenomena_FIXED.md`, `crowding_taxonomy.md`, `listwise_topk_summary_FIXED.md`, `engine_consensus_signal_FIXED.md`
- `data/oracle/DATA_BUG_target_return_1_raw_2026_05_23.md` (provenance of bug + workaround)
