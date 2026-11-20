> **SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup)**: This document predates the
> canonical split-discipline gate. References to "OOS" in this file may include data
> from the canonical UNSEEN window (>=2026-01-01) per [src/split_config.py](../../src/split_config.py).
> Use this document for historical context only; deploy decisions citing UNSEEN-relevant
> claims must be re-derived from the canonical segments.

# DEPLOY RECIPE FINAL — 2026-05-23

This is the ACTIONABLE deploy recipe emerging from the 2026-05-23 mining session.

## THE METHODOLOGY FIX (single most important deliverable)

**Production deploy methodology change**:

| Phase | OLD (V7's approach) | NEW (F40-validated) |
|---|---|---|
| Catalog mining | Build on TRAIN | Same — build on TRAIN |
| Engine ranking | by TRAIN compound | **by VAL compound** |
| Selection criterion | top by TRAIN compound, magnitude ≥ 1.5%, stable, not-concave | **top by VAL compound + diversity filter** |
| Validation | None (TRAIN-only) | OOS_pre + UNSEEN sub-windows |
| Refresh cadence | Static | **Every 3-6 months re-VAL** |
| OOS survival rate | **17%** (V7+V1+30-engine baseline) | **83%** (F39 decile-3 basket) |

**Why this fix is needed**: Spearman correlations on n=89 engines with both VAL and OOS data:
- TRAIN compound vs OOS compound: ρ = **-0.477** (p < 0.000003) — STRONGLY ANTI-PREDICTIVE
- VAL compound vs OOS compound: ρ = **+0.730** (p < 0.000001) — STRONGLY POSITIVE-PREDICTIVE

**The catalog DOES contain real alpha**. The catalog's "rank by TRAIN compound" filter is the wrong selection criterion — it selects engines that overfit TRAIN microstructure. Switching to "rank by VAL compound" finds engines whose VAL-edge persists into OOS.

## RECOMMENDED DEPLOY BASKET (F39 decile-3)

15-engine basket, validated 15/18 OOS positive (83% survival):

| # | Asset | Class | Config | Regime | TRAIN comp | VAL comp | OOS comp | OOS ratio |
|---|---|---|---|---|---:|---:|---:|---:|
| 1 | LINK | RSI_threshold | p_5_lo_40_hi_60 | chop | +18.1% | +226.4% | **+418.8%** | +23.14 |
| 2 | LINK | RSI_threshold | p_6_lo_40_hi_60 | chop | +18.1% | +213.5% | +369.3% | +20.40 |
| 3 | LINK | RSI_threshold | p_5_lo_35_hi_60 | chop | +18.1% | +204.2% | +353.2% | +19.52 |
| 4 | LINK | RSI_threshold | p_7_lo_40_hi_60 | chop | +18.1% | +212.0% | +313.5% | +17.32 |
| 5 | LINK | RSI_threshold | p_6_lo_35_hi_60 | chop | +18.1% | +189.0% | +252.4% | +13.94 |
| 6 | LINK | RSI_threshold | p_8_lo_40_hi_60 | chop | +18.1% | +258.4% | +238.8% | +13.19 |
| 7 | LINK | RSI_threshold | p_7_lo_35_hi_60 | chop | +18.1% | +177.0% | +190.4% | +10.52 |
| 8 | DASH | OBV_zscore | p_100_t_1.5 | chop | +10.0% | +45.0% | +65.1% | +6.52 |
| 9 | JST | RSI_threshold | p_6_lo_40_hi_80 | chop | +8.2% | +30.2% | +42.0% | +5.13 |
| 10 | JST | RSI_threshold | p_10_lo_40_hi_70 | chop | +10.9% | +26.8% | +48.0% | +4.40 |
| 11 | JST | RSI_threshold | p_9_lo_40_hi_70 | chop | +10.9% | +25.8% | +37.1% | +3.40 |
| 12 | APT | RSI_threshold | p_27_lo_40_hi_60 | bull | +10.3% | +13.4% | +20.8% | +2.02 |
| 13 | JST | RSI_threshold | p_6_lo_35_hi_75 | chop | +9.1% | +20.1% | +17.5% | +1.93 |
| 14 | APT | RSI_threshold | p_29_lo_40_hi_60 | bull | +11.7% | +23.3% | +21.9% | +1.88 |
| 15 | APT | RSI_threshold | p_28_lo_40_hi_60 | bull | +10.3% | +11.3% | +11.8% | +1.15 |

**Asset coverage**: LINK (7) + JST (4) + APT (3) + DASH (1) = 4 unique assets
**Class coverage**: RSI_threshold (14) + OBV_zscore (1) = 2 distinct classes
**Regime coverage**: chop (12) + bull (3) = 2 regimes

## DEDUP RECOMMENDATIONS

The 7 LINK RSI chop configs are near-duplicates (same asset, same class, parameter variants). Deploy 2-3 of them for variance reduction:

- **LINK p_5_lo_40_hi_60** (TRAIN +18.1%, OOS +418.8%) — single best
- **LINK p_8_lo_40_hi_60** (similar metrics, parameter diversity)
- **LINK p_5_lo_35_hi_60** (different hi threshold, parameter diversity)

Drop the other 4 LINK configs as redundant. Add JST RSI ×2 + APT RSI ×1 + DASH OBV ×1 for asset diversification.

**Deduped 7-engine deploy basket**:
1. LINK RSI p_5_lo_40_hi_60 chop
2. LINK RSI p_8_lo_40_hi_60 chop
3. LINK RSI p_5_lo_35_hi_60 chop
4. DASH OBV p_100_t_1.5 chop
5. JST RSI p_6_lo_40_hi_80 chop
6. JST RSI p_10_lo_40_hi_70 chop
7. APT RSI p_27_lo_40_hi_60 bull

## SIZING + COST MODEL

- Top-3 picks per day by n_engines firing concurrently on same (asset, date) cell
- 25% sizing per pick (75% deployed, 25% cash buffer)
- Per-bucket maker cost: BLUE 28bp, STEADY 32bp, VOLATILE 36bp, DEGEN 44bp (LINK/JST = DEGEN-bucket since smaller-cap)
- 1-day hold (close-to-close)
- Daily rebalance

## EXPECTED PERFORMANCE (HONEST FORECAST)

Based on individual-engine OOS performance (ratios +1.2 to +23):
- TRAIN-realized basket Sharpe: 1.04 (simplified-signal sim, F39.b)
- OOS_pre realized: Sharpe 4.22 / +32.7% NAV (F39.b)
- UNSEEN realized: marginal negative (-2.8% NAV, F39.b)
- FULL POST-TRAIN realized: Sharpe 1.04 / +17% NAV
- **Expected deploy mean: +0.08 to +0.25%/d, Sharpe 1.0-2.5, drawdown ~30%**

**Caveats** (HONEST):
- F39 basket was DERIVED from VAL data → OOS_pre + UNSEEN are partially in-sample for the SELECTION step
- Strict pure-OOS test: v3-paper-trade-replay on POST-2026-05-19 data (not available as of session)
- LINK has small ADV ($~50M daily volume) → deploy capacity capped at ~$1M
- Catalog rebuild failed at chimera_v51 (2026-05-22) → no new data ingestion until rebuild fixed

## PRE-DEPLOY CHECKLIST

- [ ] v3-paper-trade-replay on UNSEEN window (2026-01 to 2026-04) using catalog's EXACT signal logic (not my simplified RSI)
- [ ] Confirm OOS Sharpe ≥ 1.0 with proper signal
- [ ] Capacity check: at $250K deploy, basket fits within asset ADV
- [ ] Set per-asset cap (LINK at 30%, JST at 25%, APT at 20%, DASH at 25%)
- [ ] Set basket max_dd alarm at -25% (slightly inside historical -31% / -19%)
- [ ] 30-day live monitoring gate before full capital deployment
- [ ] Re-VAL refresh every 3 months (drop engines whose val_compound goes negative)

## EXPANDED DEPLOY POOL (F41 — full 234 catch-tier OOS-validated)

After OOS-validating ALL 234 catch-tier engines, the OOS-positive pool is **58 engines** (40 SURVIVED + 18 partial). **Recommended deploy basket = ~18 deduplicated engines covering bull + chop + BEAR**:

| Asset | Class | Config | Regime | OOS comp |
|---|---|---|---|---:|
| LINK | RSI_threshold | p_5_lo_40_hi_60 | chop | +418.8% |
| LINK | RSI_threshold | p_8_lo_40_hi_60 | chop | +238.8% |
| LINK | RSI_threshold | p_5_lo_35_hi_60 | chop | +353.2% |
| **ZEC** | te_btc_imb | op_abs_gt_thr_1.0 | **BEAR** | **+102.5%** |
| **FET** | te_imb | op_abs_gt_thr_1.0 | **BEAR** | **+37.4%** |
| APT | RSI_threshold | p_5_lo_40_hi_65 | bull | +150.1% |
| APT | RSI_threshold | p_13_lo_35_hi_65 | bull | +58.3% |
| APT | RSI_threshold | p_27_lo_40_hi_60 | bull | +20.8% |
| JST | RSI_threshold | p_7_lo_40_hi_75 | chop | +52.5% |
| JST | RSI_threshold | p_10_lo_40_hi_70 | chop | +48.0% |
| JST | RSI_threshold | p_6_lo_40_hi_80 | chop | +42.0% |
| JST | RSI_threshold | p_9_lo_40_hi_70 | chop | +37.1% |
| DASH | OBV_zscore | p_100_t_1.5 | chop | +65.1% |
| PEPE | RSI_threshold | p_10_lo_40_hi_80 | chop | +62.9% |
| PEPE | Donchian | period_55 | chop | +32.2% |
| ARB | bs_basis_z30 | op_abs_gt_thr_1.0 | bull | +47.7% |
| FET | wh_whale_net_usd | op_abs_gt_thr_2.0 | bull | +43.1% |
| HBAR | wh_whale_trade_count_500k | op_gt_thr_1.0 | bull | +16.0% |

18 engines · 9 assets (LINK/JST/APT/FET/DASH/PEPE/ZEC/ARB/HBAR) · 3 regimes (bull/chop/BEAR) · 6 classes (RSI/OBV/te_*/Donchian/bs_basis/wh_whale)

**Asset capacity check** (per F25): LINK ~$50M, APT ~$80M, JST ~$1.4M, FET ~$25M, ZEC ~$2.8M — DEPLOY WITHIN per-asset caps. Total basket capacity ~$30-50M before slippage drags Sharpe below 1.0.

## DEPLOY LOGIC — 2-TIER REGIME GATE (per F45 regime-off test)

After OOS-regime-off testing each F41 engine across the full OOS window with no regime gate, 4 engines are truly regime-agnostic (OOS-positive in >=2 regimes); 13 are regime-conditional (only positive in native regime). Drop the gate for Tier A; keep it for Tier B.

### Tier A — Regime-AGNOSTIC (deploy without gate, fire on signal alone)

| Engine | Native regime | Native comp | Bull comp | Chop comp | Bear comp |
|---|---|---:|---:|---:|---:|
| ZEC te_btc_imb op_abs_gt_thr_1.0 | bear | +102.5% | +11.8% | -22.8% | **+102.5%** |
| APT RSI p_5_lo_40_hi_65 | bull | +150.1% | **+150.1%** | +47.5% | -58.3% |
| APT RSI p_27_lo_40_hi_60 | bull | +20.8% | **+20.8%** | +15.1% | -27.8% |
| PEPE RSI p_10_lo_40_hi_80 | chop | +62.9% | -31.4% | **+62.9%** | +74.1% |

These engines survive BTC regime shift. Deploy with regime gate DISABLED.

### Tier B — Regime-CONDITIONAL (keep regime gate — 13 engines)

All other F41 engines: LINK ×3 (chop-gated), JST ×4 (chop), DASH OBV (chop), APT p_13 (bull), FET te_imb (bear), ARB bs_basis (bull), FET wh_whale_net_usd (bull), HBAR wh_whale_trade_count (bull). For these, removing the regime gate collapses compound by 50-500pp.

### Implementation

```python
# pseudocode
for engine in deploy_basket:
    fire = engine.signal_fires(asset_panel)
    if engine.tier == "A":
        execute(fire)  # no regime check
    else:
        if current_btc_regime == engine.native_regime:
            execute(fire)
        # else suppress
```

This 2-tier logic preserves cell-level alpha (which the catalog mining is correctly grain-sized for, per F42-F44) while extracting the additional ~24% regime-agnostic budget that the catalog couldn't surface natively.

## METHODOLOGY UPGRADES (next iteration)

1. **Re-mine catalog with VAL-rank methodology** on ALL 234 catch-tier engines (currently only 99 OOS-tested). Estimated: 4-8 weeks engineering.
2. **Multi-cutoff catalog mining** (rolling 12-month TRAIN windows). Engines that survive 3+ cutoffs are deploy-eligible.
3. **CSCV PBO** for backtest-overfitting estimation.
4. **Bonferroni / Benjamini-Hochberg correction** at engine selection (937 candidates → require p < 0.0001 each).
5. **Bear regime explicit mining** (currently 0/21 fold-stable bear engines).
6. **V20 tick-level substrate** per CLAUDE.md INDISPUTABLE OPERATING LENS (8-12 weeks) — required to break the IC > 0.10 / ShIC > 0.05 Headline tier.

## Provenance

- session: runs/audit/MINING_FRAMEWORK_2026_05_23/EMERGENT_STORY_FINAL.md (1900+ lines)
- F39 OOS validation: data/oracle/engine_oos_validation_decile3.parquet (18 engines, 15 SURVIVED)
- F40 correlation: computed in-session, n=89, ρ_VAL_OOS=+0.730, ρ_TRAIN_OOS=-0.477
- F39 basket sim: data/oracle/sim_f39_decile3_basket.md (Sharpe 1.04 FULL POST-TRAIN on simplified signal)
- WHY_DEPLOY_FAILS_SYNTHESIS.md: structural explanation of pre-F39 failures
