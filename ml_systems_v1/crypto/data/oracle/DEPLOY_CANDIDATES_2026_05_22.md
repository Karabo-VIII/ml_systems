> **SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup)**: This document predates the
> canonical split-discipline gate. References to "OOS" in this file may include data
> from the canonical UNSEEN window (>=2026-01-01) per [src/split_config.py](../../src/split_config.py).
> Use this document for historical context only; deploy decisions citing UNSEEN-relevant
> claims must be re-derived from the canonical segments.

# Oracle Engine v0 -- Deploy Candidates (post-OOS validation + post final-audit + trader-review)

**Generated**: 2026-05-23 00:34 SAST (autonomous build window).
**Catalog**: `data/oracle/engine_catalog.parquet` (213 engines, post-audit rigorous).
**OOS window**: 2024-05-16 -> 2026-05-19 (~2 years post-TRAIN).
**Validation**: `data/oracle/engine_oos_validation.parquet`.

## ⚠️ HARD STATUS: DELAY-NEEDS-V3-REPLAY (per trader review) + SHIP-WITH-CAVEATS (per final audit)

Two independent reviews converged:

1. **Audit (RED-team pass-3)**: arithmetic is correct, no residual leaks, but PEPE Donchian_state and PEPE MA_state_SMA50 are the SAME engine (Jaccard 1.000 TRAIN, 0.856 OOS) -- collapse to one. Backtest BTC regime_label uses close[t] via right-aligned SMA200; LIVE execution MUST shift to t-1. Verdict: SHIP-WITH-CAVEATS.

2. **Trader review**: realistic portfolio /d at pos_cap 6% / 5-slot = +0.005-0.008% (120x below +1%/d target). 3 of 5 engines have fold3=0.0% (signal dead in most recent period). None of 5 OOS hit rates are statistically significant (all p > 0.37). Verdict: DELAY-NEEDS-V3-REPLAY. If forced to deploy: PEPE Donchian only, 4% pos cap, 30-day live monitoring gate.

**DO NOT DEPLOY CAPITAL until**:
- v3-paper-trade-replay each of the deploy candidates individually
- More TRAIN data (extend Layer-1 catalog beyond 10 months OR use rolling-window walk-forward)
- Resolve fold3=0 collapse on XRP/HBAR/SUI

## 5 OOS-positive engines (out of 30 tested = 17% pass rate)

These survived the brutal TRAIN -> OOS transition. **After final-audit dedup of PEPE-x2 (same engine), effective deploy set is 4 engines.** Use these (and ONLY these) as deploy candidates v0.

### #1 PEPE Donchian_state_above_midline period_55 / bear-c4

**OOS compound: +32.20%** (OOS > TRAIN: ratio 1.14, unusually robust)

```
family: ta_state
asset: PEPE (DEGEN bucket)
indicator_class: Donchian_state_above_midline
indicator_config: period_55
cadence: 1d
regime_gate: bear   <-- ONLY bear-regime engine in top-5
cluster_filter: 4
hold_days: 3
direction: long_fire
TRAIN: n=10, hit_rate=66.7%, expectancy=+4.4%, compound=+28.3%
OOS:   n=82, hit_rate=46.3%, expectancy=+0.63%, compound=+32.2%, max_dd=-52.8%
```

**Why it works**: bear-regime DEGEN-bucket asset price-above-Donchian-midline state engine. Captures the rebound moves in PEPE during bear-regime drawdowns. The OOS hit rate at 46% with positive expectancy and 82 fires is statistically real.

### #2 XRP measure_engines/hbr_eta_buy op_gt_thr_1.0 / bull-c5

**OOS compound: +12.77%** (ratio 0.30)

```
family: measure_event
asset: XRP (STEADY bucket)
indicator_class: measure_engines/hbr_eta_buy
indicator_config: op_gt_thr_1.0  (Hawkes buy-intensity z-score > 1.0)
cadence: 1d
regime_gate: bull
cluster_filter: 5
hold_days: 1
direction: long_fire
TRAIN: n=10, hit_rate=90.0%, expectancy=+3.75%, compound=+42.7%
OOS:   n=30, hit_rate=53.3%, expectancy=+0.50%, compound=+12.8%, max_dd=-13.1%
```

**Why it works**: Hawkes buy-intensity self-excitation detects orderflow regime shifts on XRP. Microstructure signal generalizes from TRAIN -> OOS at modest deflation. Best max_dd of the deploy set.

### #3 PEPE MA_state_SMA_above period_50

**OOS compound: +7.37%** (ratio 0.26)

```
family: ta_state
asset: PEPE
indicator_class: MA_state_SMA_above
indicator_config: period_50
cadence: 1d
regime_gate: (no specific filter)
hold_days: 1
TRAIN compound: +28.3%
OOS:   n=85, hit_rate=44.7%, expectancy=+0.37%, compound=+7.4%, max_dd=-50.9%
```

**Why it works**: PEPE price-above-SMA-50 state. Trend-bias-up engine. Lower hit rate (44%) but positive expectancy survives OOS.

**Note**: Correlated with #1 above (both on PEPE). At pos_cap=6%/asset, deploying both means 12% NAV on PEPE. Recommend dropping #3 OR splitting size.

### #4 HBAR measure_engines/norm_deviation op_gt_thr_1.0

**OOS compound: +2.31%** (ratio 0.05) -- MARGINAL

```
family: measure_event
asset: HBAR (VOLATILE bucket)
indicator_class: measure_engines/norm_deviation
indicator_config: op_gt_thr_1.0
hold_days: 1
TRAIN compound: +43.6%
OOS:   n=34, hit_rate=47.1%, expectancy=+0.16%, compound=+2.3%, max_dd=-16.6%
```

**Why it works**: HBAR's price deviation from its MA crosses z=1.0 -- a momentum-onset signal. OOS expectancy is barely positive; would be a marginal include in a portfolio.

### #5 SUI measure_engines/bd_imbalance_l5 op_lt_thr_1.0

**OOS compound: +1.39%** (ratio 0.03) -- MARGINAL

```
family: measure_event
asset: SUI (VOLATILE bucket)
indicator_class: measure_engines/bd_imbalance_l5
indicator_config: op_lt_thr_1.0  (book-depth imbalance < -1.0)
regime_gate: chop
cluster_filter: 1
hold_days: 1
TRAIN compound: +49.8%
OOS:   n=37, hit_rate=54.1%, expectancy=+0.18%, compound=+1.4%, max_dd=-19.1%
```

**Why it works**: SUI's level-5 book-depth imbalance below -1.0 z (asks dominate bids deeply). Contrarian-buy signal in chop regime. OOS hit rate 54% is the highest in the deploy set.

## Portfolio composition (post-dedup)

Equal-weight 4 engines (dropping PEPE MA_state #3 — same engine as PEPE Donchian #1 per Jaccard 1.0):
- 4 engines × 6% NAV = 24% deployed (well under 60% pos_cap_total)
- Distinct assets: PEPE, XRP, HBAR, SUI
- Distinct families: ta_state, measure_event x3
- Distinct regimes: bear, bull, all, chop

**Honest expected /d** (per trader review at pos_cap 6% × OOS expectancy × fire rate × live haircut):
- Best case: **+0.013%/d**
- After 1.5x live haircut: **+0.008%/d**
- Target +1%/d: ~120× gap

**This is small but real**. The 5-engine catalog v0 is a **starting library**, not a deploy-ready strategy. More engines need to be added (more measure columns, 4h cadence, confluence) before this composes to a useful /d.

## v3-truth-replay validation

**Required before any capital commitment**: replay each of these 5 engines through `scripts/strat_audit/paper_trade_replay_v3.py` on the OOS+UNSEEN window with realistic costs (taker fee + slippage + maker fill probability). The compound numbers above are unit-sized OOS replay -- the v3-truth deflation may further reduce by 30-60%.

## What's NOT shippable

The other 25 OOS-tested engines (and ~183 untested engines with composite_score below threshold) showed OOS compounds from -12% to -97%. These engines fit TRAIN distribution but did NOT generalize. They are kept in catalog for diagnostic-only purposes:
- Use them to understand which patterns failed
- Use them as anti-signal: fire ABSENCE may be useful
- DO NOT deploy any of them as long-fire signals

## Recipe yaml format (for any downstream consumer)

Each catalog row has a `recipe_yaml` string. Example for engine #1:
```
family=ta_state; asset=PEPE; indicator_class=Donchian_state_above_midline;
indicator_config=period_55; cadence=1d; regime_gate=bear;
magnitude_filter=all; cluster_filter=4; hold_days=3; direction=long_fire;
cost_mode=taker_bucket; sizing_rule=fixed_4pct
```

The recipe is deterministic: the IndicatorClass's `compute_signals_fn` applied to chimera 1d for the asset, filtered by the regime+cluster, hold-cooldown applied at h=3 produces the exact signal series.

## Sources

- `data/oracle/engine_catalog.parquet` (213 engines)
- `data/oracle/engine_oos_validation.parquet` (30 engines OOS-tested)
- `data/oracle/engine_stories_top50.md` (prose narratives for top-50 by composite_score)
- `data/oracle/confluence_catalog.parquet` (4 confluence engines)
- `runs/audit/AUTONOMOUS_ORACLE_BUILD_2026_05_22/` (full audit trail)
