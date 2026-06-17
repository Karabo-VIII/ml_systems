# V4 Crypto System -- End-to-End Runbook

Master execution guide. Run in order. Each phase depends on the previous.

## TL;DR — Cold-start to first V1 baseline (PowerShell)

```powershell
# 0. Pipeline status — see what's already usable
python src/pipeline/run_pipeline.py --status

# 1. Run pipeline DAG end-to-end (resumable, skips stale stages)
python src/pipeline/run_pipeline.py --universe u10 `
    --stage-workers chimera_legacy=4 `
    --stage-workers chimera_v51=4 `
    --stage-workers frontier_consolidate=8

# 2. Pre-train CI gate (5 validators)
python src/pipeline/pre_train_gate.py --asset BTC

# 3. V0 baselines (IC floor / ceiling) — sets the bar V1.x must beat
foreach ($f in 13, 25, 29, 41, 60, 121, 127) {
  python src/wm/v0/v0_baseline/linear_baseline.py    --features $f --workers 8
  python src/wm/v0/v0_baseline/nonlinear_baselines.py --features $f --workers 4
}

# 4. V1.0 reference baseline (4-6h on RTX 4060)
python src/wm/v1/v1_0_training/train_world_model.py --features 13
```

> Note: each `--stage-workers KEY=N` requires its own flag (argparse uses `action="append"`). The trailing backtick is PowerShell line-continuation.

---

## Phase 0: Data Pipeline (DAG runner — `run_pipeline.py`)

The pipeline is a 7-stage DAG with per-stage usability tracking + resumability. Each stage is `is_stale`-checked against its inputs; only out-of-date stages re-run.

**Two orthogonal CLI knobs** (don't confuse them):
- `--universe {u10|u50|u100}` — which **assets** to operate on (config/universes/*.yaml)
- `--tiers <stage1,stage2,...>` — which **stages** to run (`fetch_binance`, `bar_fabric`, `hawkes_branching`, `frontier_consolidate`, `chimera_legacy`, `chimera_v51`, `validate`, `gc_snapshots`); default `all`

```powershell
# Show what's ready right now (safe to run anytime)
python src/pipeline/run_pipeline.py --status

# Cold-start: run all stages for the u10 universe (parallel where safe)
python src/pipeline/run_pipeline.py --universe u10 `
    --stage-workers chimera_legacy=4 `
    --stage-workers chimera_v51=4 `
    --stage-workers frontier_consolidate=8

# Cold-start: full u50 (production deployable universe — ~6-12h)
python src/pipeline/run_pipeline.py --universe u50 `
    --stage-workers chimera_legacy=4 `
    --stage-workers chimera_v51=4 `
    --stage-workers frontier_consolidate=8

# Selective re-run: rebuild only chimera + validate stages (any universe)
python src/pipeline/run_pipeline.py --universe u10 `
    --tiers chimera_v51,chimera_legacy,validate

# Continue past failures (don't abort on first per-asset error)
python src/pipeline/run_pipeline.py --universe u10 --continue-on-fail
```

**Stage-by-stage usability** (each stage produces something the next consumer can use immediately):

| Tier | Stage | Output location | Usable for |
|---|---|---|---|
| T0 | fetch_binance | `data/raw/<SYM>/` | bar_fabric, dollar bars, tick-level analysis |
| T1 | bar_fabric (`build_bars.py`) | `data/processed/bars/<bartype>/` | bar-type strategies (DIB flow, runs scalping) |
| T1 | hawkes_branching | `data/processed/hawkes/` | ranker feature, microstructure analysis |
| T2 | build_panels | `data/processed/panels/daily/` | s3 / basis / liquidations / whale / top_trader / etf panels — inputs to frontier_consolidate |
| T3 | frontier_consolidate | `data/processed/frontier/` | xsec ranker, daily strategies, v51 frontier features |
| T4 | chimera_legacy | `data/processed/chimera_legacy/dollar/` | **V0 + V1.x training (f13-f41)**, V2-V14 inference |
| T5 | chimera_v51 | `data/processed/chimera/dollar/` | **V0 + V1.x training (f46-f121)**, ChimeraLoader, full strategy stack |
| T6 | validate | `data/_manifests/_pipeline_state.json` | pre-train CI gate; 5 validators |
| T7 | gc_snapshots | (deletes older dated files) | disk reclaim; default-on after validate, column-aware |

**Pre-train CI gate** — composes 5 validators (data_health, chimera_v51, xd_consistency, e2e, split). Per-asset only; loop to cover a universe:

```powershell
python src/pipeline/pre_train_gate.py --asset BTC           # exit 0 = clean, 1 = warn, 2 = hard fail
python src/pipeline/pre_train_gate.py --asset BTC --quick   # skip slow raw scan

# Gate the full u10 (loop because the gate is per-asset; stop on first failure)
foreach ($a in 'BTC','ETH','SOL','BNB','XRP','DOGE','ADA','AVAX','LINK','LTC') {
  python src/pipeline/pre_train_gate.py --asset $a --quick
  if ($LASTEXITCODE -ne 0) { break }
}
```

**GC dated snapshots — runs by default after validate.** Every pipeline run writes a new `<sym>usdt_v50_chimera_<YYYYMMDD>.parquet` (or `_v51_*`); without GC they would accumulate. GC keeps the newest 1 *valid* snapshot per asset.

Why this is safe — and why the default is `keep=1`, not `keep=2`:
1. **Frozen split dates** (`config/data_config.yaml`) mean train/val/oos segments are byte-identical run-to-run; only the unseen segment grows by a few bars/day. Older snapshots add no training information.
2. **Column-aware validity check**: if today's snapshot is missing required features (partial build), GC treats it as invalid and preserves an older healthy snapshot automatically. No risk of deleting your fallback.

```powershell
# Default: GC runs automatically after validate stage passes
python src/pipeline/run_pipeline.py --universe u10

# Opt out of auto-GC (snapshots accumulate)
python src/pipeline/run_pipeline.py --universe u10 --no-gc

# Standalone GC tool (preview + apply)
python src/pipeline/gc_snapshots.py --dry-run
python src/pipeline/gc_snapshots.py                    # default keep=1
python src/pipeline/gc_snapshots.py --layer chimera_legacy
python src/pipeline/gc_snapshots.py --keep 2           # extra-conservative
```

See: [src/pipeline/USAGE.md](src/pipeline/USAGE.md), [src/pipeline/README.md](src/pipeline/README.md)

---

## Phase 1: V0 Baselines (~1-3h depending on sweep width)

Non-DL baselines that define the IC floor (linear ridge) and ceiling (poly + GBT + MLP). Every world model must beat these to justify its compute cost. Both scripts auto-switch source: `f13-f41` reads `chimera_legacy/dollar/`, `f46-f121` reads `chimera/dollar/` (v51 frontier).

```powershell
# Single feature count (recommended first run — cheap sanity check)
python src/wm/v0/v0_baseline/linear_baseline.py    --features 41 --workers 8   # ~5-10 min
python src/wm/v0/v0_baseline/nonlinear_baselines.py --features 41 --workers 4   # ~15-25 min

# Full sweep across V1.x boundaries — sets IC floor for every feature count we'll train
foreach ($f in 13, 25, 29, 41, 60, 121, 127) {
  python src/wm/v0/v0_baseline/linear_baseline.py    --features $f --workers 8
  python src/wm/v0/v0_baseline/nonlinear_baselines.py --features $f --workers 4
}
```

**Useful flags** (both scripts share most):
- `--features N` — feature count (13/18/25/29/34/41 from v50, 46/60/73/78/81/84/97/110/121 from v51)
- `--workers N` — process-pool parallelism for ridge/poly/GBT/MLP fits
- `--full` — single 90/10 split (skip walk-forward purge — faster, less rigorous)
- `--model {poly,gbt,mlp}` — nonlinear: pick a single model instead of all three

**Interpretation**:
- DL IC < Linear IC → DL is overfitting; abandon that architecture.
- DL IC ~ Nonlinear IC → temporal modeling adds no value; the gain comes from feature interactions, not sequence.
- DL IC > Nonlinear IC by ≥30% → temporal architecture earns its keep.
- Linear IC plateau across feature counts → the "extra" features aren't pulling weight; pick the smallest count that hits the plateau.

**Output**: `logs/v0/v0/linear_baseline_results.txt`, `logs/v0/v0/nonlinear_*.txt`

See: [src/wm/v0/USAGE.md](src/wm/v0/USAGE.md)

---

## Phase 2: V1 World Models (~4-6h per model)

Train in this order. Each model is ~4-6h on RTX 4060 at 10-asset u10 universe.

**Universe**: V0/V1.x are hardcoded to the canonical u10 (BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC) — see `ASSET_LIST` in each settings.py. To train at u50, edit `ASSET_LIST` + bump `NUM_ASSETS=50` (asset embedding cost: 10×32→50×32 = +1280 params, negligible against 5M total).

**Best results to date** (per CLAUDE.md, 2026-04 retraining cycle):
- V1.0 f34: IC=0.0660, ShIC=0.0320 PASS
- V1.1 f34: IC=0.0674, ShIC=0.0330 PASS — current ShIC record
- V1.4 f34: IC=0.0679, ShIC=0.0314 PASS
- V1.6 f34: IC=0.0619, ShIC=0.0329 PASS

### Step 2a: V1.0 f13 -- Reference Baseline (FIRST)

```powershell
python src/wm/v1/v1_0_training/train_world_model.py
```

Fixed 13-feature architecture. Must pass all gates before proceeding. This is the reference point for all other models.

**Gate check**: ShIC > 0.015, ShIC/IC ratio > 0.3, Recon MSE < 0.12

### Step 2b: V1.1 f13 -- Flexible Architecture Baseline

```powershell
python src/wm/v1/v1_1_training/train_world_model.py --features 13
```

Same features as V1.0 but with flexible architecture (XD anti-memorization, ablation support). Compare IC to V1.0 -- should be similar.

### Step 2c: V1.4 f13 -- FeatureAttention Ensemble Diversity

```powershell
python src/wm/v1/v1_4_training/train_world_model.py --features 13
```

iTransformer-style cross-feature attention. Different inductive bias provides ensemble diversity.

### Step 2d: V1.6 f13 -- Best-of-V1 (All Techniques)

```powershell
python src/wm/v1/v1_6_training/train_world_model.py --features 13
```

KL annealing + Gumbel tau schedule + ATME + dream consistency loss. Most complex V1 variant.

### Step 2e: Feature Scaling (after f13 baselines pass gates)

```powershell
# V1.1 with more features (pick one to start)
python src/wm/v1/v1_1_training/train_world_model.py --features 18
python src/wm/v1/v1_1_training/train_world_model.py --features 25
python src/wm/v1/v1_1_training/train_world_model.py --features 30
python src/wm/v1/v1_1_training/train_world_model.py --features 37

# V1.1 CRPS A/B test (compare to ce baseline)
python src/wm/v1/v1_1_training/train_world_model.py --features 13 --loss-type crps

# Ablation run (measures marginal feature contribution)
python src/wm/v1/v1_1_training/train_world_model.py --features 37 --ablation
```

### Step 2f: V1.E Ensemble (after 2+ V1 models pass gates)

```powershell
# Validate ensemble performance
python src/wm/v1/validate_ensemble.py

# Custom model selection
python src/wm/v1/validate_ensemble.py --models v1_0 v1_1_f13 v1_4_f13
```

See: [src/wm/v1/USAGE.md](src/wm/v1/USAGE.md)

---

## Phase 3: V2-V9 World Models (~2-6h per model)

Train base models only (no variants until base passes gates). Always start with f13.

**Priority order** (highest to lowest):

| Priority | Version | Architecture | Why |
|----------|---------|-------------|-----|
| 1 | V4 | Mamba SSM | Linear complexity, selective scan -- different inductive bias |
| 2 | V3 | WaveNet-GRU | Dilated causal convolutions -- multi-scale temporal patterns |
| 3 | V9 | Mixture-of-Experts | 3 regime-gated experts -- regime specialization |
| 4 | V5 | Hybrid Mamba-Attention | Mamba + local windowed attention |
| 5 | V2 | JEPA + VICReg | Contrastive learning (low priority, different paradigm) |
| 6 | V7 | Vision Transformer | 2D patch processing (experimental) |
| 7 | V6 | JEPA + Adversarial | Time-shuffle discriminator (low priority) |
| 8 | V8 | Neural ODE | Continuous-time dynamics (slow, experimental) |

### Training Commands (base models, always start with f13)

```powershell
# V4 Mamba (highest priority) -- supports --features 13/18/30/37
python src/wm/v4/v4_training/train_world_model.py --features 13

# V3 WaveNet
python src/wm/v3/v3_training/train_world_model.py

# V9 MoE
python src/wm/v9/v9_training/train_world_model.py

# V5 Hybrid
python backups/BKP_20260429_MODEL_HARMONIZATION/v5/v5_training/train_world_model.py

# V2 JEPA (lower priority)
python backups/BKP_20260429_MODEL_HARMONIZATION/v2/v2_training/train_world_model.py

# V7 ViT (lower priority)
python backups/BKP_20260429_MODEL_HARMONIZATION/v7/v7_training/train_world_model.py

# V6 JEPA+Adv (lower priority)
python src/wm/v6/v6_training/train_world_model.py

# V8 Neural ODE (lowest priority)
python src/wm/v8/v8_training/train_world_model.py
```

### V4 Variants (after V4 base passes gates)

```powershell
# FiLM Adapter
python src/wm/v4/v4_training/train_adapter.py --features 13

# Snapshot Ensemble
python src/wm/v4/v4_training/train_snapshot.py --features 13

# NCL Diversity
python src/wm/v4/v4_training/train_ncl.py --features 13

# Validation
python src/wm/v4/v4_training/validate_world.py --features 13
```

**NOTE**: V4 base now supports flexible `--features` with feature-tagged checkpoints and collision guard. V2-V9 variant models (.1/.2/.3) have stale `get_feature_config()` that only maps 13/17/18/22. Fix those settings before training variants with new feature counts.

See per-version docs: [src/wm/v4/USAGE.md](src/wm/v4/USAGE.md), etc.

---

## Phase 4: V1 Variants (only after base passes gates)

These are lightweight add-ons to a frozen base model.

### Snapshot Ensemble (.E) -- Priority 1

Cyclical cosine LR creates diverse snapshots from a single training run.

```powershell
# V1.1 snapshot ensemble
python src/wm/v1/v1_1_training/train_snapshot.py --features 13

# V1.4 snapshot ensemble
python src/wm/v1/v1_4_training/train_snapshot.py --features 13
```

### FiLM Adapter (.X) -- Priority 2

~5-25K params fine-tuned on frozen base. Fast to train.

```powershell
python src/wm/v1/v1_1_training/train_adapter.py --features 13
```

### NCL Diversity (.D) -- Priority 3 (Lowest)

K=5 parallel return prediction heads with negative correlation learning.

```powershell
python src/wm/v1/v1_1_training/train_ncl.py --features 13
```

---

## Phase 5: Analysis & Backtesting

Requires trained checkpoints from Phase 2+.

### Strategy Evaluation (no WM needed)

```powershell
# Full multi-strategy sweep across all 10 assets
python src/analysis/strategy_lab.py --sweep

# Live terminal replay with HTML report
python src/analysis/live_backtest.py --strategy donchian --asset btcusdt

# Time-calibrated Donchian analysis
python src/analysis/donchian_calibrated.py
```

### WM-Enhanced Strategies (requires V1.E ensemble)

```powershell
# WM-filtered sweep (requires trained ensemble)
python src/analysis/strategy_lab.py --sweep --wm-ensemble

# Live WM-filtered Donchian
python src/analysis/live_backtest.py --strategy wm_donchian_filter --asset btcusdt --wm-ensemble

# WM horizon analysis across checkpoints
python src/analysis/wm_horizon_analysis.py
```

### WM Signal Evaluation

```powershell
# Save GBT baseline predictions for WM comparison
python src/wm/v0/v0_baseline/save_baseline_preds.py

# WM backtest (14 WM signal strategies)
python src/analysis/wm_backtest.py
```

See: [src/analysis/USAGE.md](src/analysis/USAGE.md)

---

## Phase 6: Agent Training (after V1.E ensemble exists)

PPO trading agent using frozen world model predictions.

```powershell
# V1.E ensemble agent (recommended)
python src/agent/train_agent.py --ensemble --steps 2000000 --sav

# With stress augmentation
python src/agent/train_agent.py --ensemble --augment --steps 2000000 --sav

# Single model (lightweight test)
python src/agent/train_agent.py --world-model v1_0 --features 13

# Evaluate existing agent
python src/agent/train_agent.py --ensemble --eval-only --sav
```

See: [src/agent/USAGE.md](src/agent/USAGE.md)

---

## Phase 7: V10 Meta-Ensemble (after V1-V9 have passing checkpoints)

Aggregation-only layer -- no base training. Combines predictions from multiple world model versions.

```powershell
python src/wm/v10/train_meta.py
```

---

## Batch Training (one-shot per feature count)

Run all variants of a version serially with preflight checks. Failures are logged and skipped.

```powershell
# V1 family -- all 4 variants (V1.0, V1.1, V1.4, V1.6)
python src/wm/v1/run_training.py --features 13       # ~8-24h total
python src/wm/v1/run_training.py --features 18       # after f13 passes
python src/wm/v1/run_training.py --features 37       # full feature set

# V2-V9 -- all 4 sub-variants (base + FiLM + snapshot + NCL)
python src/run_version_training.py --version 4 --features 13   # V4 Mamba
python src/run_version_training.py --version 3 --features 13   # V3 WaveNet
python src/run_version_training.py --version 9 --features 13   # V9 MoE

# Base-only (skip variants, fastest)
python src/run_version_training.py --version 4 --features 13 --only base

# Dry run (preflight checks, no training)
python src/wm/v1/run_training.py --features 13 --dry-run
python src/run_version_training.py --version 4 --features 13 --dry-run

# Specific variants only
python src/wm/v1/run_training.py --features 18 --only v1_1 v1_4
python src/run_version_training.py --version 3 --features 13 --only base snapshot
```

---

## Phase 5b: Advanced Analysis

### Per-Asset Strategy Selector

Picks the best WM strategy per asset based on IS performance, then validates on OOS.

```powershell
python src/analysis/strategy_selector.py --wm-ensemble
python src/analysis/strategy_selector.py --wm-ensemble --min-sharpe 0.3
```

### Walk-Forward Validation

Tests strategy robustness across multiple time periods (not just one OOS window).

```powershell
python src/analysis/walk_forward.py --wm-ensemble
python src/analysis/walk_forward.py --wm-ensemble --folds 8
python src/analysis/walk_forward.py --wm-ensemble --assets btcusdt ethusdt solusdt
```

### Position Sizing

Applies Kelly criterion or volatility targeting to improve risk-adjusted returns.

```powershell
python src/analysis/position_sizing.py --wm-ensemble
python src/analysis/position_sizing.py --wm-ensemble --method kelly
python src/analysis/position_sizing.py --wm-ensemble --method voltarget --target-vol 0.15
```

### Paper Trading

Runs best strategies on latest data to generate live signals (no real capital).

```powershell
python src/analysis/paper_trader.py --config logs/analysis/selected_strategies.json
python src/analysis/paper_trader.py --config logs/analysis/selected_strategies.json --dry-run
```

### Ensemble Ablation

Tests which model combinations produce the best trading signals.

```powershell
python src/analysis/ensemble_ablation.py --asset btcusdt    # Quick BTC test
python src/analysis/ensemble_ablation.py --sweep             # All 10 assets
```

### Meta-Strategy Allocator

Combines multiple strategy signals into one position (voting, GBT, or PPO).

```powershell
python src/analysis/meta_strategy.py --mode vote --min-agree 3 --asset btcusdt
python src/analysis/meta_strategy.py --mode compare --sweep
```

---

## Strategy Coverage Map

### Layer 1: Pure Price-Action (no ML needed)
| Strategy | Type | Dollar-Bar Edge | Command |
|----------|------|----------------|---------|
| DonchianBreakout | Trend following | Moderate | `--strategy donchian` |
| BollingerMeanRevert | Mean reversion | **High** (vol-normalized bars) | `--strategy bollinger` |
| SMA_Crossover | Trend following | Low | `--strategy sma_cross` |
| VPIN_Trigger | Microstructure | **Very high** (volume-sync) | `--strategy vpin` |
| FlowMomentum | Microstructure | **High** (equal-$ bars) | `--strategy flow_momentum` |
| VolBreakout | Volatility | High (vol_cluster feature) | `--strategy vol_breakout` |
| HurstAdaptive | Regime-switching | **High** (Hurst is a feature) | `--strategy hurst_adaptive` |

### Layer 2: WM-Filtered (WM as signal overlay on rule-based)
| Strategy | Base | WM Gate | Command |
|----------|------|---------|---------|
| WM_DonchFilter | Donchian | h=64 agreement | `--strategy wm_donchian_filter` |
| WM_BollingerFilter | Bollinger | h=64 agreement | `--strategy wm_bollinger_filter` |
| WM_VPIN_Filter | VPIN | h=64 agreement | `--strategy wm_vpin_filter` |
| WM_RegimeSwitch | Donchian | Regime label | `--strategy wm_regime_switch` |
| WM_Threshold | WM signal | Confidence gate | `--strategy wm_threshold` |
| WM_Momentum | WM signal | Direction | `--strategy wm_momentum` |

### Layer 3: WM Direct (WM as standalone signal)
| Strategy | Horizon | Notes | Command |
|----------|---------|-------|---------|
| WM_Mom_h1 | h=1 | Only generalizing horizon | `--strategy wm_momentum_h1` |
| WM_Thr_h1 | h=1 | With confidence threshold | `--strategy wm_threshold_h1` |
| WM_DonchF_h1 | h=1 | Donchian gated by h=1 | `--strategy wm_donch_h1` |
| WM_PseudoMA | h=1 | Rolling prediction accumulation | `--strategy wm_pseudo_ma` |
| WM_DreamTrader | Dream | GPU-intensive imagination | `--strategy wm_dream` |
| WM_TopK | h=64 | Cross-asset rotation (multi-asset mode only) | N/A (sweep only) |

### Layer 4: Agent (RL, backburner)
| Strategy | Status | Notes |
|----------|--------|-------|
| PPO Agent | Not trained | End-to-end RL on frozen WM predictions |

---

## Next Steps (Priority Order)

### Step 1: Validate the edge (before any real capital)
```powershell
# Walk-forward: does +0.836 Sharpe survive across multiple time windows?
python src/analysis/walk_forward.py --wm-ensemble

# Position sizing: can Kelly/vol-targeting improve risk-adjusted returns?
python src/analysis/position_sizing.py --wm-ensemble
```

### Step 2: Test new dollar-bar strategies
```powershell
# Full sweep with new microstructure strategies (no WM needed)
python src/analysis/strategy_lab.py --sweep

# WM-filtered sweep (includes new WM_Bollinger, WM_VPIN)
python src/analysis/strategy_lab.py --sweep --wm-ensemble

# Re-select best strategies per asset (new candidates available)
python src/analysis/strategy_selector.py --wm-ensemble
```

### Step 3: Expand model diversity
```powershell
# Train V1 with more features
python src/wm/v1/run_training.py --features 18

# Train V4 Mamba (different architecture = ensemble diversity)
python src/run_version_training.py --version 4 --features 13 --only base
```

### Step 4: Refresh data and paper trade
```powershell
# Refresh data via the DAG runner (fetches incrementally, rebuilds chimera if needed)
python src/pipeline/run_pipeline.py --universe u10

# Generate live signals
python src/analysis/paper_trader.py --latest
```

---

## Current State (as of 2026-04-28)

| Phase | Status | Next Action |
|-------|--------|-------------|
| Data fetch (T0) | **Complete** (115K+ files, all u50 + 55/69 u100) | — |
| Pipeline T1-T6 | **Ready** — 77 v50 chimera + 53 v51 chimera | `run_pipeline.py --status` |
| Pre-train gate | Ready | `pre_train_gate.py --asset BTCUSDT` |
| V0 baselines | **Pending** (post-feature_sets centralization) | Run f13/25/41/60/121 sweep |
| V1 training | Pre-2026-04-27 checkpoints stale | Retrain V1.0 f13 first (reference) |
| Model validation harness | **18/18 active models PASS** (V13/V14 frozen) | `scripts/validate_all_models.py` |
| V2-V9 | Validated; settings invariants pinned | Train after V1 baselines pass gates |
| V11/V12 | Validated forward+backward | Train after V1 baselines |
| V15-V18 (frontier) | Validated; SOTA architectures | Defer until V1.x exhausted |
| Strategy layer (xsec ranker u50) | **Deployed champion** (Sharpe 3.36 WF) | Live execution focus |
| Walk-forward / position sizing | Built | Re-run after V1 retraining |

## Data Splits — calendar-aligned, no cross-asset leakage

Frozen split dates in `config/data_config.yaml` (loaded via `src/pipeline/purge_split.get_split_dates()`). Every asset's train/val/oos/unseen boundary lands on the **same calendar date**, eliminating cross-asset overlap leakage via xd_* features.

| Segment | Calendar window | Fraction | Used for |
|---|---|---:|---|
| train | start → 2023-07-01 | ~50% | model fitting |
| val   | 2023-07-01 → 2024-05-15 | ~20% | early stopping, ShIC, gate checks |
| oos   | 2024-05-15 → 2025-03-15 | ~20% | model selection, walk-forward eval |
| unseen | 2025-03-15 → present | ~10% | backtesting, never touched in dev |

Purge gap: 400 bars between segments (covers `WINDOW_ADAPTIVE=200` z-score window + Hurst R/S `window=200` cascading dependency).

**Implementation** (post-2026-04-28 unification):
- V0 baselines + V1.x training: `WalkForwardSplitter.split_four_way_dated(segments)` — uses timestamps from chimera + frozen YAML boundaries.
- V11/V12/V13/V14: still on legacy `split_four_way()` (per-asset fractional). V13/V14 frozen; V11/V12 will switch when retrained.
- Strategy layer / xsec ranker: uses `pipeline.purge_split.split_chimera()` directly via `TrainingLoader`.

All three paths now read the SAME `config/data_config.yaml` boundaries — single source of truth.

## Asset Universes

Declarative specs at `config/universes/{u10,u50,u100}.yaml`. Loaded by `src/pipeline/universe_loader.py`. Inline `is_u10/is_u50/is_u100` columns in v51 chimera (no code lookups needed in strategies).

| Universe | n | Used by | Status |
|---|---|---|---|
| **u10** | 10 | V0 + V1.x WM training (hardcoded `ASSET_LIST`), litmus gate | 10/10 chimera ready |
| **u50** | 50 | xsec ranker (deployed Sharpe 3.36), production strategies | 50/50 chimera ready |
| **u100** | 69 effective | Research universe, breadth scans, top-mover capture | 55/69 chimera ready |

**WM at u50/u100**: Currently gated, not blocked. Asset embedding hardcoded at `NUM_ASSETS=10`. To train V1.x at u50: replace `ASSET_LIST` literal in settings.py with `Universe.load('u50').list_symbols()` + bump `NUM_ASSETS=50` (~1280 extra params). Strategy layer + xsec ranker already consume u50 directly via `universe_loader`.

## Quick Reference: Feature Counts

Centralized in [src/feature_sets.py](src/feature_sets.py) — single source of truth across V0-V19.

### v50 (chimera_legacy/dollar/) — 77 assets ready

| Count | Contents | Source |
|-------|----------|--------|
| 13 | Legacy base (norm_deviation ... norm_spread_bps) | base |
| 18 | +5 extended (ma_distance, whale, efficiency, return_4, return_16) | extended |
| 25 | +4 Hawkes (intensity, buy/sell, imbalance) +3 Tier 1 (kurtosis, bar_duration, funding_momentum) | Hawkes+ |
| 29 | f34 minus 5 dead features (Pattern P, +5-10% ShIC) | post-autopsy |
| 30 | +5 IC-boost (momentum_accel, vol_price_corr, vol_ratio, flow_persistence, oi_price_divergence) | IC-boost |
| 34 | f30 + SOTA (yz_volatility, cs_spread, perm_entropy, kyle_lambda) | SOTA |
| 41 | f34 + 7 XD cross-asset (btc_return, btc_vol, funding_spread, cross_return/vol, ma_distance, momentum_rank) | **v50 ceiling** |

### v51 (chimera/dollar/) — 53 assets ready (frontier features)

| Count | Adds (vs f41) | Family |
|-------|--------------|--------|
| 46 | +5 Hawkes branching ratios | HBR |
| 60 | +14 Stable-3 cross-asset signals | S3 |
| 73 | +9 Basis/spread features | BS |
| 78 | +5 Liquidity-13 + Whale-5 (top picks) | LIQ + WH |
| 81 | +3 social signals | SOC |
| 84 | +3 cross-exchange | XEX |
| 97 | +13 stablecoin metrics | STBL |
| 110 | +13 ETF flow features | ETF |
| 121 | +1 funding panel | FP, prev v51 ceiling |
| **127** | +6 BPV/JV/jump features (BNS 2004 / Lee-Mykland 2008) | **RV_JUMPS, post-2026-04-28 SOTA addition** |

**No f31 exists** — the v50 base counts go 13/18/25/29/30/34/37/41. f29 = Pattern P (drops 5 dead features, expected +5-10% IC vs f34).

**Recommended V0 baseline sweep (~2h):**
```powershell
foreach ($f in 13, 25, 29, 41, 60, 121, 127) {
  python src/wm/v0/v0_baseline/linear_baseline.py    --features $f --workers 8
  python src/wm/v0/v0_baseline/nonlinear_baselines.py --features $f --workers 4
}
```
This 7-point sweep covers: V1.0 reference (f13), V1.x record (f25), Pattern P (f29), v50 ceiling (f41), mid-frontier (f60), v51 ceiling pre-RV (f121), v51 ceiling with RV/jumps (f127).

## Per-Directory Usage Docs

| Directory | Doc | Contents |
|-----------|-----|----------|
| `src/pipeline/` | [USAGE.md](src/pipeline/USAGE.md), [README.md](src/pipeline/README.md) | DAG runner, fetch, chimera, validators |
| `src/wm/v0/` | [USAGE.md](src/wm/v0/USAGE.md) | Linear + non-linear baselines (f13-f121) |
| `src/wm/v1/` | [USAGE.md](src/wm/v1/USAGE.md) | V1 world models (V1.0/V1.1/V1.4/V1.6 active) |
| `backups/BKP_20260429_MODEL_HARMONIZATION/v2/` | [USAGE.md](backups/BKP_20260429_MODEL_HARMONIZATION/v2/USAGE.md) | V2 JEPA + VICReg |
| `src/wm/v3/` | [USAGE.md](src/wm/v3/USAGE.md) | V3 WaveNet-GRU |
| `src/wm/v4/` | [USAGE.md](src/wm/v4/USAGE.md) | V4 Mamba-3 SSM (priority architecture) |
| `backups/BKP_20260429_MODEL_HARMONIZATION/v5/` | [USAGE.md](backups/BKP_20260429_MODEL_HARMONIZATION/v5/USAGE.md) | V5 Hybrid Mamba-Attention |
| `src/wm/v6/` | [USAGE.md](src/wm/v6/USAGE.md) | V6 JEPA + Adversarial |
| `backups/BKP_20260429_MODEL_HARMONIZATION/v7/` | [USAGE.md](backups/BKP_20260429_MODEL_HARMONIZATION/v7/USAGE.md) | V7 Vision Transformer |
| `src/wm/v8/` | [USAGE.md](src/wm/v8/USAGE.md) | V8 Neural ODE (RK2/RK4 selectable) |
| `src/wm/v9/` | [USAGE.md](src/wm/v9/USAGE.md) | V9 Mixture-of-Experts |
| `src/wm/v15/` - `src/wm/v18/` | [README.md](src/wm/v15/README.md) | Frontier (PatchTST, DreamerV3, TD-MPC2, Chronos) |
| `src/analysis/` | [USAGE.md](src/analysis/USAGE.md) | Strategy backtesting |
| `src/agent/` | [USAGE.md](src/agent/USAGE.md) | PPO trading agent |
| `src/frontier/` | [README.md](src/frontier/README.md) | Frontier alpha subtree (DIB, ETF, stable mints) |
| `src/strategy/` | [README.md](src/strategy/README.md) | 17-engine trend-follow stack |
| `scripts/` | `validate_all_models.py` | End-to-end model validation harness (18/18 PASS) |

## Hardware

- **GPU**: RTX 4060 (8GB VRAM) -- mixed precision, one model at a time
- **Training time**: ~2-6h per world model, ~90 min for agent
- **OS**: Windows 11 -- `NUM_WORKERS=0`, no emoji in print()
