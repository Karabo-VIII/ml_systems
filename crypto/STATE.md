# V4 Crypto System — Current State (load on demand)

> Sister doc to [CLAUDE.md](CLAUDE.md). CLAUDE.md holds INVARIANTS (rules
> that don't change). STATE.md holds CURRENT STATE (versions, paths, tables,
> roadmap) that drifts as work progresses.
>
> Update STATE.md when: a new version ships, a path changes, a table
> entry shifts, a roadmap step completes. Do NOT auto-load this on every
> turn — read when you need the specific section.

## Working model — the market-decomposition harness (CURRENT, 2026-06-09)

> This is the live operating model and research layer. It SUPERSEDES the archived wealth-bot v8.5 framework block
> below (kept only as historical provenance). Canonical binding pointer is in [CLAUDE.md](CLAUDE.md) §"MARKET-DECOMPOSITION HARNESS".

| Field | Value |
|---|---|
| Working model | 7 gated stages **research/decomposition → mining → engine → strat → bot → execution → deployment** — [`docs/SOLUTIONING_PIPELINE.md`](docs/SOLUTIONING_PIPELINE.md) |
| Tool | `python -m framework.pipeline {init\|record\|gate\|advance\|status\|registry\|doctor} <market> <instrument>` ([`src/framework/pipeline.py`](src/framework/pipeline.py)) — machine-checked gates + lineage + run-registry + crash-safe store; selftest 14/14 (`python -m framework.selftest`); CDAP-self-policed |
| Market-agnostic via | `MarketAdapter` contract ([`src/framework/adapter.py`](src/framework/adapter.py)); crypto realized in [`src/framework/crypto_adapter.py`](src/framework/crypto_adapter.py) (`isinstance==True`, RWYB) — stocks = a data-adapter swap |
| Research layer (STAGE 00) | [`docs/MARKET_FRAMEWORK/`](docs/MARKET_FRAMEWORK/) — 01_DEAD_LIST (63 refuted) · 02_APPROACH_LEDGER · 03_METHODOLOGY (35 gates) · 04_MARKET_MODEL (38 facts) · 05_OPEN_THREADS (19). Companion theory: CRYPTO_MARKET_UNDERSTANDING / CHIMERA_FEATURE_DICTIONARY / STRATEGY_PLAYBOOK |
| Decomposer/viewer | `python src/mining/decompose.py --asset <SYM> --cadence <TF>` (descriptive feature-behaviour viewer) |
| Store | [`workspaces/`](workspaces/) + [`workspaces/REGISTRY.md`](workspaces/REGISTRY.md). `crypto/_market` = 3/7 (00-02 done+gated, 26 artifacts); `crypto/BTCUSDT` at stage 03 (strat NOT started); `stocks/_market` fresh at 00 (proves agnosticism) |
| Current stage-blocker | The **A/B/C fork** (accept beta+yield ceiling vs chase sub-bar/info-bar frontier) — [`docs/MARKET_FRAMEWORK/05_OPEN_THREADS.md`](docs/MARKET_FRAMEWORK/05_OPEN_THREADS.md) #1; unblocks stage-03 strat work. Pending USER. |
| Audit | [`docs/PIPELINE_SOTA_AUDIT_2026_06_09.md`](docs/PIPELINE_SOTA_AUDIT_2026_06_09.md) — SOTA checklist + honest deferred gaps (stage-06 live monitoring; difficulty-adaptive evaluator depth) |

## Wealth-bot strat layer (2026-05-26)

> **🪦 ARCHIVED BY THE 2026-06-04 RESET — DO NOT TREAT THIS BLOCK AS CURRENT STATE (tombstone 2026-06-05).**
> The 2026-06-04 restart archived ALL prior experiment work (dossiers, leaderboard, failure catalog, the v8.5
> framework + §SM rules, and the "in-flight MAXX session" below) to `archive/restart_2026_06_04/`. Everything in
> this block is **HISTORICAL reference, not the current world** — there is **no verified active alpha post-reset**,
> and the prior "REFUTED/survivor" verdicts were produced by an apparatus now known to be broken (see the
> apparatus-lockdown, 2026-06-05). Current source of truth: the ground-zero framing in
> [`docs/FOUNDATION_2026_06_04.md`](docs/FOUNDATION_2026_06_04.md) +
> [`docs/APPARATUS_LOCKDOWN_SPEC_2026_06_04.md`](docs/APPARATUS_LOCKDOWN_SPEC_2026_06_04.md) (and the session
> auto-memory's founding-framing note, external to the repo). The Fork A (accept
> beta+yield ceiling) vs Fork B (chase the unproven active frontier) decision is **pending the USER**. The
> pipeline / model-layer / WM-cohort sections further below are unaffected by the reset and remain current.
> (Body retained verbatim, not deleted, for provenance — neutralized, not rewritten.)

> Added by INST-MAXX-2026-05-26-NIGHT. Carries the post-v8.5-reset state.

| Field | Value |
|---|---|
| Framework version | **v8.5** (binding 2026-05-27 00:00 SAST) per `docs/WEALTH_BOT_FRAMEWORK_v8_5_AMENDMENTS_2026_05_26.md` |
| Active dossiers | 2: `docs/dossiers/PEPE_SMA__inst_MAXX_2026_05_26.md` + `docs/dossiers/PEPE_EMA__inst_MAXX_2026_05_26.md` (both Phase 0 STARTING, 0% exhaustion) |
| Recent refutations (2026-05-26) | 5 candidates REFUTED-UNDER-STRESS: R12 perp Strat B, R23a, R23c, R23h (AB_AND / ABC_AND), P4_route_basis_pos_only — all fail v8.5 §SM12 (n_eff < 15) AND §SM13 (DSR p > 0.91 @ N=1000). 33-33-33 blend = REFERENCE, re-mine under v8.5. |
| Old combined dossier | `docs/dossiers/PEPE_MA_EMA_*` archived to `docs/dossiers/archive_pre_v8_5/` — historical reference only |
| Key artifacts | `runs/oracle/WEALTH_BOT_LEADERBOARD.md` (post-RED-team verdict table) · `runs/coordination/GAP_CLOSURE_STATUS_2026_05_26.md` · `docs/WEALTH_BOT_FRAMEWORK_v8_5_AMENDMENTS_2026_05_26.md` · `runs/oracle/WEALTH_BOT_FAILURE_CATALOG.md` |
| Active scripts policy | Only POST-v8.3 harness (`framework.data_loader` / `wealth_bot.harness.CanonicalHarness`) is Phase 1-eligible. **12 pre-v8.3 Pattern T scripts FROZEN** (advisory-only, not deploy-eligible) per `src/wealth_bot/harness.py::MIGRATION_BACKLOG`. |
| Audit JSON contract | `claim_contract.py` MUST populate: `per_trade_returns_sorted_desc`, `top_3_pct_of_compound`, `jackknife: {K=0..K=5}`, `combined_K2_plus_S9_pct`, `mechanism_falsifier_check.verified_by`, `sample_size_discipline.passes_stressed_gate`, `phase1_n_eff_gate`, `dsr_at_family_N_p_value`, `canonical_seeds: {bag_seed, feat_seed, rng_seed}`, `quality_signals_gates: {QS1-QS6}`. CDAP `check_wealth_bot_claims.py` exit 2 on violation. |
| Multi-timeframe mandate | §SM15 — Phase 1 mining MUST cover all five canonical cadences {15m, 30m, 1h, 4h, 1d}. NULL cells per cadence are valid (coverage required, not coverage-and-pass). DSR family-N includes ALL cadences. |
| Ranking metric | §SM19 — `compound_discounted = compound_raw × min(1, n_eff/12)`. Sharpe = tiebreak only. Robustness (10/10 seeds positive, p05 > 0, max DD < 30%) = CONSTRAINT not ranking. |
| In-flight | MAXX-INST-2026-05-26-NIGHT autonomous 8h session — Phase 0 diagnostics dispatched, Phase 1 v8.5 mining queued. See `CURRENT_PLAN.md §4`. |

The pipeline / model-layer / WM cohort sections below remain current (no v8.5 invalidation — wealth-bot reset affects strategy/dossier layer only).

## Architecture pipeline

```
Raw Data (Binance) -> Dollar Bars -> 18 Base Features -> Cross-Asset Enrichment (+6) -> 24 Features -> Targets (10) + regime_label
                                                                                           |
                                                                V1.0: 13 features (reference baseline)
                                                                V1.1: 13/18/21/25/30/37 features (flexible, XD anti-memorization)
                                                                V1.4: 13/18/21/25/30/37 features (FeatureAttentionBlock, cross-feature attention)
                                                                V1.6: 13/18/21/25/30/37 features (all techniques: KL anneal, Gumbel, ATME, dream)
                                                                V1.2, V1.3, V1.5, V1.7: ARCHIVED in src/wm/v1/archive/
                                                                V2-V9: 13/18/30/37 features (flexible, select by name from FEATURE_LIST)
                                                                                           v
                                                                      World Models (V1-V9) -> Validation -> Trading
```

## Directory structure (V51 v2 SOTA layout, 2026-04-25)

See `src/pipeline/README.md`, `docs/PIPELINE_HARDENING_2026_04_25.md`,
`docs/PIPELINE_FRAMEWORK_2026_05_01.md` for full reference.

```
config/data_config.yaml               # Assets, bar sizes, date range
config/feature_registry.yaml          # 11 sources × 80 frontier features (declarative, file-extensible)
config/universes_index.yaml           # u10/u50/u100 references
data/manifests/v51_<SYM>.json         # per-build lineage + checksums (was `_manifests/`; doc corrected 2026-05-03)
config/universes/{u10,u50,u100}.yaml  # canonical universe specs
data/raw/<SYMBOL>/                    # bronze: Binance native (aggTrades/funding/metrics/bookTicker)
data/raw_external/                    # bronze: non-Binance (farside/defillama/deribit/coinbase_okx_bybit/wikipedia/binance_futures_panels)
data/processed/panels/daily/          # silver: multi-asset panels (s3, basis, liq, etf, hawkes_branching, te, lob_proxy, ...)
data/features/<SYMBOL>/               # silver: per-asset frontier_daily.parquet (80 features)
data/bars/<SYMBOL>/                   # silver bar fabric (dib, runs_tick, runs_volume, range, adaptive_vol)
data/processed/chimera_legacy/dollar/<sym>usdt_v50_chimera_<DATE>.parquet  # GOLD legacy (V1-V14 inference)
data/processed/chimera/dollar/<sym>usdt_v51_chimera_<DATE>.parquet         # GOLD v51 (154 cols)
data/processed/chimera/{1d,4h,1h,15m}/                                     # GOLD v51 cadence views
data/lob/<SYMBOL>/<DATE>/<HH>.parquet # streaming LOB collector output
src/pipeline/                         # all pipeline tools (see src/pipeline/README.md)
  parquet_io.py                       # FRAMEWORK: atomic_write_parquet + is_fresh + safe_unlink
  dispatch.py                         # FRAMEWORK: run_per_task (serial/thread/process)
  cli.py                              # FRAMEWORK: add_standard_args + resolve_assets
  feature_registry.py                 # registry loader + validator
  universe_loader.py                  # u10/u50/u100 loader
  bar_fabric.py                       # unified API for 11 bar types
  chimera_loader.py                   # strategy-facing single API (USE THIS, not pl.read_parquet)
  frontier_consolidator.py            # silver layer builder
  make_dataset_legacy.py              # legacy v50 chimera builder
  make_dataset.py                     # SOTA v51 chimera builder
  validate_chimera.py                 # 14-check v51 validator
  cross_asset_consistency.py          # xd_* sanity audit
  pipeline_e2e_test.py                # 12-stage end-to-end smoke
  cadence_correctness.py              # cadence views match dollar resampling
  v50_backward_compat.py              # V1-V14 inference path verifier
  data_health_check.py                # bronze + silver drift / freshness / coverage
  pre_train_gate.py                   # composite CI hook (5 validators)
  purge_split.py                      # leak-proof train/val/oos/unseen w/ cadence-aware purge
  benchmark_loads.py                  # cold + warm load times across cadences
  strategy_parity_test.py             # v51 features identical to raw frontier (max diff < 1e-6)
src/wm/v1/v1_0_training/                 # V1.0: 13-feature reference baseline
src/wm/v1/v1_1_training/                 # V1.1: 13/18/21/25/30/37 features, XD anti-memorization
src/wm/v1/v1_4_training/                 # V1.4: FeatureAttentionBlock (iTransformer cross-feature attention)
src/wm/v1/v1_6_training/                 # V1.6: all techniques (KL anneal, Gumbel, ATME, dream)
src/wm/v1/archive/                       # V1.2, V1.3, V1.5, V1.7
src/wm/v{3,4,6,8,9}/v{N}_training/    # V3-V9 active: base + FiLM(.X) + ensemble(.E) + NCL(.D) variants
src/wm/v10/                              # V10: Meta-ensemble aggregation
src/wm/v11-v14/                       # FROZEN, deprecated
backups/BKP_20260429_MODEL_HARMONIZATION/v{2,5,7}/  # ARCHIVED versions
src/wm/v15/patchtst_encoder.py           # G10 PatchTST drop-in encoder (stub)
src/agent/                            # PPO trading agent (environment, policy, rewards)
src/analysis/                         # Strategy backtesting + validation suite
src/frontier/                         # Frontier features + strategies + LOB collector
src/anti_fragile.py                   # Walk-forward CV, augmentation, shuffled IC, overfitting detection
src/validation_utils.py               # RobustValidator, hallucination detection
docs/                                 # PIPELINE_HARDENING_2026_04_25.md, V50_TO_V51_FIXES.md, ...
backups/                              # Chronological snapshots (see backups/LINEAGE.md)
models/v{0-15}/v{N}/                  # Saved model checkpoints (grouped by major version)
logs/                                 # Training logs, analysis results
```

## 10 Assets

BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC (all USDT pairs on Binance Futures)

## Data Schema — V51 v2 Chimera (154 cols, 2026-04-25)

V51 v2 = V50's 63 cols + 80 frontier features + 11 helpers/metadata.

**Strategy code rule**: read via `from pipeline.chimera_loader import ChimeraLoader; ChimeraLoader().load(sym, cadence='1d')`. Do NOT call `pl.read_parquet` directly.

**41 Pipeline Features (34 Base + 7 Cross-Asset)** — present in both v50 and v51 chimera.

### Base Features (0-33) — per-asset

| # | Column | Description | Used by |
|---|--------|-------------|---------|
| 0 | norm_deviation | Volatility regime (EMA spread) | All |
| 1 | norm_fd_close | Fractional diff (stationary trend memory) | All |
| 2 | norm_vpin | Volume-sync probability of informed trading | All |
| 3 | norm_flow_imbalance | Buy/sell volume delta | All |
| 4 | norm_vol_cluster | Volatility of volatility | All |
| 5 | norm_funding | Funding rate (positioning sentiment) | All |
| 6 | norm_tick_count | Liquidity activity proxy | All |
| 7 | norm_log_volume | Absolute volume (log-scaled) | All |
| 8 | norm_hl_spread | Rogers-Satchell realized volatility | All |
| 9 | hurst_regime | Mean-reversion vs trending (R/S statistic) | All |
| 10 | norm_oi_change | Open interest rate of change | All |
| 11 | norm_return_1 | Lagged 1-bar return | All |
| 12 | norm_spread_bps | Effective bid-ask spread proxy | All |
| 13 | norm_ma_distance | SMA-200 distance | Extended+ |
| 14 | norm_whale | Avg trade size (institutional flow) | Extended+ |
| 15 | norm_efficiency | Price efficiency ratio | Extended+ |
| 16 | norm_return_4 | Lagged 4-bar cumulative return | Extended+ |
| 17 | norm_return_16 | Lagged 16-bar cumulative return | Extended+ |
| 18 | norm_return_kurtosis | Rolling excess kurtosis | Tier 1+ |
| 19 | norm_bar_duration | Bar duration — volume clock speed | Tier 1+ |
| 20 | norm_funding_momentum | Funding rate of change | Tier 1+ |
| 21 | norm_hawkes_intensity | Tick rate vs EMA | Hawkes+ |
| 22 | norm_hawkes_buy_intensity | Buy-side clustering | Hawkes+ |
| 23 | norm_hawkes_sell_intensity | Sell-side clustering | Hawkes+ |
| 24 | norm_hawkes_imbalance | Buy - sell clustering | Hawkes+ |
| 25 | norm_momentum_accel | Second derivative of price | IC-boost |
| 26 | norm_vol_price_corr | Volume-price correlation | IC-boost |
| 27 | norm_vol_ratio | Volatility term structure | IC-boost |
| 28 | norm_flow_persistence | Flow autocorrelation | IC-boost |
| 29 | norm_oi_price_divergence | OI building while price flat | IC-boost |
| 30 | norm_yz_volatility | Yang-Zhang volatility (MVUE) | SOTA |
| 31 | norm_cs_spread | Corwin-Schultz alpha | SOTA |
| 32 | norm_perm_entropy | Permutation entropy | SOTA |
| 33 | norm_kyle_lambda | Kyle's lambda | SOTA |

**SOTA equivalences** (old features KEPT, new ADDED for backward compat):
- `norm_yz_volatility` (#30) upgrades `norm_hl_spread` (#8) — adds overnight correction
- `norm_cs_spread` (#31) complements `norm_spread_bps` (#12)

### Cross-Asset Features

| # | Column | Description |
|---|--------|-------------|
| 34 | xd_btc_return | BTC leader signal (pass-through) |
| 35 | xd_btc_volatility | BTC risk regime (pass-through) |
| 36 | xd_funding_spread | Asset funding vs BTC (z-scored); BTC=0 |
| 37 | xd_cross_return_mean | Market breadth, excl. BTC (z-scored) |
| 38 | xd_cross_vol_mean | Systemic risk, excl. BTC (z-scored) |
| 39 | xd_ma_distance | Cross-sectional SMA-200 trend vs market avg |
| 40 | xd_momentum_rank | Cross-sectional return rank vs all peers |

### Auxiliary Labels (not features)

| Column | Description |
|--------|-------------|
| regime_label | SMA-200 regime classification (0=bear, 1=neutral, 2=bull) |

### 10 Multi-Horizon Targets

| Column | Description | Used for |
|--------|-------------|----------|
| target_return_1 | Raw next-bar return | Model training (default) |
| target_return_4 | Raw 4-bar cumulative return | Model training (default) |
| target_return_16 | Raw 16-bar cumulative return | Model training (default) |
| target_return_64 | Raw 64-bar cumulative return | Model training (default) |
| target_voladj_1/4/16/64 | Vol-adjusted return (symlog) | Legacy (deprecated for training — vol shortcut) |
| target_return_50 | 50-bar risk-adjusted return | Auxiliary |
| target_vol_20 | 20-bar forward volatility | Auxiliary |

### Dollar Bar Columns

`timestamp, bar_id, open, high, low, close, volume, volume_usd, buy_vol, sell_vol, tick_count` + 41 features + 10 targets + regime_label

## Model Variants + Architecture Summary

- **V.0 (base)**: Base WM — train first, gate check, then variants
- **V.X**: FiLM adapter on frozen base (~5-25K params) — only if base passes gates
- **V.E**: Snapshot Ensemble (cyclical cosine LR) — priority variant after base
- **V.D**: Multi-Head NCL diversity (K=5 parallel return prediction paths) — lowest priority

| Ver | Core | Params | Innovation | Status |
|-----|------|--------|------------|--------|
| V1.0 | Transformer + RSSM | 2.0M | Reference baseline | f34 COMPLETE (IC=0.0660 ShIC=0.0320) |
| V1.1 | V1.0 + XD anti-memorization | 2.0M | base_dim/input_dim split | f34 COMPLETE (IC=0.0674 ShIC=0.0330 — record) |
| V1.4 | V1.1 + FeatureAttention | 2.0M | iTransformer cross-feature | f34 COMPLETE (IC=0.0679 ShIC=0.0314) |
| V1.6 | V1.1 + KL/Gumbel/ATME/dream | 2.0M | All anti-memorization techniques | f34 COMPLETE (IC=0.0619 ShIC=0.0329) |
| V3-clean | WaveNet-Direct | 1.9M | Multi-scale dilated causal conv | Ready (`--clean`) |
| V6-clean | Transformer + Discriminator | 3.1M | Time-shuffle adversarial | Ready (`--clean`) |
| V8 | Neural ODE (RK4) | 2.5M | Continuous-time dynamics | Ready (low priority) |
| V9-clean | GRU + 3 MoE Experts | 7.6M | Regime-gated bear/neutral/bull | Ready (`--clean`) |
| V11 | WaveNet + MoE + Discriminator | 2.9M | Combined V3/V6/V9 | Ready |
| V12 | Cross-Asset Attention | 841K | 10 assets jointly, per-timestep | Ready |
| V13 | Temporal Fusion Transformer | 2.2M | Per-timestep feature selection (VSN) | Ready |
| V14 | Diffusion Return Distribution | 2.4M | Full return distribution | Ready |
| V4 | Mamba-3 SSM + RSSM | 3.5M | Complex SSD, trapezoidal, QK-Norm+RoPE | Ready |
| V2,V5,V7 | -- | -- | -- | ARCHIVED in `backups/BKP_20260429_MODEL_HARMONIZATION/` |

V4: Mamba-3 (ICLR 2026) with RSSM categorical bottleneck. Complex-valued state via RoPE, SSD chunk-based scan.
V11-V14: no RSSM, no reconstruction, no dream. Clean architectures.
V3/V6/V9-clean: stripped versions, use `--clean` flag with train scripts.

## 2026-05-21 WM-layer SOTA sweep — retrain status

43 commits closed all documented issues from the pre-retrain RED-team audit.
The WM cohort is ready for post-rebuild retraining. Key items per version:

### Cross-cutting fixes (apply to all RSSM family + downstream importers)
- Canonical Jensen-correct `TwoHotSymlog` at [src/wm/_shared/twohot.py](src/wm/_shared/twohot.py). Every version's `components.py` imports from there. Empirically validated: peaked@max → 1.7183 = e¹-1 (Jensen-correct), not 1.0 (Jensen-wrong).
- AMP fp32 wraps around Gumbel-Softmax + Categorical KL on V1.x / V3 / V4 family.
- `load_latest_collision` guard label on every trainer.

### Per-version SOTA designs
- **V1.6**: A3 (decoder always sees full feat; ATME only on heads — HRSSM pattern). B2 FULL (separate `dream_return_trunk` + `dream_return_heads` — TD-MPC2 full isolation).
- **V3/V4/V8** + ablations: posterior reads MASKED `input_obs`, not raw `obs_seq`. V8 ODE also reads masked input.
- **V11**: Hurst gate uses `shifted` not `obs_seq` (1-bar look-ahead fix). `HEADLINE_DISC_SPECTRAL_NORM` wired. `HEADLINE_MOE_EXPERTS=1` wired (skips reverting expert entirely — saves ~74K params + drops V9 leak).
- **V13**: `InterpretableAttention` LayerNorm → RMSNorm.
- **V14**: DPM-Solver++ 2M sampler (Lu et al NeurIPS 2022) with final-step `x0_hat` shortcut. `HEADLINE_MODE` re-enabled (was gated OFF when DDIM was broken). K=15 ≈ DDPM K=50 quality on simple manifolds (untrained probe confirms determinism + convergence by K=5; quality validation pending trained model).
- **V22 / V25**: Last-bar supervision (Timer-XL ICLR 2025 / TimesFM pattern). Encoder still processes 96-bar context; only the LAST bar's prediction is supervised — by construction no future-bar leak. `AntifragileDataset stride=1` for both (consecutive bars each become a "last bar" target — recovers per-bar supervision count). `USE_LAST_BAR_SUPERVISION = True` default.
- **V25**: `regime_ffn` allocation gated on `USE_REGIME_FFN` — saves ~9.85M dead params (52% reduction: 19.17M → 9.32M).

### Checkpoint compatibility on retrain

Most fixes preserve internal module shapes — existing ckpts will load with `strict=False` warnings on benign mismatches:

| Version | Ckpt-compat | Notes |
|---|---|---|
| V1.0 | ✅ Clean | No shape changes |
| V1.1 / V1.4 | ✅ Clean | TwoHotSymlog has no state to load |
| V1.6 | ⚠ New keys | `dream_return_trunk`, `dream_return_heads.*` are new; will randomly init on resume |
| V3 + V3_1/2/3 | ✅ Clean | Loss-only changes |
| V4 + V4_1/2/3 | ✅ Clean | Loss-only changes |
| V6 + V6_1/2/3 | ✅ Clean | TwoHotSymlog import only |
| V8 + V8_1/2/3 | ✅ Clean | Loss-only changes |
| V11 | ⚠ Multiple breaks | (a) `discriminator.net.*.weight` → `discriminator.net.*.parametrizations.weight.original` (spectral_norm rename); (b) `expert_reverting` MISSING when `HEADLINE_MOE_EXPERTS=1`. Trainer `strict=False` warns; fresh-init both. |
| V13 | ⚠ Norm shape | `InterpretableAttention.norm` was `nn.LayerNorm` (weight + bias), now `RMSNorm` (weight only). Bias param will be ignored on load. |
| V14 | ✅ Clean | Sampler-only change; denoiser weights unchanged |
| V22 / V25 | ✅ Clean | Loss-only + dataset stride change; model shapes unchanged |
| V25 (regime_ffn) | ⚠ Missing keys | `regime_ffn.1.*` / `regime_ffn.2.*` no longer exist when `USE_REGIME_FFN=False`. `regime_ffn.0.*` loads cleanly. |

### Retrain acceptance gates (per CLAUDE.md Validation Gates + Indisputable Operating Lens)

Before declaring a retrained model "good", it MUST clear:

| Gate | Threshold | Source |
|---|---|---|
| IC (h=1) | ≥ 0.030 (SHIP-tier) for V1.1/V14/V22/V25; ≥ 0.015 (Filter-tier) for all others | CLAUDE.md Validation Gates |
| ShIC (h=1) | ≥ 0.015 (per-model) AND ShIC / IC ≥ 0.30 (anti-fragile invariant) | CLAUDE.md Cross-Version Training Invariants |
| Train / Val loss ratio | < 2.0 | CLAUDE.md GATE_LOSS_RATIO_MAX |
| Reconstruction MSE (RSSM family) | < 0.12 | CLAUDE.md GATE_REC_MSE_MAX |
| KL range (RSSM family) | 0.01 - 15.0 | CLAUDE.md GATE_KL_MIN/MAX |
| **Pre-mortem for V22/V25**: IC ≥ 0.03 AND ShIC ≥ 0.015. If retrain shows IC=0.01 / ShIC=0.005, that is "broken differently" (no signal at all) — NOT a successful fix. | — | Oracle validation 2026-05-21 |
| **V11 disc-stability**: discriminator loss should stay in [0.3, 1.0] after warmup. If collapses to 0 OR grows unbounded, the bumped `DISC_WEIGHT=0.3` × spectral_norm combination needs tuning. | — | Oracle validation 2026-05-21 |

### Known residuals (intentional)

- **V14 `HEADLINE_CFG_SCALE = 1.5`**: dead flag. V14 has no unconditional path; classifier-free guidance not implemented. Either add unconditional training or remove the flag — neither blocks retrain.
- **V25 inline `CryptoPeriodEmbedding` / `RateBudgetVIB`**: ckpt-compat preserved by keeping inline copies. Consolidating to `_shared/frontier_components.py` would change parameter names.

## Key Model Architecture (V1 Reference)

- Transformer encoder (d_model=256, 8 heads, 3 layers) + RSSM latent space (24x24 categorical)
- Asset embeddings (32-dim per asset, 10 assets)
- TwoHot return prediction (255 bins, range [-1, 1] ALL versions, raw return targets, no focal/smoothing)
- Multi-horizon prediction heads (return_1, return_4, return_16, return_64)
- ACTIVE_HORIZONS = [1, 4, 16, 64] — ALL horizons contribute to loss. h16/h64 act as multi-scale regularizers preventing ShIC decline.
- Pairwise ranking auxiliary loss — DISABLED (weight=0.0). V1 only, code retained but inactive.
- Sequence length: 96 bars (~24h of BTC data)
- Training: batch_size=32 (V1), steps_per_epoch=2000.

## Pattern P + Q Bundle (2026-04-14, retraining cycle)

**Pattern P — 5 dead features (14.7% capacity drag)**:
- `norm_funding` (idx 5, raw IC +0.0001)
- `hurst_regime` (idx 9, raw IC +0.0003)
- `norm_funding_momentum` (idx 20, raw IC +0.0003)
- `norm_vol_price_corr` (idx 26, raw IC +0.0004)
- `norm_perm_entropy` (idx 32, raw IC +0.0001)

Resolution: `FEATURE_LIST_29` added to all 12 active versions. f29 = f34 minus the 5 dead. Use `--features 29` in training scripts.

**Pattern Q — Reconstruction loss dominance**: top gradient norms invest in reconstructing log_volume / spread_bps (high info, IC~0.001) at the expense of return heads. Resolution: `REC_LOG_VAR_CLAMP_MIN = 0.5` (was 0.0). Reconstruction max weight reduced from 1.0x to exp(-0.5)=0.61x. Applied to all RSSM-with-recon versions. Not applicable to V6 (JEPA) or V11-V14 (no RSSM).

**Expected impact**: Each pattern projected +5-10% ShIC. Bundle target: +10-20% ShIC vs V1.6=0.0333 baseline → ~0.037-0.040.

**Validation order**: Train V1.0 f29 first as A/B vs current V1.0 f34 (ShIC=0.0319). If new ShIC > 0.035, propagate to V1.1/V1.4/V1.6/V3/V6/V9. If <0.025, bisect P vs Q.

**Existing checkpoints invalidated**: All V1-V9 best_ema checkpoints stale (different feature schema + rec weight). Retraining required across the board.

## Trading Architecture (Reconciled)

```
World Model predictions (h1, h4 return distributions)
         |
    Signal Layer (IC~0.03, used as FILTER not standalone signal)
         |
    Strategy Layer (Donchian / Bollinger / VPIN / VolBreak / HurstAdaptive)
         |
    Risk Layer (position sizing, drawdown circuit breaker)
         |
    Execution (SPOT mode, 0.10%/side, long-only)
```

- WM signal too weak to trade alone (IC~0.03 vs 0.12% per-trade cost). Value is as FILTER on rule-based strategies.
- PPO agent (`src/agent/`) on backburner until signal improves
- Analysis tools: `src/analysis/` (strategy_lab, strategy_selector, walk_forward, position_sizing, paper_trader, donchian, wm_backtest, price_action_mc, live_backtest)
- 20 strategies across 4 layers: 7 price-action, 9 WM-filtered, 4 WM-direct
- VPIN_Trigger best BTC (OOS Sharpe +1.379)
- WM-filtered per-asset best: avg OOS Sharpe +0.836 (9/10 assets, vs Donchian +0.545)

## Training Roadmap (Post-Reconciliation)

1. **V1 f13+f18+f25+f37 COMPLETE**: 11 checkpoints across 4 variants. f25 best (+13% IC).
2. **Ensemble ablation**: solo/leave-one-out/group combinations
3. **Strategy validation**: Walk-forward + position sizing on 20-strategy universe
4. **V4 Mamba f13**: Architecture diversity
5. **V1.E snapshots**: Ensemble expansion after walk-forward validates
6. **V2-V9**: V4 (Mamba) → V3 (WaveNet) → V9 (MoE) → rest (f13 then f25)

## Backup Lineage

See `backups/LINEAGE.md` for full chronological record. Key snapshots:
- `BKP_20260307/` — last good pre-memorization baseline (ShIC=0.030)
- `BKP_V51_ERA/` — memorization reference (voladj + focal + bins[-5,5])
- `BKP_20260313_RECONCILE/` — pre-reconciliation snapshot (all SOTA fixes applied)

## Project-state pointers (latest deployment / strat)

- [docs/TRUE_VALIDATION_VERDICT_2026_05_05.md](docs/TRUE_VALIDATION_VERDICT_2026_05_05.md) — **5-quarter validation verdict; all 3 blends SHIP**
- [docs/PRODUCTION_BACKTEST_2026_Q1_2026_05_05.md](docs/PRODUCTION_BACKTEST_2026_Q1_2026_05_05.md) — Q1 2026 monthly per-seed per-blend results
- [docs/PHASE13_5_CAVEAT_RESOLUTION_AND_DEPLOY_BLENDS_2026_05_05.md](docs/PHASE13_5_CAVEAT_RESOLUTION_AND_DEPLOY_BLENDS_2026_05_05.md) — **canonical deployment blends + 3 caveats resolved**
- [docs/PHASE13_FINAL_TIERED_RESULTS_2026_05_05.md](docs/PHASE13_FINAL_TIERED_RESULTS_2026_05_05.md) — comprehensive %-metrics, 18 measures × 42 strategies
- [docs/PHASE12_CORRECTED_RESULTS_2026_05_05.md](docs/PHASE12_CORRECTED_RESULTS_2026_05_05.md) — corrected WF audit (matched horizons; v51_full SHIPS at Sh 4.72)
- [docs/PHASE12_VALIDATION_AUDIT_2026_05_05.md](docs/PHASE12_VALIDATION_AUDIT_2026_05_05.md) — methodology audit (horizon mismatch, CAGR artifact)
- [docs/DEPLOY_GUIDE_U87_2026_05_03.md](docs/DEPLOY_GUIDE_U87_2026_05_03.md) — original u87 blend
- [docs/STRAT_TUNNEL_VISION_DIAGNOSIS_2026_05_03.md](docs/STRAT_TUNNEL_VISION_DIAGNOSIS_2026_05_03.md) — wealth-max gaps + recommendations
- [docs/STRAT_WEALTH_PATH_TRIAGE_2026_05_03.md](docs/STRAT_WEALTH_PATH_TRIAGE_2026_05_03.md) — pipeline / connectors / patch-risk triage
- [docs/STRAT_WEALTH_MAX_PATHS_2026_05_03.md](docs/STRAT_WEALTH_MAX_PATHS_2026_05_03.md) — paths from u87 to 10X-100X PA

## Canonical deployment blends (Phase 13.5, 2026-05-05)

Three blends defined; full details in [PHASE13_5_CAVEAT_RESOLUTION_AND_DEPLOY_BLENDS](docs/PHASE13_5_CAVEAT_RESOLUTION_AND_DEPLOY_BLENDS_2026_05_05.md).

### BLEND PRIME (recommended primary deploy)

| Sleeve | Weight | Full-window Sharpe | DD% |
|---|---|---|---|
| `4h_K5_h32_sleeve` | 30% | +14.20 | -0.54% |
| `blend_v6_with_4h_h32` | 25% | +7.06 | -3.13% |
| `revised_blend_u87` | 25% | +4.55 | -8.03% |
| `v51_full_T4` (rolling-cutoff + lag-fix) | 20% | +3.92 (Q1 2026) | -0.76% |

Expected: Sharpe 6.0-7.0 / CAGR 70-100%/yr live / Max DD < -5%.

### BLEND CONSERVATIVE (capital-preservation)

| Sleeve | Weight |
|---|---|
| `4h_K5_h32_sleeve` | 40% |
| `v51_full_T4` | 30% |
| `blend_v2_DIB_cppi` | 20% |
| Cash / USDC reserve | 10% |

Expected: Sharpe 5.0-6.0 / CAGR 50-70%/yr live / DD < -3%.

### BLEND AGGRESSIVE (max absolute return)

| Sleeve | Weight |
|---|---|
| `revised_blend_u87` | 35% |
| `V3_E3_enhanced_u87` | 25% |
| `4h_K5_h32_sleeve` | 20% |
| `blend_v6_with_4h_h32` | 20% |

Expected: Sharpe 4.5-5.5 / CAGR 90-130%/yr live / DD -10 to -15%.

### v51_full_T4 retrain protocol

- Cadence: **quarterly rolling retrain** (next due 2026-08-01)
- Train window: rolling 12-15 months
- Purge gap: 3 months before test
- Suspect-col lag: 1-day groupby-shift on `bd_`, `te_`, `hbr_`, `lob_`, `mv_` prefixes
- Target: `target_return_4` with 4-day matched hold
- Validation gate: bagged-CPCV PBO < 0.30 + DSR p < 0.05
