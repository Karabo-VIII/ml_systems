# V0 — Linear / Non-linear Floor Benchmark

> **Role in cohort**: defines the IC floor every neural WM must beat by ≥3×.
> Not a model to ship. The gatekeeper that prevents the cohort from
> celebrating noise.

## Purpose

Test whether the engineered features carry signal that any DL model (V1-V25)
must beat to justify its complexity. If linear IC ≈ DL IC, the architecture
is buying minimal improvement. If DL IC >> linear IC, the temporal + non-linear
modeling is justified.

Per CLAUDE.md "Indisputable Operating Lens": every Headline-tier WM (IC > 0.10)
must clear V0 by 6.6×. Even the current Trader-tier record (V1.1 IC 0.073) is
~4× V0.

## Architecture

Four sub-models, all running on the same chimera features:

| Sub-model | Algorithm | What it tests |
|---|---|---|
| **Ridge linear** | scikit-learn Ridge over α ∈ [0.01, 0.1, 1, 10, 100] | True IC floor — no temporal, no non-linear |
| **Polynomial** | Ridge over degree-2 polynomial features | Quadratic feature interactions |
| **HistGBT** | scikit-learn HistGradientBoostingRegressor | Non-parametric tree ensemble |
| **MLP** | sklearn MLPRegressor, 2-layer | Smooth non-linear |

**None have temporal modeling.** The DL win must come from RSSM / Transformer /
Mamba / WaveNet capturing time-dependent dynamics.

### Anti-memo (yes, even floor models need it)

- **Walk-forward split**: 50/20/20/10 (train/val/oos/unseen) via
  `config/data_config.yaml` frozen dates
- **400-bar purge gap** between segments (prevents rolling z-score leakage)
- **ShIC test**: 5 random shuffle seeds × 4 horizons × 10 assets = 200 replicates
- **Subsample for masking section**: 500K train, 200K val (avoids OOM on pooled
  10-asset data)

### Design rationale

- **Why Ridge over OLS**: feature multi-collinearity (XD features are derived
  from base; cross-asset features inherit BTC). Ridge α-sweep finds the right
  regularization without hand-tuning.
- **Why 5 α's**: log-spaced grid is empirically sufficient to find the optimum
  without overfitting to the validation split.
- **Why these 4 non-linear models**: cheapest representatives of major
  non-linear families. If signal exists, at least one of (poly, GBT, MLP)
  captures it.
- **Why no GPU**: V0 is the audit; running it on CPU enforces "the floor must
  be cheap to compute". Anyone can rerun on a laptop.
- **Why feature_sets.py centralization**: post-2026-04-27 — 18 supported feature
  counts (13-121) from a single source of truth. V0 picks via `--features N`.

## Files

```
src/wm/v0/v0_baseline/
├── settings.py              # config (features, ASSET_LIST, ridge α grid, paths)
├── linear_baseline.py       # Ridge sweep + ablation + ShIC + feature-group masking
├── nonlinear_baselines.py   # Poly / GBT / MLP (ProcessPool parallel)
├── _workers.py              # top-level worker fns for ProcessPoolExecutor
├── save_baseline_preds.py   # persist predictions for downstream meta-ensemble
└── dsr_pbo.py               # NEW 2026-05-16: Bailey DSR + CSCV PBO computation
```

## Usage

### Linear baseline

```bash
# Default: f41 (full v50 schema)
python src/wm/v0/v0_baseline/linear_baseline.py

# Other feature counts (any from feature_sets.SUPPORTED_COUNTS)
python src/wm/v0/v0_baseline/linear_baseline.py --features 13     # legacy V1.0 base
python src/wm/v0/v0_baseline/linear_baseline.py --features 29     # Pattern P (no dead features)
python src/wm/v0/v0_baseline/linear_baseline.py --features 121    # full v51 frontier (needs v51 chimera)

# Parallel
python src/wm/v0/v0_baseline/linear_baseline.py --workers 8

# Simple 90/10 (no purge gap) — for sanity
python src/wm/v0/v0_baseline/linear_baseline.py --full
```

### Non-linear baselines

```bash
python src/wm/v0/v0_baseline/nonlinear_baselines.py
python src/wm/v0/v0_baseline/nonlinear_baselines.py --model poly      # poly only
python src/wm/v0/v0_baseline/nonlinear_baselines.py --workers 4
```

### DSR / PBO (NEW)

```bash
python src/wm/v0/v0_baseline/dsr_pbo.py                                # synthetic smoke
# Use as library:
#   from dsr_pbo import compute_dsr, compute_pbo, dsr_pbo_summary
#   summary = dsr_pbo_summary(returns_per_trial, n_splits_pbo=16)
```

## Key settings (settings.py)

| Setting | Default | What |
|---|---|---|
| `FEATURE_LIST` | `FEATURE_LIST_41` | Default feature set (v50 full = 41 cols) |
| `ASSET_LIST` | u10 majors | 10 fixed assets (BTC/ETH/SOL/BNB/XRP/DOGE/ADA/AVAX/LINK/LTC) |
| `REWARD_HORIZONS` | `[1, 4, 16, 64]` | Multi-horizon prediction |
| `RIDGE_ALPHAS` | `[0.01, 0.1, 1.0, 10.0, 100.0]` | α grid for Ridge sweep |
| `N_SHUFFLE_SEEDS` | 5 | ShIC replicates per (asset, horizon) |
| `PURGE_GAP_BARS` | 400 | Bars between train/val to prevent normalization leakage |
| `target_prefix` | `"target_return"` | Raw returns (NOT voladj) |

## Output

Linear baseline writes to `logs/v0/v0/linear_baseline_results.txt` with:
- Per-(asset, horizon) IC + rank_IC + dir_acc + best α + n
- Pooled ridge coefficients (feature importance)
- Feature ablation (drop-one IC delta)
- Linear shuffled IC (anti-mem floor across 200 replicates)
- Feature group masking (f37→f30→f25→... → contribution of each group)
- Aggregate summary with DL-gate margin (IC > 0.015)

## Validation gate

V0 itself doesn't have a "pass/fail" — it IS the gate. But for the DL cohort:

```
DL_gate = 0.015
margin = (linear_IC) - DL_gate
```

Margin reported at the end of linear_baseline.py output:
- `margin > 0`: linear model captures signal; DL must significantly beat
- `margin < 0`: features are weak OR non-linear modeling is essential

## Known gaps

| # | Gap | Status |
|---|---|---|
| 1 | DSR + PBO computation was missing | ✅ FIXED 2026-05-16 (`dsr_pbo.py`) |
| 2 | `FEATURE_LIST_37` hard-coded in group-masking section | OPEN — coupled to v50 schema, documented |
| 3 | `max_train_masking = 500_000` magic number | OPEN — works for u10; needs scaling for u100+ |
| 4 | No `__contract__` declaration | OPEN — cosmetic |
