# V0 Baselines -- Usage Guide

## 2026-05-02 Frontier-ML upgrade status: **n/a (non-DL)**

V0 is non-DL baselines (XGBoost / linear / persistence) used to define
the IC floor. Frontier-ML upgrades (SAM, PCGrad, MTP, MDN, etc.) target
neural architectures and don't apply here. V0 stays as-is. See
[../UPGRADE_INVENTORY_2026_05_02.md](../UPGRADE_INVENTORY_2026_05_02.md)
for the cross-version matrix.

## Purpose

Non-DL baselines that define the IC floor and ceiling. Every world model (V1-V9) must beat these to justify its complexity.

| Baseline | Function Shape | What it Tests |
|----------|---------------|---------------|
| Linear Ridge | y = Xw + b | Do features carry signal at all? |
| Polynomial Ridge | y = sum(w_ij * x_i * x_j) + ... | Do feature interactions matter? |
| Gradient Boosted Trees | Ensemble of axis-aligned splits | Can non-linear splits beat linear? |
| MLP (2-layer) | Smooth non-linear | Fairest DL comparison (same features, no temporal modeling) |

## Quick Start

```powershell
# Linear baseline (fastest)
python src/wm/v0/v0_baseline/linear_baseline.py --features 37

# All 3 non-linear baselines
python src/wm/v0/v0_baseline/nonlinear_baselines.py --features 37

# Individual non-linear model
python src/wm/v0/v0_baseline/nonlinear_baselines.py --model gbt
python src/wm/v0/v0_baseline/nonlinear_baselines.py --model mlp
python src/wm/v0/v0_baseline/nonlinear_baselines.py --model poly

# With 13 features (V1.0 legacy comparison)
python src/wm/v0/v0_baseline/nonlinear_baselines.py --features 13
```

## Scripts

| Script | Models | Runtime (10 assets) |
|--------|--------|-------------------|
| `linear_baseline.py` | Ridge regression | ~30 sec |
| `nonlinear_baselines.py` | Polynomial + GBT + MLP | ~30-60 min |
| `save_baseline_preds.py` | K-fold OOF GBT predictions | ~20-40 min |

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--features` | 13 / 18 / 21 / 25 / 30 / 37 | 37 |
| `--model` | poly / gbt / mlp (nonlinear only) | all 3 |
| `--full` | Simple 90/10 split (no purge gap) | walk-forward |

## Feature Counts

- **13**: Legacy V1.0 base features (norm_deviation through norm_spread_bps)
- **18**: Extended base (+ma_distance, whale, efficiency, return_4, return_16)
- **21**: +Tier 1 (return_kurtosis, bar_duration, funding_momentum)
- **25**: +Hawkes (intensity, buy/sell intensity, imbalance)
- **30**: +IC-boost Tier 2 (momentum_accel, vol_price_corr, vol_ratio, flow_persistence, oi_price_divergence)
- **34**: +SOTA Tier 3 (yz_volatility, cs_spread, perm_entropy, kyle_lambda)
- **37**: Legacy 30 base + 7 XD (backward compat, skips SOTA features)
- **41**: Full (34 base + 7 cross-asset XD features)

## What Each Model Reports

Per asset, per horizon (t+1, t+4, t+16, t+64):
- **IC**: Pearson correlation (prediction vs actual)
- **Rank IC**: Spearman rank correlation
- **Dir**: Directional accuracy (sign agreement %)
- **Shuffled IC**: IC after temporal shuffling (anti-memorization)

Plus feature group masking, ablation, and cross-asset summary.

## Interpretation Guide

| Linear IC vs DL IC | Meaning |
|-------------------|---------|
| DL >> Linear | Temporal/non-linear modeling justified |
| DL ~ Linear | Architecture adds no value, features do the work |
| DL < Linear | DL is overfitting; simpler model is better |

| Non-linear IC vs Linear IC | Meaning |
|---------------------------|---------|
| Non-linear >> Linear | Feature interactions carry signal |
| Non-linear ~ Linear | Signal is mostly linear |
| Non-linear < Linear | Non-linear overfits; regularization issue |

## Output

- `logs/v0/v0/linear_baseline_results.txt`
- `logs/v0/v0/nonlinear_baselines_results.txt`
