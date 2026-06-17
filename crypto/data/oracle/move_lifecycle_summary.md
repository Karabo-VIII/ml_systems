# Move-Lifecycle Signature Mining — Summary

- TRAIN window: dates <= 2024-05-15
- Movers definition: top-25% per day by raw close-to-close fwd_ret_1d (cross-sectional rank across 5 assets)
- Measures scanned: 3 columns from families ['norm_', 'liq_', 'wh_', 'etf_', 'bd_', 'lob_', 'hbr_', 'te_', 'xd_', 'stbl_', 'bs_', 'rv_', 'xex_', 'xrel_', 's3_', 'soc_']
- Lags scanned: PRE = [1, 3, 6, 12], IN = 0, EXIT horizon = +1..+3d (CONT/REV thresholds = +/-3%)
- Folds: F1=(datetime.date(2023, 7, 1), datetime.date(2023, 10, 31)), F2=(datetime.date(2023, 11, 1), datetime.date(2024, 2, 29)), F3=(datetime.date(2024, 3, 1), datetime.date(2024, 5, 15))

## PRE-move (t-N) -- Top z_lift by (measure, lag), aggregated across assets

### Lag = t-1 (top 10 by |median z_lift|)

| measure | median z_lift | mean z_lift | n_assets | n_3fold_consistent |
|---|---|---|---|---|
| etf_any_inflow_shock | -0.0218 | 0.0018 | 5 | 0 |
| norm_bar_duration | 0.0108 | -0.0027 | 5 | 1 |
| rv_bpv_5m | -0.0038 | 0.0044 | 5 | 2 |

### Lag = t-3 (top 10 by |median z_lift|)

| measure | median z_lift | mean z_lift | n_assets | n_3fold_consistent |
|---|---|---|---|---|
| norm_bar_duration | -0.0450 | -0.0531 | 5 | 2 |
| rv_bpv_5m | -0.0248 | -0.0149 | 5 | 0 |
| etf_any_inflow_shock | -0.0200 | 0.0066 | 5 | 0 |

### Lag = t-6 (top 10 by |median z_lift|)

| measure | median z_lift | mean z_lift | n_assets | n_3fold_consistent |
|---|---|---|---|---|
| etf_any_inflow_shock | 0.0218 | -0.0014 | 5 | 0 |
| norm_bar_duration | -0.0029 | 0.0196 | 5 | 1 |
| rv_bpv_5m | -0.0004 | 0.0016 | 5 | 0 |

### Lag = t-12 (top 10 by |median z_lift|)

| measure | median z_lift | mean z_lift | n_assets | n_3fold_consistent |
|---|---|---|---|---|
| etf_any_inflow_shock | -0.0354 | -0.0019 | 5 | 0 |
| rv_bpv_5m | -0.0228 | -0.0160 | 5 | 1 |
| norm_bar_duration | -0.0209 | -0.0334 | 5 | 2 |

## IN-move (t-0) -- Top concurrent measures, aggregated across assets

| measure | median z_lift | mean z_lift | n_assets | n_3fold_consistent |
|---|---|---|---|---|
| norm_bar_duration | -0.0357 | -0.0212 | 5 | 4 |
| etf_any_inflow_shock | -0.0286 | 0.0050 | 5 | 0 |
| rv_bpv_5m | -0.0093 | 0.0069 | 5 | 3 |

## EXIT-move -- Top CONTINUES-vs-REVERTS lift at t-0 (measures that predict continuation)

| measure | median lift (CONT-REV) | mean lift | n_assets | tot_cont | tot_rev |
|---|---|---|---|---|---|
| etf_any_inflow_shock | 0.2609 | 0.2335 | 5 | 1207 | 587 |
| norm_bar_duration | -0.0700 | -0.0533 | 5 | 1207 | 587 |
| rv_bpv_5m | -0.0293 | 0.0217 | 5 | 1207 | 587 |

## 3-Fold WF-Stable Signatures (sign consistent across F1/F2/F3 on >=20 assets)

### PRE-move (3-fold consistent, n_assets >= 20)

| measure | lag | n_consistent | median z_lift |
|---|---|---|---|

### IN-move (3-fold consistent, n_assets >= 20)

| measure | n_consistent | median z_lift |
|---|---|---|

## Narrative Findings

**LEADING (t-3) — strongest precursors 3 days BEFORE the move:**
- `norm_bar_duration` median z_lift = -0.0450
- `rv_bpv_5m` median z_lift = -0.0248
- `etf_any_inflow_shock` median z_lift = -0.0200

**LEADING (t-6) — strongest precursors 6 days BEFORE the move:**
- `etf_any_inflow_shock` median z_lift = +0.0218
- `norm_bar_duration` median z_lift = -0.0029
- `rv_bpv_5m` median z_lift = -0.0004

**CONCURRENT (t-0) — strongest same-day signatures:**
- `norm_bar_duration` median z_lift = -0.0357
- `etf_any_inflow_shock` median z_lift = -0.0286
- `rv_bpv_5m` median z_lift = -0.0093

**EXIT PREDICTORS — measures whose t-0 value distinguishes CONTINUES vs REVERTS:**
- `etf_any_inflow_shock` median lift = +0.2609
- `norm_bar_duration` median lift = -0.0700
- `rv_bpv_5m` median lift = -0.0293

## Methodology Notes
- `fwd_ret_1d` computed as close[t+1]/close[t] - 1 directly (the `target_return_1` column is volatility-normalized in this dataset, so unsuitable as a raw mover definition).
- Mover flag is cross-sectional: top-25% of all assets that have a valid `fwd_ret_1d` on each date.
- `z_lift` per asset normalized by per-asset measure-column std (so the cross-asset aggregate medians are comparable in z-units).
- No look-ahead: lag values are `.shift(N)` (so lag-1 references the value from the prior bar; lag-0 = same-day state at the bar; mover label uses `close[t+1]` but the measure value being z-lift'd is from earlier bars only).
- Sign-consistency requires non-zero, finite z-lift in all three folds with identical sign.