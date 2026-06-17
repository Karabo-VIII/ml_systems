---
name: WM U50 retrain + live integration (backlog)
description: Retrain WM ensemble on U50 universe, fix dump alignment, enable real WM predictions in the deployment stack. Queued for after 30-90 day paper trade validation.
type: project
---

## Status
**DEFERRED** — selected Path C on 2026-04-17. Current paper trader stack ships WITHOUT real WM (uses momentum proxy). WM integration is queued for after 30-90 day paper validation completes.

## Why deferred

1. Current Sharpe +3.32 bear / +2.53 bull was achieved without real WM contributing. System works.
2. WM was trained on per-dollar-bar features; current dump uses `X[:n_days]` truncation treating dollar bars as days — has alignment issue.
3. Existing dumps are U10-only. Shape `(2284, 10)` mismatches U50 panel's 39 assets → silently skipped by `_maybe_add_wm_engines`.
4. Adding a live WM with known alignment issues mid-paper-trade would contaminate validation data.

## Work items (when ready to resume)

### 1. Retrain WM on U50
- V1.0/V1.1/V1.4/V1.6/V3/V6 train scripts read from `settings.ASSET_LIST` (currently 10 assets)
- Need to either:
  - Extend `ASSET_LIST` in settings.py to U24 or U50 (requires all assets have sufficient history)
  - Add `--universe` flag to train_world_model.py
- Cost: ~10 GPU-hours per variant on RTX 4060

### 2. Fix dump alignment bug in `dump_wm_full_outputs.py`
Current code:
```python
X = load_chimera_features(asset, n_feat, max_days=n_days)
X = X[:n_days]   # ← BUG: truncates dollar bars, treats them as days
```
Should aggregate chimera dollar bars to daily first (matching `load_daily_asset_data` semantics), OR run WM on per-dollar-bar and aggregate predictions to daily afterwards.

### 3. Regenerate dumps at U50 scale
```bash
python src/analysis/dump_wm_full_outputs.py --universe 50 \
    --versions v1_0,v1_1,v1_6,v3,v6 --features 34 --horizons 1,4,16,64
```
Produces `logs/wm_full_outputs/u50/{pred_h4.npy, p_bear.npy, ...}` with shape `(n_days, 39)`.

### 4. Verify `_maybe_add_wm_engines` triggers for U50
In `integrated_walk_forward.py` line 89:
```python
if ph4.shape[1] == len(asset_names):   # 39 == 39 now
    engines.append(HorizonDivergenceEngine(...))
```
Shape match → real WM engines added alongside proxy.

### 5. Add LiveWMEngine refresh step to daily orchestrator
After U50 dumps exist, add an incremental mode to `dump_wm_full_outputs.py`:
```python
# Dump only new bars since last run, append to existing .npy files
--incremental --from-date 2026-04-17
```
Orchestrator chain becomes:
```
fetch_all → make_dataset_legacy → dump_wm_incremental → paper_trader --update   # V1-V14 retrain path; new v51 path uses make_dataset.py
```

### 6. Measure real WM LOO contribution
Re-run `bear_attribution.py` with real WM in the stack. Compare `wm_ensemble_h4` LOO delta to current momentum-proxy contribution (~+37pp historically). If real WM < proxy: investigate.

## Incomplete code in tree (inert)

`src/strategy/engine_wm.py LiveWMEngine` has a partial rewrite (batch-predict + proper asset_idx). Currently DEAD CODE — no caller constructs it. Leave in place or revert; doesn't affect production.

## Success criteria

- Real WM predictions replace momentum proxy for U10 assets in the U50 stack
- `_maybe_add_wm_engines` adds 4+ real WM engines (HorizonDivergence, WMRegime, WMResidual, KLSurprise)
- Paper-trader U50 realized Sharpe ≥ +3.0 after 30 days with real WM (empirical comparison to proxy baseline)
- No degradation in DD (target: -15% max DD maintained)
