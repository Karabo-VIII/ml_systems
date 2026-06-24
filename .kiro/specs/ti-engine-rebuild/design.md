# Design: TI Engine Rebuild

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  CHIMERA LOADER (per TF)                                     │
│  ChimeraLoader.load(sym, cadence) → 243-feature panel        │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  BAND ENSEMBLE (committed infra)                             │
│  select_band_trainval(ma_type, tf) → sub_band configs        │
│  make_hold_provider(sub_band, ma_type) → held(sym, OHLC)     │
│  Fixed-EW ensemble, TRAIN+VAL selected, OOS never read       │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  DECOUPLED SIGNAL + HOLD                                     │
│  Signal: band_held AND close > EMA(lookback_bars)            │
│          AND EMA(lookback) is non-falling                    │
│  Hold:   max slice_bars duration per position                │
│  Re-entry: only at slice boundary if signal confirmed        │
│  Exit: slice end OR signal lost mid-slice                    │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  WEALTH ACCOUNTING (verified, reused verbatim)               │
│  _compound(): lag-1, maker cost on turnover, MtM reconciled  │
│  Per-slice windowing: slice net = prod(1+net_in_slice) - 1   │
│  Baselines: always-long, true HODL, random-entry null        │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  PER-SLICE EVALUATION                                        │
│  7d non-overlapping slices → per-slice: net%, maxDD, trades  │
│  Aggregate: median, mean, % profitable, block-bootstrap p05  │
│  Cross-TF comparison table                                   │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Reuse committed infra (not rebuild)
- `ma_movecatch_engine.select_band_trainval()` — band selection
- `ma_movecatch_engine.make_hold_provider()` — ensemble holds
- `ma_wealth_driver._compound()` — verified wealth accounting
- `fleet_lab.load_wide()` — DEV-walled u100 panel at any TF
- `ma_type_upgrade._ema()` — causal EMA for lookback filter

### 2. Decoupled lookback vs hold (from ma_wealth_driver_decoupled.py)
- `_decoupled_held(close, band_held, lookback_bars, slice_bars)` — the core
- Signal lookback (14d) ≠ trade hold (7d)
- Recency: a signal that fired in the last 7d is a FRESH confirmation;
  a signal from 14d ago that hasn't re-confirmed is stale → exit

### 3. Per-TF bar conversion
- 30m: bars_per_day=48, lookback_7d=336bars, lookback_14d=672bars, slice_7d=336bars
- 1h: bars_per_day=24, lookback_7d=168bars, lookback_14d=336bars, slice_7d=168bars
- 2h: bars_per_day=12, lookback_7d=84bars, lookback_14d=168bars, slice_7d=84bars
- 4h: bars_per_day=6, lookback_7d=42bars, lookback_14d=84bars, slice_7d=42bars
- 1d: bars_per_day=1, lookback_7d=7bars, lookback_14d=14bars, slice_7d=7bars

### 4. Cost model
- Maker RT = 0.0006 (entry + exit = 0.0006 total, consistent with V1)
- Each ON→OFF or OFF→ON transition in the held-state = one side = 0.0003
- Higher TF frequency → more transitions per calendar time → higher cost drag
- The 1h cost cliff (D60: -0.43% OOS) was at TAKER 0.0024. At MAKER 0.0006 the
  cost is 4x lower — may be viable.

### 5. Slice evaluation methodology
- Non-overlapping 7d calendar slices (anchor: Monday to Sunday)
- Per-slice: compound of the per-bar net stream within [slice_start, slice_end)
- "Deployed" slice: has ≥1 bar with held=1 (engine was active that week)
- Metrics computed only on deployed slices (don't penalize for weeks in cash)
- Always-long baseline: same slice, same assets, held=1 for all bars

## File Structure (output)
```
crypto/runs/engines/V2_subtf_decoupled/
├── README.md (manifest)
├── cells/
│   ├── EMA_30m_7d_7d.json
│   ├── EMA_30m_14d_7d.json
│   ├── EMA_1h_7d_7d.json
│   ├── ... (one per cell)
│   └── DEMA_1d_14d_7d.json
├── summary/
│   ├── cross_tf_comparison.json
│   └── best_cells.json
└── progress_log.md (15-min updates)
```
