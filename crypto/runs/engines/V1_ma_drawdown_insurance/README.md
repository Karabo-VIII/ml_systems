# V1_ma_drawdown_insurance — run manifest

The canonical artifact home for the **1st VERIFIED SUCCESS** of the move-capture campaign: the **MA drawdown-insurance
sleeve** at coarse TFs (**1d / 4h**) over **u100** (56 of 100 point-in-time names; >=20 floor met). It HALVES portfolio
max-drawdown (~-35% vs buy-hold ~-82%) via genuine same-exposure timing skill. It is **capital-preservation within
long-only spot — NOT alpha, NOT a money-maker.** DEV-window incubation (band selected on 2020 TRAIN+VAL; OOS/UNSEEN
sealed) — not live-validated.

- **Success commit:** `efb23a4b` ("SUCCESS (1st verified): MA drawdown-insurance sleeve at coarse TFs (1d/4h), u100").
  State note `957b756d`; engine run artifacts `a4184bd3`; 2h TF fix `34be2be5`; capture STEP-1 `19e291a2`.
- **Full writeup (read this):** [`../../../docs/engines/V1_ma_drawdown_insurance.md`](../../../docs/engines/V1_ma_drawdown_insurance.md)
  (`crypto/docs/engines/V1_ma_drawdown_insurance.md`) — claim, methodology, the journey (incl. 2 fabrications caught),
  verification axes, out-of-fit OOS check, caveats. Engine index: `crypto/docs/engines/ENGINE_BUILDS.md`.

## Layout — what each subdir holds

| Subdir | Status | Contents | Count |
|---|---|---|---|
| `engines/` | **VERIFIED** | `engine_{EMA,SMA,WMA,HMA,DEMA,TEMA,KAMA,VIDYA}_clean.json` — the per-(type x TF) wealth cells (authoritative scalars from `ma_wealth_driver`). | 8 |
| `verification/` | **VERIFIED** | The 3-axis red-team: `_skeptic1_exposure_{DEMA_1d,EMA_1d,EMA_4h,SMA_1d,SMA_4h}.json` (exposure-matched skill control), `_skeptic2_results.json` (leak/survivorship/cost), `_skeptic3_per_event_dd.json` (per-event drawdown robustness). | 7 |
| `step1/` | **VERIFIED** | `step1_ema_1d_7d.json` — the capture STEP-1 cell (held_cross alignment, capture-rate, band sizes). | 1 |
| `slices/` | **VERIFIED** | Forensic per-setup slices: `forensic_slice_bull_2021-01.csv`, `forensic_slice_oos_val_2024-03-25_2024-03-31.{csv,json}`, `slices_within_window.{csv,json}`. | 5 |
| `outoffit/` | **VERIFIED** | `outoffit_2021_2024-05.json` — the out-of-fit (2020-excluded) slice of the verified curves; shows the full-window compound is partly the compounded 2020 bull (the overclaim to kill). | 1 |
| `charts/` | supporting (untracked) | `portfolio_equity_*.png`, `trade_<SYM>_*.png` (6 names), `trades_*.csv`, `trade_summary_*.csv`, and `slices/` (per-regime + forensic PNGs). | 10 + slices/ |
| `open_unverified/` | **UN-VERIFIED — verify before use** | `decoupled_7slice_14lb_*.json` (16; 1d + 4h per type), `val_forward_*.json` (8; held-out forward windows), `_decoupled_compare_{1d,4h}.json` (2). These are 7d-slice / 14d-lookback and held-out-forward probes that have NOT passed the adversarial verification the `engines/` + `verification/` cells did. Do NOT cite as results. | 26 |

**VERIFIED vs UN-VERIFIED:** everything outside `open_unverified/` carries the red-teamed GO (engines / verification /
step1 / slices / outoffit). `open_unverified/` is on-disk exploratory output pending verification — boxed, never headline.

## Canonical inputs that stay ELSEWHERE (not relocated — documented here)

These are shared / canonical and are deliberately NOT moved into this engine home:

- **Band-grid leaderboard (the config source):** `runs/periods/TRAIN/2020/DEEP_DIVE/config_leaderboard.json`
  (+ `robust_ma_runners.json`). Trade the fixed-EW ENSEMBLE of `band.band_configs` per (type, TF) — NEVER `ranked[0]`
  (hindsight). Selection reads only the TRAIN/VAL net field; OOS never read in development.
- **Drivers (`src/strat/`):** `ma_wealth_driver.py` (wealth, authoritative — the single verified wealth path),
  `ma_exposure_control.py` (exposure-matched skill control), `ma_movecatch_engine.py` (band selection + holds),
  `capture_lab.py` (capture STEP-1). Plus the decoupled/val-forward profilers
  (`ma_val_forward.py`, `ma_wealth_driver_decoupled.py`, `ma_decoupled_trade_charts.py`).
- **Forensic-slice scripts (`scripts/`):** `scripts/forensic_slice_bull_2021.py`, `scripts/forensic_slice_oos_val.py`,
  `scripts/outoffit_slice.py`.

## Reproduce (run from `crypto/`)

```bash
# (1) capture STEP-1  -> step1/step1_ema_1d_7d.json
python -m strat.capture_lab --selftest --tf 4h

# (2) wealth cells (authoritative) -> engines/engine_<TYPE>_clean.json   (per type x {1d,4h})
python -m strat.ma_wealth_driver               # writes engine_<TYPE>_clean.json

# (3) red-team verification -> verification/_skeptic{1,2,3}_*.json
python -m strat.ma_exposure_control            # -> _skeptic1_exposure_<TYPE>_<TF>.json (skill control)
#    _skeptic2_results.json (leak/survivorship/cost) + _skeptic3_per_event_dd.json (per-event dd)

# (4) forensic slices -> slices/*  + charts/slices/*
python scripts/forensic_slice_bull_2021.py     # -> forensic_slice_bull_2021-01.csv + slices_within_window.*
python scripts/forensic_slice_oos_val.py       # -> forensic_slice_oos_val_2024-03-25_2024-03-31.*

# (5) out-of-fit (2020-excluded) slice -> outoffit/outoffit_2021_2024-05.json
python scripts/outoffit_slice.py
```

Exact output paths are configured in the drivers; see `crypto/docs/engines/V1_ma_drawdown_insurance.md` §8 for the full
reproduce block and §8.5 for this layout.
