> **SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup)**: This document predates the
> canonical split-discipline gate. References to "OOS" in this file may include data
> from the canonical UNSEEN window (>=2026-01-01) per [src/split_config.py](../../src/split_config.py).
> Use this document for historical context only; deploy decisions citing UNSEEN-relevant
> claims must be re-derived from the canonical segments.

# OOS-Validated Deploy Basket Sim (2026-05-23T13:17)

## Basket members (5 engines, OOS-positive each per F29/F30)
- FET | measure_engines/wh_whale_net_usd | op_abs_gt_thr_2.0 | bull
- PEPE | Donchian_state_above_midline | period_55 | chop
- FIL | measure_engines/liq_long_usd | op_gt_thr_1.0 | bull
- SUPER | measure_engines/norm_deviation | op_abs_gt_thr_1.0 | chop
- WLD | MA_state_SMA_above | period_50 | chop

## Sim methodology
- top-3 picks per day by n_engines firing, 25% sizing per pick, 1d hold
- Cost: per-bucket maker (BLUE 28bp, STEADY 32bp, VOLATILE 36bp, DEGEN 44bp)
- Returns: close.pct_change().shift(-1) (FIXED)
- Signal: re-derived from chimera (60-day rolling z-score for measure engines; Donchian midline; MA_state)

## Results

| Window | n_days | mean %/d | Sharpe | hit% | NAV % | maxDD % |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN_<=2024-05-15 | 225 | +0.206 | 1.35 | 52.9 | +48.8 | -17.7 |
| VAL_2024-05-16_to_2025-03-15 | 80 | -0.029 | -0.16 | 46.2 | -5.3 | -31.5 |
| OOS_2025-03-16_to_2025-12-31 | 60 | +0.541 | 2.38 | 51.7 | +33.4 | -12.6 |
| UNSEEN_2026-01-01_to_2026-05-19 | 26 | -0.412 | -6.54 | 42.3 | -10.3 | -10.6 |
| FULL_POST_TRAIN | 166 | +0.117 | 0.63 | 47.6 | +13.3 | -31.5 |

## TRAIN→OOS deflation factor: **0.57**
## TRAIN→UNSEEN deflation factor: **-2.00** (most recent ~5 months)
- TRAIN: +0.206%/d / Sharpe 1.35
- OOS: +0.117%/d / Sharpe 0.63

**VERDICT: positive OOS but Sharpe 0.63 below 1.0 — marginal deploy**