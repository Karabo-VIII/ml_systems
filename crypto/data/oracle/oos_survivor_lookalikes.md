> **SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup)**: This document predates the
> canonical split-discipline gate. References to "OOS" in this file may include data
> from the canonical UNSEEN window (>=2026-01-01) per [src/split_config.py](../../src/split_config.py).
> Use this document for historical context only; deploy decisions citing UNSEEN-relevant
> claims must be re-derived from the canonical segments.

# OOS-Survivor Lookalikes (2026-05-23T13:12)

## Filter
- hit_rate in [0.54, 0.95]
- n_fires in [8, 25]
- mean_pnl_pct in [1.0, 5.5]
- compound_return_pct < 100%
- indicator_class IN ['YZ_vol_regime', 'Donchian_state_above_midline', 'measure_engines/liq_long_usd', 'measure_engines/wh_whale_net_usd', 'MA_state_SMA_above', 'MA_state_EMA_above', 'measure_engines/hbr_eta_buy', 'measure_engines/norm_deviation', 'measure_engines/bd_imbalance_l5']
- catch_tier_eligibility = True
- NOT already OOS-tested in F29 set (61 engines)

## Result: 16 candidate engines (NOT yet OOS-tested)

| Asset | Class | Config | Regime | hit | n_fires | mean_pnl | compound |
|---|---|---|---|---:|---:|---:|---:|
| DYDX | MA_state_EMA_above | period_20 | chop | 0.700 | 10 | 1.26% | +12.5% |
| SOL | Donchian_state_above_midline | period_20 | chop | 0.571 | 14 | 1.36% | +17.6% |
| SOL | MA_state_SMA_above | period_20 | chop | 0.571 | 14 | 1.36% | +17.6% |
| PEPE | MA_state_EMA_above | period_100 | chop | 0.643 | 14 | 1.68% | +23.3% |
| UNI | MA_state_EMA_above | period_20 | chop | 0.696 | 23 | 1.02% | +24.7% |
| OP | MA_state_SMA_above | period_20 | chop | 0.700 | 20 | 1.45% | +31.2% |
| HBAR | measure_engines/norm_deviation | op_abs_gt_thr_1.5 | chop | 0.727 | 11 | 2.63% | +32.4% |
| SUPER | measure_engines/norm_deviation | op_abs_gt_thr_1.0 | chop | 0.692 | 13 | 2.28% | +33.2% |
| BTC | MA_state_SMA_above | period_50 | chop | 0.800 | 20 | 1.60% | +36.6% |
| WLD | MA_state_SMA_above | period_50 | chop | 0.600 | 15 | 2.24% | +37.6% |
| PEPE | measure_engines/bd_imbalance_l5 | op_gt_thr_1.0 | chop | 0.714 | 14 | 2.37% | +38.1% |
| OP | MA_state_EMA_above | period_20 | chop | 0.700 | 20 | 1.73% | +39.1% |
| OP | Donchian_state_above_midline | period_20 | chop | 0.737 | 19 | 1.87% | +40.4% |
| BTC | MA_state_EMA_above | period_20 | bull | 0.727 | 11 | 1.22% | +13.8% |
| FIL | measure_engines/bd_imbalance_l5 | op_abs_gt_thr_1.0 | bull | 0.643 | 14 | 2.18% | +31.1% |
| FET | measure_engines/wh_whale_net_usd | op_abs_gt_thr_2.0 | bull | 0.545 | 11 | 3.74% | +45.7% |

## Regime distribution: {'chop': 13, 'bull': 3}

## Class distribution (top 10):
```
  MA_state_EMA_above: 5
  MA_state_SMA_above: 4
  Donchian_state_above_midline: 2
  measure_engines/norm_deviation: 2
  measure_engines/bd_imbalance_l5: 2
  measure_engines/wh_whale_net_usd: 1
```


## Asset distribution (top 20):
```
  OP: 3
  SOL: 2
  PEPE: 2
  BTC: 2
  DYDX: 1
  UNI: 1
  HBAR: 1
  SUPER: 1
  WLD: 1
  FIL: 1
  FET: 1
```

## Next step
Run validate_engines_oos.py on this lookalike set to test if the OOS survivor profile heuristic predicts OOS survival on a held-out engine sample.