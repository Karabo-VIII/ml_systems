# Stable-Only Basket TRAIN Sim (2026-05-23T09:24)

## Inputs
- Source: 17 STABLE engines from `engine_lifecycle_decay.parquet` (stable_flag=True)
- Engines after fire-floor filter (>=5 fires): 17
- Assets: ['APT', 'AR', 'BTC', 'DASH', 'DOT', 'FET', 'FIL', 'ICP']
- Sizing: top-3 per day at 25% NAV each, round-trip cost 0.24%
- TRAIN window: data <= 2024-05-15

## Headline
- TRAIN days with active picks: 199 / 199
- Mean day return: **-0.130%/d**
- Median day return: -0.098%/d
- Std day return: 0.115%/d
- Sharpe (annualized 252): **-17.86**
- Hit rate (positive-day on active): 7.0%
- Total NAV ROI: **-22.8%**
- Final NAV: 0.7723
- Max drawdown: -22.7%

## Compare vs 32-engine moderate-decoupling basket (J<0.50)
| Metric | Stable-only (17) | 32-engine moderate (J<0.50) | Verdict |
|---|---:|---:|---|
| Engines | 17 | 32 | smaller |
| Mean %/d | -0.130 | +0.71 | below |
| Sharpe | -17.86 | 3.46 | below |
| NAV ROI | -22.8% | +699% | below |
| Max DD | -22.7% | -24% | better |
