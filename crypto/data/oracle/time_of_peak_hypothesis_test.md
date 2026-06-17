# Time-of-Peak Hypothesis Test (F34) (2026-05-23T13:23)

## Hypothesis
- F34: Catalog was mined at the peak of asset bull runs. Engines on assets that PEAKED at TRAIN_END (= high TRAIN return) will INVERT on OOS.
- Test: correlate engine TRAIN→OOS ratio with the asset's TRAIN cumulative return.

## Correlation results
- Pearson(train_return, OOS_ratio): **r=-0.0512**, p=0.6581
- Spearman: **rho=-0.0579**, p=0.6167
- F34 hypothesis WEAK SUPPORT: correlation is small/non-significant

## By TRAIN return tier

| Tier | n_engines | mean_ratio | median_ratio | n_positive | pos_rate |
|---|---:|---:|---:|---:|---:|
| DECLINING | 16 | -0.92 | -0.55 | 4 | 25.0% |
| MODEST | 20 | -0.93 | -1.06 | 5 | 25.0% |
| STRONG | 22 | -0.67 | -0.49 | 3 | 13.6% |
| EXTREME | 19 | -0.90 | -0.77 | 2 | 10.5% |

## Per-asset survival rate (≥2 engines tested)

| Asset | TRAIN return | n_engines | n_positive | pos_rate | mean_ratio |
|---|---:|---:|---:|---:|---:|
| SOL | +4752% | 8 | 0 | 0.0% | -1.72 |
| FET | +4052% | 11 | 2 | 18.2% | -0.31 |
| ADA | +978% | 2 | 0 | 0.0% | -0.69 |
| NEAR | +938% | 4 | 0 | 0.0% | -0.40 |
| BTC | +711% | 3 | 0 | 0.0% | -1.19 |
| LINK | +510% | 4 | 0 | 0.0% | -1.38 |
| PEPE | +301% | 6 | 2 | 33.3% | -0.07 |
| WLD | +134% | 2 | 2 | 100.0% | +0.42 |
| OP | +100% | 3 | 0 | 0.0% | -1.08 |
| HBAR | +62% | 4 | 2 | 50.0% | -0.18 |
| AR | +50% | 2 | 0 | 0.0% | -1.34 |
| APT | +16% | 2 | 0 | 0.0% | -1.95 |
| ZEC | -39% | 2 | 0 | 0.0% | -3.96 |
| SUPER | -63% | 3 | 1 | 33.3% | -0.12 |
| FIL | -87% | 3 | 2 | 66.7% | -0.04 |
| ICP | -97% | 4 | 0 | 0.0% | -0.85 |