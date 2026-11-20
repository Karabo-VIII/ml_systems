# F39 Decile-3 Basket Sim (2026-05-23T13:42)

## Basket: 15 engines (7 LINK RSI chop + 4 JST RSI chop + 3 APT RSI bull + 1 DASH OBV chop)
Methodology: top-3 picks/day by n_engines firing, 25% sizing each, 1d hold, per-bucket maker cost

| Window | n_days | mean %/d | Sharpe | hit% | NAV % | maxDD % |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 324 | -0.123 | -1.12 | 50.6 | -36.0 | -49.4 |
| VAL | 68 | -0.128 | -1.16 | 45.6 | -9.3 | -19.3 |
| OOS_pre | 45 | +0.660 | 4.22 | 53.3 | +32.7 | -4.7 |
| UNSEEN | 30 | -0.085 | -1.00 | 56.7 | -2.8 | -7.8 |
| FULL_POST_TRAIN | 143 | +0.129 | 1.04 | 50.3 | +17.0 | -19.3 |