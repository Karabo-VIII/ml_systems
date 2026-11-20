# LINK RSI Chop Full Audit (2026-05-23T13:28)

## Per-config per-window results (LINK RSI chop, 32bp RT cost)

| Config | TRAIN | VAL | OOS_pre | UNSEEN |
|---|---|---|---|---|
| p_8_lo_40_hi_60 | +39.2% (n=65, hit 57%) | +12.3% (n=9, hit 56%) | +13.2% (n=13, hit 54%) | -3.5% (n=11, hit 64%) |
| p_5_lo_40_hi_60 | -47.2% (n=79, hit 38%) | -5.2% (n=16, hit 38%) | -25.4% (n=15, hit 33%) | -1.2% (n=9, hit 56%) |
| p_6_lo_40_hi_60 | -43.4% (n=67, hit 45%) | -12.9% (n=13, hit 38%) | +5.3% (n=14, hit 57%) | +3.7% (n=12, hit 75%) |
| p_7_lo_40_hi_60 | +11.4% (n=72, hit 51%) | -6.1% (n=7, hit 43%) | -23.0% (n=13, hit 15%) | +2.3% (n=8, hit 75%) |
| p_5_lo_35_hi_60 | -51.3% (n=73, hit 37%) | -7.2% (n=16, hit 38%) | -16.5% (n=14, hit 43%) | +2.3% (n=8, hit 62%) |
| p_6_lo_35_hi_60 | -27.4% (n=61, hit 46%) | +11.1% (n=8, hit 50%) | -2.6% (n=16, hit 56%) | +0.8% (n=11, hit 73%) |
| p_7_lo_35_hi_60 | -1.6% (n=63, hit 52%) | +16.0% (n=7, hit 57%) | -4.8% (n=11, hit 36%) | +0.8% (n=8, hit 62%) |

## Aggregate per-window (across all 7 LINK RSI chop configs)

| Window | n_configs_with_fires | mean_compound |
|---|---:|---:|
| TRAIN | 7 | -17.2% |
| VAL | 7 | +1.1% |
| OOS_pre | 7 | -7.7% |
| UNSEEN | 7 | +0.7% |

## Honest verdict

- LINK RSI chop SURVIVES UNSEEN with mean compound +0.7%