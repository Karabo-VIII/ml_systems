# Catch-Gap Map per (asset, regime) — TRAIN+WF only (2026-05-23T09:04)

Per (asset, BTC regime): ceiling = sum of fwd_ret_1d on top-25%-mover days; caught = max engine compound from catalog.
Unrealized = ceiling - caught (clipped at 0).

Total rows: 171

## TOP-20 (asset, regime) by UNREALIZED opportunity

| asset | bucket | regime | n_top25_days | ceiling_top25 | n_engines | max_caught | catch% | unrealized |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CRV | VOLATILE | bull | 274 | +1274.3% | 0 | +0.0% | 0.0% | +1274.3% |
| FET | DEGEN | bull | 375 | +1339.7% | 12 | +95.0% | 7.1% | +1244.8% |
| INJ | VOLATILE | bull | 277 | +1204.1% | 0 | +0.0% | 0.0% | +1204.1% |
| SOL | STEADY | bull | 284 | +1147.3% | 1 | +58.6% | 5.1% | +1088.7% |
| ZEN | VOLATILE | bull | 300 | +959.3% | 0 | +0.0% | 0.0% | +959.3% |
| AVAX | STEADY | bull | 292 | +996.2% | 1 | +38.1% | 3.8% | +958.1% |
| FET | DEGEN | chop | 278 | +979.8% | 1 | +33.2% | 3.4% | +946.5% |
| NEAR | VOLATILE | bull | 266 | +1041.4% | 5 | +104.8% | 10.1% | +936.6% |
| FET | DEGEN | bear | 244 | +971.8% | 2 | +36.4% | 3.7% | +935.4% |
| CRV | VOLATILE | chop | 214 | +919.4% | 0 | +0.0% | 0.0% | +919.4% |
| UNI | VOLATILE | bull | 306 | +895.5% | 1 | +20.7% | 2.3% | +874.8% |
| INJ | VOLATILE | bear | 178 | +862.0% | 0 | +0.0% | 0.0% | +862.0% |
| AAVE | VOLATILE | bull | 261 | +854.5% | 0 | +0.0% | 0.0% | +854.5% |
| INJ | VOLATILE | chop | 219 | +846.3% | 0 | +0.0% | 0.0% | +846.3% |
| NEAR | VOLATILE | bear | 207 | +829.5% | 0 | +0.0% | 0.0% | +829.5% |
| AR | VOLATILE | bull | 221 | +921.6% | 8 | +106.4% | 11.5% | +815.2% |
| SUPER | VOLATILE | chop | 191 | +900.2% | 2 | +88.1% | 9.8% | +812.1% |
| DOGE | VOLATILE | bear | 203 | +808.4% | 0 | +0.0% | 0.0% | +808.4% |
| SUPER | VOLATILE | bull | 218 | +1011.3% | 2 | +207.5% | 20.5% | +803.9% |
| CHZ | VOLATILE | bull | 370 | +786.5% | 0 | +0.0% | 0.0% | +786.5% |

## TOP-20 (asset, regime) by CATCH% (highest efficiency)

| asset | bucket | regime | n_engines | max_caught | ceiling | catch% |
|---|---|---|---:|---:|---:|---:|
| BTC | BLUE | bull | 7 | +32.3% | +28.1% | 100.0% |
| SHIB | DEGEN | bull | 4 | +142.5% | +424.3% | 33.6% |
| BTC | BLUE | chop | 1 | +36.6% | +118.4% | 30.9% |
| FLOKI | DEGEN | chop | 3 | +92.6% | +310.3% | 29.8% |
| ARKM | VOLATILE | bull | 2 | +104.1% | +372.9% | 27.9% |
| ICP | VOLATILE | bull | 9 | +167.0% | +634.4% | 26.3% |
| PEPE | DEGEN | bull | 1 | +115.1% | +468.2% | 24.6% |
| ARKM | VOLATILE | chop | 2 | +55.2% | +235.0% | 23.5% |
| ADA | STEADY | bull | 7 | +62.5% | +300.2% | 20.8% |
| SUPER | VOLATILE | bull | 2 | +207.5% | +1011.3% | 20.5% |
| FLOKI | DEGEN | bull | 1 | +45.9% | +243.1% | 18.9% |
| ARB | VOLATILE | bull | 1 | +35.1% | +212.9% | 16.5% |
| WLD | DEGEN | chop | 1 | +37.6% | +230.5% | 16.3% |
| BCH | STEADY | bull | 3 | +27.6% | +179.4% | 15.4% |
| SUI | VOLATILE | bull | 1 | +31.5% | +240.4% | 13.1% |
| APT | VOLATILE | bull | 12 | +62.6% | +508.3% | 12.3% |
| JST | DEGEN | bull | 4 | +48.9% | +398.9% | 12.3% |
| AR | VOLATILE | bull | 8 | +106.4% | +921.6% | 11.5% |
| DASH | DEGEN | bull | 5 | +30.4% | +281.7% | 10.8% |
| HBAR | VOLATILE | bull | 5 | +49.7% | +466.7% | 10.6% |

## Per-regime aggregate (where regime-level catch% is)

| regime | n_cohorts | n_assets_with_engines | total_ceiling | total_caught | mean_catch% |
|---|---:|---:|---:|---:|---:|
| bull | 57 | 32 | +28197% | +1893% | 8.3% |
| chop | 57 | 26 | +23761% | +902% | 4.0% |
| bear | 57 | 7 | +22917% | +136% | 0.5% |

## Per-bucket aggregate

| bucket | n_cohorts | total_ceiling | total_caught | mean_catch% |
|---|---:|---:|---:|---:|
| VOLATILE | 90 | +42096% | +1395% | 2.8% |
| STEADY | 39 | +17776% | +533% | 3.3% |
| DEGEN | 36 | +14116% | +935% | 6.0% |
| BLUE | 6 | +888% | +69% | 21.8% |

## Asset coverage gap (assets with NO engines in catalog)

- Uncovered (asset, regime) cohorts: **106**
- Total uncovered ceiling: **+40987%**

Top 15 highest-ceiling uncovered cohorts (mining priority targets):
| asset | bucket | regime | n_days | ceiling_top25 |
|---|---|---|---:|---:|
| CRV | VOLATILE | bull | 274 | +1274.3% |
| INJ | VOLATILE | bull | 277 | +1204.1% |
| ZEN | VOLATILE | bull | 300 | +959.3% |
| CRV | VOLATILE | chop | 214 | +919.4% |
| INJ | VOLATILE | bear | 178 | +862.0% |
| AAVE | VOLATILE | bull | 261 | +854.5% |
| INJ | VOLATILE | chop | 219 | +846.3% |
| NEAR | VOLATILE | bear | 207 | +829.5% |
| DOGE | VOLATILE | bear | 203 | +808.4% |
| CHZ | VOLATILE | bull | 370 | +786.5% |
| NEAR | VOLATILE | chop | 210 | +780.5% |
| CRV | VOLATILE | bear | 189 | +777.3% |
| ENJ | VOLATILE | chop | 303 | +770.8% |
| SOL | STEADY | bear | 214 | +763.3% |
| ENJ | VOLATILE | bear | 239 | +752.2% |