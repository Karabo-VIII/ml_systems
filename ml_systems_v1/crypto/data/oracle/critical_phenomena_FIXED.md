# F18 Critical-Phenomena FIXED (2026-05-23T11:25)

## Data-bug correction note
- Prior mine_critical_phenomena_engines.py used `returns_clean` (CORRUPTED in chimera v51).
- This script uses `close.pct_change()` directly.

- BTC regime flips: **11**

## Universal early-warning observables (median across assets)

| observable_K | threshold | n_assets | median_precision | median_lift |
|---|---:|---:|---:|---:|
| ac1_K10 | 0.3 | 56 | 0.102 | 1.39x |
| ac1_K10 | 0.4 | 53 | 0.069 | 0.99x |
| ac1_K20 | 0.3 | 56 | 0.177 | 1.33x |
| ac1_K20 | 0.4 | 53 | 0.154 | 1.09x |
| vsusc_K10 | 1.3 | 56 | 0.020 | 0.27x |
| vsusc_K10 | 1.5 | 56 | 0.000 | 0.00x |
| vsusc_K20 | 1.3 | 56 | 0.046 | 0.35x |
| vsusc_K20 | 1.5 | 56 | 0.018 | 0.14x |

## Top 15 per-asset SPECIALIST early-warning rules

| asset | observable | threshold | K | precision | lift | n_trig |
|---|---|---:|---:|---:|---:|---:|
| MOVR | ac1_14 | 0.4 | 10 | 0.429 | 6.57x | 7 |
| UNI | ac1_14 | 0.4 | 10 | 0.421 | 5.63x | 19 |
| UNI | ac1_14 | 0.3 | 10 | 0.344 | 4.60x | 32 |
| FLOKI | ac1_14 | 0.4 | 10 | 0.333 | 4.19x | 9 |
| UNI | ac1_14 | 0.4 | 20 | 0.579 | 4.14x | 19 |
| TRX | ac1_14 | 0.4 | 10 | 0.286 | 4.14x | 14 |
| TRX | ac1_14 | 0.4 | 10 | 0.286 | 4.14x | 14 |
| NEAR | ac1_14 | 0.4 | 10 | 0.308 | 3.99x | 13 |
| NEAR | ac1_14 | 0.3 | 10 | 0.298 | 3.87x | 57 |
| FLOKI | ac1_14 | 0.3 | 10 | 0.308 | 3.87x | 26 |
| XRP | ac1_14 | 0.3 | 10 | 0.268 | 3.86x | 41 |
| XRP | ac1_14 | 0.3 | 10 | 0.268 | 3.86x | 41 |
| CHZ | ac1_14 | 0.4 | 20 | 0.500 | 3.71x | 32 |
| ENJ | ac1_14 | 0.3 | 10 | 0.263 | 3.70x | 57 |
| MOVR | ac1_14 | 0.4 | 20 | 0.429 | 3.68x | 7 |

## Comparison vs OLD F18 (corrupted returns_clean)

- JST vsusc > 1.5 @ K=20: OLD prec=0.627 / lift=3.84x — NEW prec=0.123 / lift=0.91x / n_trig=227
- WLD ac1_14 > 0.4 @ K=20: OLD prec=0.583 / lift=3.57x — NEW prec=0.200 / lift=1.14x / n_trig=5