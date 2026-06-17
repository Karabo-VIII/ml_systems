# Engine Consensus Signal (2026-05-23T09:36)

- TRAIN cutoff: 2024-05-15
- Catch-tier engines: 234
- Unique (asset, date) cells with >=1 fire: 4675
- Total panel cells (with returns): 46400
- n_engines_consensus distribution: max=276, median_nonzero=6

## Realized fwd_ret_1d by consensus bucket

| consensus | n_cells | mean | median | std | hit_rate | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 41725 | +0.009% | +0.000% | 0.45% | 48.3% | -0.36% | +0.38% |
| 1 | 330 | -0.033% | -0.013% | 0.23% | 40.9% | -0.28% | +0.20% |
| 2 | 233 | +0.016% | +0.000% | 0.26% | 48.1% | -0.26% | +0.30% |
| 3-4 | 1368 | +0.009% | +0.000% | 0.27% | 46.2% | -0.29% | +0.32% |
| 5-7 | 426 | -0.008% | +0.000% | 0.27% | 45.8% | -0.34% | +0.33% |
| 8-11 | 517 | +0.013% | +0.000% | 0.27% | 46.6% | -0.31% | +0.36% |
| 12+ | 1801 | -0.002% | +0.000% | 0.32% | 46.8% | -0.30% | +0.32% |

## Headline

- Consensus 1: mean = -0.033%, lift vs no-fire = -0.042pp (-3.45x baseline mean)
- Consensus 2: mean = +0.016%, lift vs no-fire = +0.006pp (1.66x baseline mean)
- Consensus 3-4: mean = +0.009%, lift vs no-fire = -0.000pp (0.98x baseline mean)
- Consensus 5-7: mean = -0.008%, lift vs no-fire = -0.017pp (-0.82x baseline mean)
- Consensus 8-11: mean = +0.013%, lift vs no-fire = +0.004pp (1.41x baseline mean)
- Consensus 12+: mean = -0.002%, lift vs no-fire = -0.012pp (-0.22x baseline mean)

Interpretation: if mean(ret_fwd_1d) grows monotonically with consensus, then multi-engine consensus is a stronger signal than single-engine fire — composition interaction is real. Cost-aware basket sizing should overweight high-consensus cells.