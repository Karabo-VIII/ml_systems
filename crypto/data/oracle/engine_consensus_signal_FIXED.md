# Engine Consensus Signal FIXED (2026-05-23T10:46)

## Data-bug correction note
- Prior mine_engine_consensus.py used `target_return_1_raw` (CORRUPTED).
- This version uses `close.pct_change().shift(-1)` (real close-to-close return).

- TRAIN cutoff: 2024-05-15
- Catch-tier engines: 234
- Fire-cells (asset, date) with >=1 fire: 4675
- Total panel cells: 46400
- n_engines max: 276

## fwd_ret_1d (close-derived) by consensus bucket

| consensus | n_cells | mean % | median % | std % | hit% | p10 % | p90 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 41725 | +0.205 | +0.002 | 6.59 | 50.0 | -6.15 | +6.34 |
| 1 | 330 | +0.447 | +0.127 | 4.24 | 53.0 | -3.64 | +4.58 |
| 2 | 252 | +0.392 | +0.317 | 3.47 | 55.6 | -3.02 | +4.03 |
| 3-4 | 1383 | +1.025 | +0.265 | 6.92 | 52.8 | -4.78 | +6.56 |
| 5-7 | 438 | +0.796 | +0.233 | 6.08 | 52.3 | -5.05 | +6.32 |
| 8-11 | 543 | +0.759 | +0.280 | 6.12 | 52.7 | -5.48 | +6.98 |
| 12+ | 1729 | +0.807 | +0.470 | 5.53 | 53.6 | -4.85 | +6.60 |

## Lift vs no-fire baseline

- Consensus 1: mean = +0.447%, lift = +0.242pp, ratio = 2.18x
- Consensus 2: mean = +0.392%, lift = +0.186pp, ratio = 1.91x
- Consensus 3-4: mean = +1.025%, lift = +0.820pp, ratio = 5.00x
- Consensus 5-7: mean = +0.796%, lift = +0.590pp, ratio = 3.88x
- Consensus 8-11: mean = +0.759%, lift = +0.553pp, ratio = 3.70x
- Consensus 12+: mean = +0.807%, lift = +0.602pp, ratio = 3.93x

## Headline

- Consensus is NOT strictly monotonic. The simple 'more engines = more conviction' heuristic does NOT hold uniformly.
- BUT: max consensus-bucket beats baseline by +0.82pp — a real composition advantage exists at SOME consensus level.