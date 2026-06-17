# Listwise Top-K Mining (2026-05-23T09:22)

- TRAIN panel: 111724 rows, ~2329 dates, median 41 assets/day
- Measures scanned: 172
- K thresholds: [5, 10, 22]
- Random baseline @ K=22 = 0.537

## Top 30 listwise POSITIVE-rank predictors @ K=22 (top-22 by measure → realized top-22 mover)

| measure | prec@22 | lift@22 | prec@10 | lift@10 | prec@5 | lift@5 | n_obs |
|---|---:|---:|---:|---:|---:|---:|---:|
| soc_wiki_views | 1.000 | 1.86x | 1.000 | 4.10x | 0.502 | 4.12x | 8690 |
| xex_cb_bn_spread_bps | 1.000 | 1.86x | 1.000 | 4.10x | 1.000 | 8.20x | 4345 |
| xex_by_bn_spread_bps | 1.000 | 1.86x | 1.000 | 4.10x | 1.000 | 8.20x | 4345 |
| xex_ok_bn_spread_bps | 1.000 | 1.86x | 1.000 | 4.10x | 1.000 | 8.20x | 4345 |
| bs_basis_frenzy | 0.712 | 1.33x | 0.618 | 2.53x | 0.409 | 3.36x | 100692 |
| bs_basis_panic | 0.709 | 1.32x | 0.537 | 2.20x | 0.274 | 2.25x | 100692 |
| bs_basis_bear_shock | 0.700 | 1.31x | 0.432 | 1.77x | 0.137 | 1.12x | 100692 |
| bs_basis_bull_shock | 0.700 | 1.30x | 0.431 | 1.77x | 0.129 | 1.06x | 100692 |
| liq_capitulation | 0.698 | 1.30x | 0.424 | 1.74x | 0.128 | 1.05x | 102834 |
| liq_short_panic | 0.695 | 1.30x | 0.412 | 1.69x | 0.121 | 0.99x | 102834 |
| bd_n_snapshots | 0.692 | 1.29x | 0.191 | 0.78x | 0.235 | 1.93x | 69692 |
| liq_long_spike | 0.691 | 1.29x | 0.384 | 1.57x | 0.115 | 0.94x | 102834 |
| liq_short_spike | 0.688 | 1.28x | 0.375 | 1.54x | 0.114 | 0.94x | 102834 |
| bd_thin_book_frac | 0.685 | 1.28x | 0.215 | 0.88x | 0.127 | 1.04x | 69424 |
| xrel_wh_whale_net_usd_xpct10 | 0.654 | 1.22x | 0.266 | 1.09x | 0.067 | 0.55x | 103477 |
| xrel_liq_long_usd_xpct10 | 0.650 | 1.21x | 0.233 | 0.96x | 0.042 | 0.34x | 103477 |
| stbl_stable_crash | 0.645 | 1.20x | 0.554 | 2.27x | 0.770 | 6.32x | 111637 |
| stbl_stable_shock_strong | 0.645 | 1.20x | 0.554 | 2.27x | 0.770 | 6.32x | 111637 |
| stbl_usdt_shock | 0.645 | 1.20x | 0.554 | 2.27x | 0.762 | 6.25x | 111637 |
| stbl_compound_shock | 0.645 | 1.20x | 0.554 | 2.27x | 0.758 | 6.22x | 111637 |
| stbl_stable_shock | 0.645 | 1.20x | 0.554 | 2.27x | 0.758 | 6.22x | 111637 |
| stbl_usdt_zscore_30d | 0.645 | 1.20x | 0.554 | 2.27x | 0.750 | 6.15x | 111637 |
| stbl_total_delta_30d_pct | 0.645 | 1.20x | 0.554 | 2.27x | 0.742 | 6.09x | 111637 |
| stbl_dai_zscore_30d | 0.645 | 1.20x | 0.554 | 2.27x | 0.742 | 6.09x | 111637 |
| stbl_total_delta_7d_pct | 0.645 | 1.20x | 0.554 | 2.27x | 0.735 | 6.03x | 111637 |
| stbl_usdt_delta_7d_pct | 0.645 | 1.20x | 0.554 | 2.27x | 0.731 | 6.00x | 111637 |
| stbl_usdc_zscore_30d | 0.645 | 1.20x | 0.554 | 2.27x | 0.723 | 5.93x | 111637 |
| stbl_total_zscore_30d | 0.645 | 1.20x | 0.554 | 2.27x | 0.716 | 5.87x | 111637 |
| s3_smart_extreme_short | 0.640 | 1.19x | 0.267 | 1.09x | 0.183 | 1.50x | 77117 |
| xrel_rv_rv_5m_xpct10 | 0.619 | 1.15x | 0.268 | 1.10x | 0.148 | 1.21x | 111637 |

## Top 30 listwise NEGATIVE-rank predictors @ K=22 (bot-22 by measure → realized top-22 mover)

| measure | prec@22 | lift@22 | prec@10 | lift@10 | prec@5 | lift@5 | n_obs |
|---|---:|---:|---:|---:|---:|---:|---:|
| soc_wiki_views | 1.000 | 1.86x | 1.000 | 4.10x | 0.495 | 4.06x | 8690 |
| xex_ok_bn_spread_bps | 1.000 | 1.86x | 1.000 | 4.10x | 1.000 | 8.20x | 4345 |
| xex_cb_bn_spread_bps | 1.000 | 1.86x | 1.000 | 4.10x | 1.000 | 8.20x | 4345 |
| xex_by_bn_spread_bps | 1.000 | 1.86x | 1.000 | 4.10x | 1.000 | 8.20x | 4345 |
| bs_basis_frenzy | 0.699 | 1.30x | 0.624 | 2.56x | 0.770 | 6.32x | 100692 |
| bs_basis_panic | 0.697 | 1.30x | 0.617 | 2.53x | 0.696 | 5.71x | 100692 |
| bd_thin_book_frac | 0.694 | 1.29x | 0.000 | 0.00x | 0.000 | 0.00x | 69424 |
| bs_basis_bull_shock | 0.693 | 1.29x | 0.616 | 2.53x | 0.671 | 5.50x | 100692 |
| bs_basis_bear_shock | 0.693 | 1.29x | 0.613 | 2.51x | 0.658 | 5.39x | 100692 |
| liq_short_panic | 0.686 | 1.28x | 0.609 | 2.50x | 0.605 | 4.96x | 102834 |
| liq_capitulation | 0.686 | 1.28x | 0.616 | 2.53x | 0.615 | 5.05x | 102834 |
| xrel_wh_whale_net_usd_xpct10 | 0.683 | 1.27x | 0.593 | 2.43x | 0.746 | 6.12x | 103477 |
| liq_short_spike | 0.682 | 1.27x | 0.599 | 2.46x | 0.545 | 4.47x | 102834 |
| xrel_liq_long_usd_xpct10 | 0.682 | 1.27x | 0.595 | 2.44x | 0.730 | 5.99x | 103477 |
| liq_long_spike | 0.682 | 1.27x | 0.594 | 2.43x | 0.509 | 4.17x | 102834 |
| bd_n_snapshots | 0.681 | 1.27x | 0.121 | 0.50x | 0.054 | 0.44x | 69692 |
| stbl_total_zscore_30d | 0.645 | 1.20x | 0.553 | 2.27x | 0.769 | 6.31x | 111637 |
| stbl_usdc_zscore_30d | 0.645 | 1.20x | 0.553 | 2.27x | 0.761 | 6.24x | 111637 |
| stbl_stable_shock | 0.645 | 1.20x | 0.553 | 2.27x | 0.770 | 6.32x | 111637 |
| stbl_stable_crash | 0.645 | 1.20x | 0.553 | 2.27x | 0.770 | 6.32x | 111637 |
| stbl_usdt_shock | 0.645 | 1.20x | 0.553 | 2.27x | 0.770 | 6.32x | 111637 |
| stbl_stable_shock_strong | 0.645 | 1.20x | 0.553 | 2.27x | 0.770 | 6.32x | 111637 |
| stbl_compound_shock | 0.645 | 1.20x | 0.553 | 2.27x | 0.770 | 6.32x | 111637 |
| stbl_usdt_delta_7d_pct | 0.645 | 1.20x | 0.553 | 2.27x | 0.754 | 6.18x | 111637 |
| stbl_total_delta_7d_pct | 0.645 | 1.20x | 0.553 | 2.27x | 0.750 | 6.15x | 111637 |
| stbl_dai_zscore_30d | 0.645 | 1.20x | 0.553 | 2.27x | 0.742 | 6.09x | 111637 |
| stbl_total_delta_30d_pct | 0.645 | 1.20x | 0.553 | 2.27x | 0.742 | 6.09x | 111637 |
| stbl_usdt_zscore_30d | 0.645 | 1.20x | 0.553 | 2.27x | 0.735 | 6.03x | 111637 |
| s3_smart_extreme_short | 0.633 | 1.18x | 0.000 | 0.00x | 0.000 | 0.00x | 77117 |
| xrel_rv_bpv_5m_xpct10 | 0.621 | 1.16x | 0.547 | 2.24x | 0.746 | 6.12x | 111637 |

## Top 20 POSITIVE @ K=5 (sharpest cross-sectional signal)

| measure | prec@5 | lift@5 | prec@10 | lift@10 | n_obs |
|---|---:|---:|---:|---:|---:|
| xex_cb_bn_spread_bps | 1.000 | 8.20x | 1.000 | 4.10x | 4345 |
| xex_by_bn_spread_bps | 1.000 | 8.20x | 1.000 | 4.10x | 4345 |
| xex_ok_bn_spread_bps | 1.000 | 8.20x | 1.000 | 4.10x | 4345 |
| stbl_stable_shock_strong | 0.770 | 6.32x | 0.554 | 2.27x | 111637 |
| stbl_stable_crash | 0.770 | 6.32x | 0.554 | 2.27x | 111637 |
| stbl_usdt_shock | 0.762 | 6.25x | 0.554 | 2.27x | 111637 |
| stbl_compound_shock | 0.758 | 6.22x | 0.554 | 2.27x | 111637 |
| stbl_stable_shock | 0.758 | 6.22x | 0.554 | 2.27x | 111637 |
| stbl_usdt_zscore_30d | 0.750 | 6.15x | 0.554 | 2.27x | 111637 |
| stbl_total_delta_30d_pct | 0.742 | 6.09x | 0.554 | 2.27x | 111637 |
| stbl_dai_zscore_30d | 0.742 | 6.09x | 0.554 | 2.27x | 111637 |
| stbl_total_delta_7d_pct | 0.735 | 6.03x | 0.554 | 2.27x | 111637 |
| stbl_usdt_delta_7d_pct | 0.731 | 6.00x | 0.554 | 2.27x | 111637 |
| stbl_usdc_zscore_30d | 0.723 | 5.93x | 0.554 | 2.27x | 111637 |
| stbl_total_zscore_30d | 0.716 | 5.87x | 0.554 | 2.27x | 111637 |
| soc_wiki_views | 0.502 | 4.12x | 1.000 | 4.10x | 8690 |
| stbl_usde_zscore_30d | 0.500 | 4.10x | 0.500 | 2.05x | 60849 |
| bs_basis_frenzy | 0.409 | 3.36x | 0.618 | 2.53x | 100692 |
| bs_basis_panic | 0.274 | 2.25x | 0.537 | 2.20x | 100692 |
| bd_n_snapshots | 0.235 | 1.93x | 0.191 | 0.78x | 69692 |

## Headline

- **4 measures lift >1.5x random** @ K=22 positive ranking
- **0 measures lift >2.0x random** @ K=22 positive ranking
- Best positive predictor: **soc_wiki_views** @ 1.86x lift
- Best negative predictor: **soc_wiki_views** @ 1.86x lift

This is the LISTWISE objective: predict top-K cross-sectional movers using ONE measure as ranker.
Compare to pointwise: pointwise asks `is asset A a mover?`; listwise asks `which K assets will move most today?`.
Lifts >1.5x random mean the catalog should also be mined under listwise objective — it's a structurally different surface.