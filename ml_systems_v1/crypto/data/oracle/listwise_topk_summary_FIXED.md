# Listwise Top-K Mining FIXED (2026-05-23T10:44)

## Data-bug correction note
- Prior mine_listwise_topk.py used `target_return_1` which is CORRUPTED in chimera.
- This version uses `close.pct_change().shift(-1)` directly (actual close-to-close fwd return).

- TRAIN panel: 59143 rows, ~1595 dates, median 38 assets/day
- Random baseline @ K=5 = 0.132
- Random baseline @ K=22 = 0.579

## Top 30 POSITIVE-rank predictors @ K=5 (top-5 by measure -> realized top-5 fwd ret)

| measure | prec@5 | lift@5 | prec@10 | lift@10 | prec@22 | lift@22 | n_obs |
|---|---:|---:|---:|---:|---:|---:|---:|
| stbl_total_zscore_30d | 0.770 | 5.86x | 0.554 | 2.11x | 0.645 | 1.11x | 59086 |
| stbl_compound_shock | 0.770 | 5.86x | 0.554 | 2.11x | 0.645 | 1.11x | 59086 |
| stbl_total_delta_7d_pct | 0.770 | 5.86x | 0.554 | 2.11x | 0.645 | 1.11x | 59086 |
| stbl_total_delta_30d_pct | 0.770 | 5.86x | 0.554 | 2.11x | 0.645 | 1.11x | 59086 |
| stbl_usdt_zscore_30d | 0.770 | 5.86x | 0.554 | 2.11x | 0.645 | 1.11x | 59086 |
| stbl_usdt_delta_7d_pct | 0.770 | 5.86x | 0.554 | 2.11x | 0.645 | 1.11x | 59086 |
| stbl_usdc_zscore_30d | 0.770 | 5.86x | 0.554 | 2.11x | 0.645 | 1.11x | 59086 |
| stbl_dai_zscore_30d | 0.770 | 5.86x | 0.554 | 2.11x | 0.645 | 1.11x | 59086 |
| stbl_stable_shock | 0.770 | 5.86x | 0.554 | 2.11x | 0.645 | 1.11x | 59086 |
| stbl_stable_shock_strong | 0.770 | 5.86x | 0.554 | 2.11x | 0.645 | 1.11x | 59086 |
| stbl_usdt_shock | 0.770 | 5.86x | 0.554 | 2.11x | 0.645 | 1.11x | 59086 |
| stbl_stable_crash | 0.770 | 5.86x | 0.554 | 2.11x | 0.645 | 1.11x | 59086 |
| soc_wiki_views | 0.541 | 4.11x | 1.000 | 3.80x | 1.000 | 1.73x | 1350 |
| bs_basis_frenzy | 0.415 | 3.16x | 0.624 | 2.37x | 0.713 | 1.23x | 51367 |
| bs_basis_panic | 0.238 | 1.81x | 0.578 | 2.20x | 0.717 | 1.24x | 51367 |
| xrel_rv_bpv_5m_xrank | 0.235 | 1.79x | 0.329 | 1.25x | 0.592 | 1.02x | 59086 |
| rv_bpv_5m | 0.235 | 1.79x | 0.329 | 1.25x | 0.592 | 1.02x | 59086 |
| xrel_rv_bpv_5m_xratio | 0.235 | 1.79x | 0.329 | 1.25x | 0.592 | 1.02x | 59086 |
| rv_rv_5m | 0.233 | 1.77x | 0.329 | 1.25x | 0.592 | 1.02x | 59086 |
| xrel_rv_rv_5m_xrank | 0.233 | 1.77x | 0.329 | 1.25x | 0.592 | 1.02x | 59086 |
| xrel_rv_rv_5m_xratio | 0.233 | 1.77x | 0.329 | 1.25x | 0.592 | 1.02x | 59086 |
| xrel_rv_bpv_5m_xpct10 | 0.230 | 1.75x | 0.355 | 1.35x | 0.670 | 1.16x | 59086 |
| xrel_rv_rv_5m_xpct10 | 0.228 | 1.73x | 0.353 | 1.34x | 0.669 | 1.16x | 59086 |
| liq_capitulation | 0.212 | 1.61x | 0.519 | 1.97x | 0.715 | 1.23x | 51387 |
| rv_jv_5m | 0.210 | 1.59x | 0.320 | 1.22x | 0.611 | 1.05x | 59086 |
| liq_long_spike | 0.205 | 1.56x | 0.490 | 1.86x | 0.716 | 1.24x | 51387 |
| liq_short_panic | 0.200 | 1.52x | 0.514 | 1.95x | 0.715 | 1.24x | 51387 |
| liq_short_spike | 0.197 | 1.50x | 0.484 | 1.84x | 0.716 | 1.24x | 51387 |
| liq_short_z30 | 0.189 | 1.44x | 0.351 | 1.33x | 0.685 | 1.18x | 51387 |
| liq_long_z30 | 0.187 | 1.42x | 0.354 | 1.34x | 0.684 | 1.18x | 51387 |

## Top 30 NEGATIVE-rank predictors @ K=5 (bot-5 by measure -> realized top-5 fwd ret)

| measure | prec@5 | lift@5 | prec@10 | lift@10 | n_obs |
|---|---:|---:|---:|---:|---:|
| stbl_usdt_shock | 0.770 | 5.86x | 0.553 | 2.10x | 59086 |
| stbl_usdt_zscore_30d | 0.770 | 5.86x | 0.553 | 2.10x | 59086 |
| stbl_usdc_zscore_30d | 0.770 | 5.86x | 0.553 | 2.10x | 59086 |
| stbl_dai_zscore_30d | 0.770 | 5.86x | 0.553 | 2.10x | 59086 |
| stbl_stable_shock | 0.770 | 5.86x | 0.553 | 2.10x | 59086 |
| stbl_stable_crash | 0.770 | 5.86x | 0.553 | 2.10x | 59086 |
| stbl_stable_shock_strong | 0.770 | 5.86x | 0.553 | 2.10x | 59086 |
| stbl_total_delta_30d_pct | 0.770 | 5.86x | 0.553 | 2.10x | 59086 |
| stbl_compound_shock | 0.770 | 5.86x | 0.553 | 2.10x | 59086 |
| bs_basis_frenzy | 0.770 | 5.86x | 0.621 | 2.36x | 51367 |
| stbl_usdt_delta_7d_pct | 0.770 | 5.86x | 0.553 | 2.10x | 59086 |
| stbl_total_delta_7d_pct | 0.770 | 5.86x | 0.553 | 2.10x | 59086 |
| stbl_total_zscore_30d | 0.770 | 5.86x | 0.553 | 2.10x | 59086 |
| xrel_hbr_eta_total_xpct10 | 0.746 | 5.67x | 0.540 | 2.05x | 59086 |
| xrel_liq_long_usd_xpct10 | 0.746 | 5.67x | 0.593 | 2.26x | 51742 |
| xrel_hbr_n_trades_xpct10 | 0.730 | 5.55x | 0.536 | 2.04x | 59086 |
| xrel_wh_whale_net_usd_xpct10 | 0.730 | 5.55x | 0.599 | 2.28x | 51742 |
| xrel_rv_bpv_5m_xpct10 | 0.714 | 5.43x | 0.547 | 2.08x | 59086 |
| xrel_rv_rv_5m_xpct10 | 0.698 | 5.31x | 0.548 | 2.08x | 59086 |
| bs_basis_bear_shock | 0.684 | 5.20x | 0.614 | 2.33x | 51367 |
| bs_basis_bull_shock | 0.671 | 5.10x | 0.619 | 2.35x | 51367 |
| bs_basis_panic | 0.658 | 5.00x | 0.616 | 2.34x | 51367 |
| liq_capitulation | 0.615 | 4.68x | 0.617 | 2.34x | 51387 |
| liq_short_panic | 0.605 | 4.60x | 0.608 | 2.31x | 51387 |
| liq_short_spike | 0.545 | 4.15x | 0.597 | 2.27x | 51387 |
| liq_long_spike | 0.491 | 3.73x | 0.593 | 2.25x | 51387 |
| soc_wiki_views | 0.459 | 3.49x | 1.000 | 3.80x | 1350 |
| xrel_liq_long_usd_xrank | 0.258 | 1.96x | 0.375 | 1.42x | 51742 |
| liq_long_usd | 0.258 | 1.96x | 0.375 | 1.42x | 51742 |
| xrel_liq_long_usd_xratio | 0.258 | 1.96x | 0.375 | 1.42x | 51740 |

## Headline

- **27 measures lift >1.5x random** @ K=5 positive ranking on REAL returns
- **14 measures lift >2.0x random** @ K=5 positive ranking
- Best positive: **stbl_total_zscore_30d** @ 5.86x lift, prec 0.770
- Best negative: **stbl_usdt_shock** @ 5.86x lift, prec 0.770