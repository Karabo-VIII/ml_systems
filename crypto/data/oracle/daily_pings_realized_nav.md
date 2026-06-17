> **SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup)**: This document predates the
> canonical split-discipline gate. References to "OOS" in this file may include data
> from the canonical UNSEEN window (>=2026-01-01) per [src/split_config.py](../../src/split_config.py).
> Use this document for historical context only; deploy decisions citing UNSEEN-relevant
> claims must be re-derived from the canonical segments.

# Daily Pings -- Realized OOS NAV (2026-05-23T01:04)

**Window**: 2024-05-16 -> 2026-05-18 (732 trading days)
**Setup**: top-3 LONG-dominant picks/day @ 25% each (75% deployed). Cost 0.24% RT.

## Headline metrics

| Metric | Value |
|---|---:|
| Mean realized %/d (post-cost) | **-0.1387%/d** |
| Median %/d | -0.2462%/d |
| Std %/d | 3.1700% |
| Sharpe (annualized) | **-0.695** |
| Total compound | **-74.96%** |
| Max DD | -79.59% |
| Positive days | 328 (44.8%) |
| Negative days | 404 |

## Caveats

- This is the daily_top_pings layer realized on OOS chimera closes. Composition is naive top-K LONG-dominant by sum_conviction without dedup.
- 24bp RT cost is maker-leaning; v3-paper-trade-replay with full bucket-aware cost may deflate further.
- 50-engine cap on the underlying daily_pings.parquet -- full 213-engine version would yield denser picks per day.
- v3-paper-trade-replay is the canonical truth; this is a fast approximation.

## Sample 10 most-recent days

| date | n_picks | assets | day_nav_pct_post_cost |
|---|---:|---|---:|
| 2026-05-18 | 2 | DOGE,JST | +0.113% |
| 2026-05-17 | 3 | FET,JST,SUPER | +0.061% |
| 2026-05-16 | 3 | FET,LINK,SOL | -1.508% |
| 2026-05-15 | 3 | FET,LINK,SOL | -2.774% |
| 2026-05-14 | 3 | FET,LINK,SOL | -3.192% |
| 2026-05-13 | 3 | FET,LINK,SOL | +0.726% |
| 2026-05-12 | 3 | BTC,LINK,SOL | -1.595% |
| 2026-05-11 | 3 | BTC,LINK,SOL | -2.062% |
| 2026-05-10 | 3 | ADA,LINK,SOL | -0.384% |
| 2026-05-09 | 3 | BTC,LINK,SOL | +2.034% |