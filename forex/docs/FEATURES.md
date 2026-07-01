# Forex pipeline — feature & column catalog

Every column the pipeline produces, by layer. **Causality:** everything except the
`fwd_*` targets is look-ahead-safe (row `t` uses only data with timestamp `<= t`).
SOTA feature design verified against the quant-FX literature (see provenance at end).

## Silver — time bars (`data/processed/bars/<cadence>/`, cadences 1m..1d)
| Column | Meaning |
|---|---|
| `timestamp` | bar-open time (ms, UTC) |
| `open/high/low/close` | MID price OHLC (`(bid+ask)/2`) |
| `bid_close` / `ask_close` | last bid / ask in the bar (execution + cost) |
| `spread_mean` / `spread_max` | intrabar spread (price units) |
| `tick_count` | ticks in the bar (the FX volume proxy) |
| `volume_est` | Dukascopy bid+ask volume estimate (millions base ccy; proxy) |

## Silver — alt bars (`bars/tick_count/`, `bars/range/`)
Information-driven sampling: `timestamp, open/high/low/close, bid_close, ask_close, tick_count`.
tick_count = new bar every N ticks; range = new bar every R pips.

## Silver — per-pair features (`features/<cadence>/`) — causal
| Group | Columns |
|---|---|
| Returns/momentum | `log_ret_1`, `ret_{1,3,5,10,20,50}`, `close_z_100`, `mom5_z` (vol-normalized) |
| Volatility | `rv_{10,20,50}` (realized), `parkinson_{10,20,50}`, `garman_klass_20`, `atr_pips_{10,20,50}`, `rvol_ratio_10_50` |
| Microstructure | `spread_pips`, `rel_spread_bps`, `spread_z_100`, `tick_intensity_z_100`, `amihud_illiq_20`, `quote_imbalance` |
| Session/time | `hour_utc`, `dow`, `hour_sin/cos`, `is_tokyo/london/newyork`, `is_ldn_ny_overlap`, `weekend_gap_pips` |
| Trend/regime | `sma20_ratio`, `sma50_ratio`, `efficiency_ratio_{20,50}` (Kaufman), `vol_regime_z` |

## Silver — cross-pair panel (`panels/<cadence>/panel_*.parquet`) — cross-sectional, causal
| Column | Meaning |
|---|---|
| `usd_strength_ret` | avg USD-side log return across the majors (sign-correct) |
| `usd_index` | cumulative USD strength (DXY-like) |
| `str_<CCY>` | per-currency strength (null-aware mean of signed returns), CCY in EUR/USD/JPY/GBP/AUD/CAD/CHF/NZD |
| `tri_<CROSS>_resid_pips` | triangular no-arb residual `ln(cross) - [ln(BASE/USD) - ln(QUOTE/USD)]` (feature + QC) |
| `tri_max_abs_resid_pips` | worst triangular residual (data-quality roll-up) |

## Rates / carry (`ingest/rates.py` → `rates/rates_daily_*.parquet`; joined into the dataset)
`rate_<CCY>` (FRED policy rate, forward-filled daily), `rate_base`, `rate_quote`, `carry_diff` = rate(base) − rate(quote).

## Econ calendar (`ingest/econ_calendar.py` → `calendar/events_*.parquet`; `event_flags()` helper)
`timestamp, currency, title, impact, forecast, previous`; helper adds `is_high_impact_event`,
`mins_to_next_high_event`. Current-week only (ForexFactory free feed); FRED releases for historical.

## Gold — model-ready dataset (`dataset/<cadence>/<pair>_dataset_*.parquet`)
Bars + features + panel + carry + universe metadata (`pair, dna, pip, price_divisor`), plus:
| Column | Meaning |
|---|---|
| `fwd_ret_{1,5,20}` | **TARGETS** — forward log returns (the ONLY non-causal columns) |
| `split` | `train` (<=2020-12-31) / `val` (<=2022-12-31) / `oos` (<=2024-06-30) / `unseen` (>2024-06-30) |
| `is_purge` | True on the `purge_bars` rows before a split boundary (no normalization leak) |

## Provenance
Feature design verified against: Macrosynergy (6 realized-vol estimators), Amihud (2002) illiquidity,
Kaufman efficiency ratio, ICE US Dollar Index formula, standard currency-strength construction, and
FX session/rollover conventions. Full research: `docs/DATA_SOURCES.md` + the build-session notes.
Known refinements (documented, not yet done): DST-aware session boundaries (v1 uses fixed UTC windows),
Rogers-Satchell/Yang-Zhang vol, ADX regime, 17:00-NY daily bar boundary.
