# forex — project guide (spot FX + metals data pipeline)

A production-grade **forex data pipeline**, sibling to `crypto/` under the `ml_systems` workspace.
Fetches FX/metals data from free + keyed sources, builds bid/ask time bars, and validates them with
FX-aware quality gates. This is the *data foundation* — you can fetch, validate, and get pipeline
metrics today; modelling/strategy layers build on top later.

Status (2026-07-01): **FULL PIPELINE COMPLETE + verified live on real data.** The whole stack —
fetch -> {time bars, alt bars} -> {features, cross-pair panels} -> gold dataset (+ carry) -> quality —
runs end-to-end via a DAG orchestrator. Validated on all 7 majors for June 2026 (7 pairs x 528 1h bars,
all quality gates PASS, realistic spreads 0.47-1.46 pips). Gates green: 13/13 unit tests, invariants,
CDAP. See `docs/DATA_SOURCES.md` (sources) + `docs/FEATURES.md` (every column).

## Run (from `forex/`; use the shared root `.venv`)
```
# the whole DAG in dependency order: fetch -> bars -> {alt_bars, features, panels} -> dataset -> quality
python src/pipeline/refresh.py --all --universe u7 --start 2026-06-01 --end 2026-07-01
python src/pipeline/refresh.py --list                       # show the stage DAG
python src/pipeline/refresh.py --target dataset --universe u7  # a target + its deps

# per stage (each is self-sufficient):
python src/pipeline/fetch_all.py       --universe u7 --start 2026-06-01 --end 2026-07-01  # bronze ticks
python src/pipeline/build_bars.py      --universe u7 --cadences 1m 5m 15m 1h 4h 1d         # silver time bars
python src/pipeline/build_alt_bars.py  --universe u7 --tick-n 500 --range-pips 10          # silver alt bars
python src/pipeline/build_features.py  --universe u7 --cadences 1h 1d                      # per-pair features
python src/pipeline/panels.py          --universe u28 --cadences 1h 1d                     # cross-pair panels
python src/pipeline/ingest/rates.py                                                        # FRED carry (keyless)
python src/pipeline/ingest/econ_calendar.py                                                # ForexFactory events
python src/pipeline/make_dataset.py    --universe u7 --cadences 1h 1d                      # GOLD model-ready
python src/pipeline/data_quality.py    --universe u7 --cadences 1h 1d                      # gate + metrics
python src/pipeline/inspect_data.py    --universe u7                                       # coverage report

# gates (RWYB): python run_tests.py  |  python run_invariants_check.py  |  python scripts/check_contracts.py
```
- Import root is `forex/src/` (each entrypoint bootstraps it; `python -m pipeline.<mod>` from `src/` also works).
- Universes: `--universe u7|u28|u52` OR `--pairs EURUSD USDJPY ...` (EUR/USD, EURUSD, eur_usd all accepted).

## Pipeline layers (bronze -> gold)
- **bronze** `data/raw/<PAIR>/ticks/` — Dukascopy ticks (bid/ask/volume). Also `m1/` (HistData).
- **silver bars** `bars/<cadence>/` (time) + `bars/{tick_count,range}/` (information-driven).
- **silver features** `features/<cadence>/` — per-pair causal features (returns/vol/microstructure/session/regime).
- **silver panels** `panels/<cadence>/` — USD strength, per-currency strength, triangular no-arb residuals.
- **rates/calendar** `rates/` (FRED carry) + `calendar/` (ForexFactory high-impact events).
- **gold** `dataset/<cadence>/` — bars + features + panel + carry + `fwd_ret_*` targets + train/val/oos/unseen split.
- Full column catalog: `docs/FEATURES.md`. Data-quality + coverage: `data_quality.py`, `inspect_data.py`.

## Data sources (see `docs/DATA_SOURCES.md` for the full verified matrix)
| Source | Role | Access | Notes |
|---|---|---|---|
| **Dukascopy** | PRIMARY (default) | free, no auth, tick `.bi5` bulk | the "Binance Vision of FX": deep tick history, bid/ask + volume proxy |
| **HistData.com** | secondary | free, no auth, M1 ZIP | M1 OHLC (bid only), volume=0, EST-no-DST timestamps; cross-check/backfill |
| **OANDA v20** | live/recent | free practice token (`OANDA_API_TOKEN`) | REST candles S5..D, bid/ask/mid; forward paper-trade leg |

`ingest/dukascopy.py` is self-contained (stdlib `lzma` + `requests`); the others' libs are optional.

## Layout (binding)
- Root = command surface + this guide + `README.md` + `requirements.txt` + `pyproject.toml` +
  `run_tests.py` / `run_invariants_check.py` (RWYB gates).
- `src/pipeline/` — stages: `fetch_all` · `build_bars` · `build_alt_bars` · `build_features` · `panels` ·
  `make_dataset` · `data_quality` · `inspect_data` · `run_pipeline` · `refresh` (DAG); framework helpers
  `layout` / `universe_loader` / `cli` / `parquet_io` / `dispatch`; `ingest/{dukascopy,histdata,oanda,rates,econ_calendar}.py`.
- `config/` — `data_config.yaml` (sources, calendar, splits) + `universes_index.yaml` +
  `universes/{u7,u28,u52}.yaml` + `asset_dag.yaml` (stage DAG) + `rates.yaml` (FRED series).
- `data/` (gitignored) — `raw/<PAIR>/{ticks,m1}/` (bronze); `processed/{bars,features,panels,dataset,rates,calendar}/` (silver+gold).
- `runs/{pipeline_metrics,coverage}/` — metrics JSON.  `scripts/check_contracts.py` — CDAP gate.
  `tests/_test_*.py` — unit tests.  `docs/` — DATA_SOURCES + FEATURES + design notes.

## Instrument universe
`u7` = the 7 majors (EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD, USDCHF, NZDUSD). `u28` = + standard
crosses + gold/silver (XAUUSD, XAGUSD). `u52` = + exotics (EM crosses) + platinum/palladium. Each pair
carries `pip`, `price_divisor`, `dna` (MAJOR/CROSS/EXOTIC/METAL), `pos_cap`, `tier`, `turnover_share`.

## FX INVARIANTS (binding — these are what differs from crypto; do NOT port crypto assumptions blindly)
- **24/5 market, NOT 24/7.** Weekly open Sun ~17:00 New York, close Fri ~17:00 NY — anchored to
  `America/New_York` 17:00 (DST-shifting; ~21:00 UTC summer / ~22:00 UTC winter), never a fixed UTC constant.
- **Weekend/holiday gaps are EXPECTED, not missing data.** The Fri-close->Sun-open silence is structural.
  `data_quality.py` classifies gaps weekend-expected vs intra-session-anomalous; never alarm on the former.
- **No native volume.** Spot FX is OTC/decentralized. `tick_count` (ticks/bar) is the volume proxy;
  Dukascopy's `volume_est` is a bid/ask liquidity estimate (millions base ccy), NOT turnover. HistData
  volume is ALWAYS 0. Never feed one feed's "volume" to a feature expecting another's semantics.
- **Bid/ask spread is first-class.** Bars carry `bid_close`/`ask_close`/`spread_mean`/`spread_max`; cost
  models must charge the actual session spread, not a constant. Mid OHLC is derived `(bid+ask)/2`.
- **Pip is pair-aware.** 0.0001 for most pairs, **0.01 for JPY pairs** (the #1 off-by-100 FX bug).
- **Carry = swap/rollover** (interest-rate differential), daily at 17:00 NY, **TRIPLE on Wednesday** (T+2
  settlement skips the weekend). There is NO perpetual funding. (Modelled later; not in bar layer.)

## Dukascopy `.bi5` gotchas (in `ingest/dukascopy.py` — verify if you touch it)
- **URL month is ZERO-INDEXED** (00=Jan ... 11=Dec). `.../EURUSD/2026/05/30/...` = **June** 30.
- `.bi5` = LZMA-compressed; 20-byte big-endian records `(ms_offset:u32, ask:u32, bid:u32, ask_vol:f32,
  bid_vol:f32)` — **ask before bid**. Timestamp = hour_start + ms_offset (UTC).
- Integer price / `price_divisor` (per-pair: 100000 non-JPY, 1000 JPY + metals). A wrong divisor is caught
  by `data_quality`'s price-plausibility band; a flipped bid/ask is caught by the negative-spread gate.
- Empty/404 hours = closed (weekend/holiday), skipped silently — not an error.

## Conventions (binding)
- **NO emoji / non-ASCII in print statements** (Windows cp1252 crashes — same rule as crypto). Note polars'
  default table repr uses box-drawing chars that break the Windows console; print custom ASCII tables.
- **RWYB (run-what-you-build):** verify every change by RUNNING it on real data before declaring done.
- **Atomic writes:** all parquet via `parquet_io.atomic_write_parquet` (tmp + os.replace); never inline.
- **`__contract__`:** every pipeline file declares a top-of-file contract dict (kind/stage/inputs/outputs/
  invariants) — the CDAP convention ported from crypto.
- **Data splits** (frozen, in `data_config.yaml`): train<=2020-12-31, val<=2022-12-31, oos<=2024-06-30,
  unseen from 2024-06-30 (reserved for backtest + forward paper-trade). 400-bar purge gap.

## Known refinements (future, documented not-yet-done)
- 1d bars use UTC-midnight boundaries; the FX-conventional 17:00-NY daily boundary is a future refinement.
- Exotic (u52) `price_divisor`/`pip` are Dukascopy-typical defaults — verify per pair on first fetch.
- HistData/OANDA ingesters are functional but only Dukascopy is smoke-tested live.
