# forex — production-grade FX data pipeline

Sibling of `crypto/` in the `ml_systems` workspace. Fetches spot-FX + metals data from free +
keyed sources, builds **bid/ask time bars**, and validates them with **FX-aware quality gates**.
You can fetch, validate, and get pipeline metrics today; feature/dataset/modelling layers build on top.

> **Binding project rules + FX invariants live in [`CLAUDE.md`](CLAUDE.md). Read it before touching the pipeline.**
> The verified data-source research is in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Quickstart

```bash
# install into the shared root .venv
pip install -r requirements.txt

# one command: fetch ticks -> build bars -> validate + emit metrics (7 majors, a date window)
python src/pipeline/run_pipeline.py --universe u7 --start 2026-06-22 --end 2026-07-01
```

Per stage:
```bash
python src/pipeline/fetch_all.py    --universe u7 --start 2026-06-22 --end 2026-07-01   # bronze: Dukascopy ticks
python src/pipeline/build_bars.py   --universe u7 --cadences 1m 5m 15m 1h 4h 1d          # silver: bid/ask time bars
python src/pipeline/data_quality.py --universe u7 --cadences 1m 1h 1d                     # validate + metrics JSON
```

Pick instruments with `--universe u7|u28|u52` or `--pairs EURUSD USDJPY XAUUSD` (EUR/USD, EURUSD,
eur_usd all accepted).

## Data sources

| Source | Role | Access | Python |
|---|---|---|---|
| **Dukascopy** | PRIMARY (default) | free, no auth — tick `.bi5` bulk archive, deep history, bid/ask + volume proxy | self-contained (`ingest/dukascopy.py`) |
| **HistData.com** | secondary | free, no auth — M1 OHLC ZIPs (bid only, volume=0) | `histdata` (optional) |
| **OANDA v20** | live/recent | free practice token (`OANDA_API_TOKEN`) — REST candles S5..D | `requests` (built in) |

Dukascopy is the "Binance Vision of FX". The other two are cross-check / live legs. Full comparison of
all researched sources (Twelve Data, Tiingo, Polygon/Massive, Alpha Vantage, EODHD, FMP, TrueFX,
Finnhub, Databento, IBKR) with verified free-tier limits is in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Instrument universe

- **u7** — the 7 majors: EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD, USDCHF, NZDUSD.
- **u28** — + standard crosses + gold/silver (XAUUSD, XAGUSD).
- **u52** — + exotics (EM crosses) + platinum/palladium.

Declared in `config/universes/*.yaml`; each pair carries pip size, Dukascopy price divisor, DNA
(MAJOR/CROSS/EXOTIC/METAL), position cap, liquidity tier, and BIS turnover share.

## Layout

```
forex/
  src/pipeline/            fetch_all · build_bars · data_quality · run_pipeline
    ingest/                dukascopy (primary) · histdata · oanda
    bars/  features/       alt bar types · feature engineering
    layout · universe_loader · cli · parquet_io · dispatch    (framework helpers)
  config/                  data_config.yaml · universes_index.yaml · universes/{u7,u28,u52}.yaml
  data/  (gitignored)      raw/<PAIR>/{ticks,m1}/   processed/bars/<cadence>/
  runs/pipeline_metrics/   data-quality metrics JSON
  docs/                    DATA_SOURCES.md + design notes
  tests/                   unit + invariants gate
```

## What's built + verified

Fetch (Dukascopy ticks) → bars (1m/5m/15m/30m/1h/4h/1d, mid OHLC + bid/ask + spread + tick_count) →
FX-aware validation (weekend-gap-aware coverage, spread sanity, OHLC integrity, price plausibility,
null rates) + metrics JSON. Smoke-tested live on real EUR/USD ticks (63k ticks/day, all gates PASS,
~0.3-pip spread — confirming bid/ask ordering + price divisor are correct).
