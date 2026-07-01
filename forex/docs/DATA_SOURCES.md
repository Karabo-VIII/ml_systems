# Forex Data Sources — verified research (2026-07-01)

The comparison behind the pipeline's source choices. Every provider claim was verified against live
provider docs on 2026-07-01; items that could not be pinned to a live page are flagged. **Do not
re-research this** — extend it.

## TL;DR — the chosen stack

| Role | Source | Why |
|---|---|---|
| **PRIMARY bulk historical** | **Dukascopy** | Free, no auth, tick-level, deep history (FX from ~2003, some ~1990s), direct hourly file pulls. The FX analog of crypto's Binance Vision. |
| **Secondary bulk / cross-check** | **HistData.com** | Free, no registration, clean M1 back to 2000. M1-only, volume=0, EST-no-DST timestamps. Backfill + validation. |
| **Live / recent API** | **OANDA v20** | Free practice account → token. REST candles (S5..D) + streaming, bid/ask/mid. Forward paper-trade leg. |

Libs: `requests` + stdlib `lzma` (Dukascopy is self-contained in this repo), optionally `histdata`,
optionally `oandapyV20`. **Skip** for a free pipeline: TrueFX (free depth now uncertain), Polygon/Massive
free tier (EOD-only), Alpha Vantage (25 req/day), Finnhub (FX candles now premium), Databento (no spot FX).

---

## 1. Free bulk archives (the "Binance Vision equivalent")

### Dukascopy — PRIMARY ✅
- **Coverage:** 1000+ instruments — FX majors/crosses/exotics, metals (XAU/XAG/XPT/XPD), commodities,
  indices, crypto. FX history to ~2003 (some pairs to ~1990s).
- **Access:** FREE, no account, no key. Direct HTTPS file download. Per-IP throttling (be polite).
- **Granularity:** tick (bid/ask + bid/ask volume estimate). Also aggregated candles.
- **Format:** `.bi5` = LZMA-compressed binary, one file per (instrument, hour).
- **URL:** `https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5`
- **Gotchas (all handled in `ingest/dukascopy.py`):**
  1. **MONTH IS ZERO-INDEXED** (00=Jan … 11=Dec). `.../EURUSD/2026/05/30/…` = **June 30**.
  2. `.bi5` = LZMA; decode to 20-byte big-endian records `(ms_offset:u32, ask:u32, bid:u32,
     ask_vol:f32, bid_vol:f32)` — **ask before bid**.
  3. Integer price ÷ per-instrument `price_divisor` (100000 for 5-dec FX, 1000 for 3-dec JPY + metals).
  4. Volume = liquidity estimate in millions of base ccy, NOT turnover.
  5. Empty/404 hour = closed (weekend/holiday) — expected, not an error.
- **Libs:** `dukascopy-python` (v4.x, active), `dukascopy-node` (CLI), `duka`, TickVault, `findatapy`.
  This repo does NOT depend on them — `ingest/dukascopy.py` implements the fetch from scratch.
- Sources: datafeed.dukascopy.com; pypi.org/project/dukascopy-python; dukascopy-node.app.

### HistData.com — secondary ✅
- **Coverage:** 66+ FX pairs + commodities (WTIUSD, BCOUSD) + indices (SPXUSD, JPXJPY, …). From 2000.
- **Access:** FREE, no registration. Per-pair/month ZIP (also FTP/SFTP mirrors).
- **Granularity:** M1 (1-min OHLC, **bid quotes only**) and tick. **Volume column is ALWAYS 0 for FX.**
- **Format:** ZIP → semicolon CSV. M1: `YYYYMMDD HHMMSS;O;H;L;C;V`. **Timezone = EST WITHOUT DST
  (fixed UTC−5)** — a gotcha; `ingest/histdata.py` adds 5h uniformly to reach UTC.
- **Libs:** `histdata` (philipperemy) or `histdatacom`.
- Sources: histdata.com/f-a-q/data-files-detailed-specification; github.com/philipperemy/FX-1-Minute-Data.

### TrueFX — not used (flagged)
- 15 major pairs, tick, from May 2009, free with registration. **CAVEAT:** third-party reports say free
  history was cut to ~last 1 year after a site redesign; could not confirm behind the login wall. No
  maintained Python lib. Skipped.

---

## 2. Broker APIs (live / recent)

### OANDA v20 — live secondary ✅
- **Access:** FREE practice (demo) account → personal access token (My Account → Manage API Access).
  Header `Authorization: Bearer <token>`. Hosts: practice `api-fxpractice.oanda.com`, live `api-fxtrade.oanda.com`.
- **Candles:** `GET /v3/instruments/{EUR_USD}/candles?granularity=H1&count=5000&price=MBA` (mid/bid/ask).
  Instrument uses an **underscore** (`EUR_USD`). Granularities S5,S10,S15,S30,M1,M2,M4,M5,M10,M15,M30,
  H1,H2,H3,H4,H6,H8,H12,D,W,M. **Max 5000 candles/request** → paginate with from/to.
- **Rate limits:** 120 req/s REST; 20 active streams; 2 new connections/s. Pricing stream throttled to
  4 prices/s/instrument.
- **Lib:** `oandapyV20` (v0.7.2, last release 2021 — functional but stale). This repo's `ingest/oanda.py`
  uses raw `requests` (no dependency).
- Sources: developer.oanda.com/rest-live-v20/*.

### Interactive Brokers — not wired (documented alternative)
- `ib_async` (v2.1.0, Py≥3.10; successor to the now-unmaintained `ib_insync`). **Requires a running
  TWS/IB Gateway** (no pure-cloud REST via this lib). FX needs no market-data subscription but a funded
  (~$500) account + paper account is practically required. Pacing: ≤50 concurrent hist requests,
  ≤60 requests/10min, BID_ASK counts double. Pro-grade fallback; not needed for the free pipeline.

### FXCM / ForexConnect / fxcmpy — DEAD, do not use ❌
- `fxcmpy` on PyPI is an **empty security-holding placeholder** (real package deleted). FXCM (now Stratos
  under Jefferies) is in ownership limbo / pending sale. ForexConnect SDK's `O2G2` namespace deprecated
  since 2015. Avoid entirely.

---

## 3. Aggregator APIs (verified free-tier limits)

| Provider | FX free? | Free limit | History | Format / lib | Verdict |
|---|---|---|---|---|---|
| **Twelve Data** | yes | 800 credits/day (~8/min) | **20+ yr**, 1-min | JSON, `EUR/USD`, official `twelvedata` | Best free API for gap-fill / cross-check |
| **Tiingo** | yes | 50/hr, 1000/day, 500 sym/mo | intraday from **Jan 2020** | JSON, `eurusd`, `tiingo` | OK but tight; no historical tick over REST |
| **Polygon.io / Massive** | yes | 5 calls/min, **EOD only**, 2yr | 2yr free / 10yr paid | JSON, `C:EURUSD`, `polygon-api-client` | Free = EOD only; Flat Files S3 = paid. Rebranded to Massive 2025-10-30. |
| **Alpha Vantage** | yes | **25 req/day** | FX_DAILY full = 20+yr | JSON/CSV, `from_symbol/to_symbol`, `alpha_vantage` | 25/day too tight for bulk |
| **EODHD** | paid for FX | free tier excludes FX (demo key aside) | 20+yr EOD; 1m since 2009 | JSON/CSV, `EURUSD.FOREX`, official `eodhd` | Clean but FX needs $19.99+ tier |
| **FMP** | ambiguous | 250 req/day | EOD free / intraday paid | JSON, `EURUSD`, community `fmpsdk` | FX-on-free unconfirmed; no official lib |
| **Finnhub** | **NO** | 60/min | — | `finnhub-python` | **FX candles now Premium-gated** (was free) |
| **Databento** | **NO spot FX** | $125 one-time credit | CME FX futures (6E) from 2010 | DBN, `databento` | Only CME FX **futures**, not spot |

---

## 4. FX-specific data considerations (what differs from crypto)

- **24/5 market.** Weekly open Sun ~17:00 New York, close Fri ~17:00 NY. **Anchor to `America/New_York`
  17:00, NOT a fixed UTC constant** (DST shifts it: ~21:00 UTC summer / ~22:00 UTC winter).
- **Sessions (approx UTC, ±1h DST):** Tokyo ~00:00–09:00 (Japan = no DST), London ~07:00–16:00,
  New York ~12:00–21:00. Deepest liquidity = **London/NY overlap ~12:00–16:00 UTC**.
- **Weekend + holiday gaps are STRUCTURAL, not missing.** Never alarm on the Fri-close→Sun-open silence;
  measure its magnitude (weekend gap) instead. Thin liquidity late December + national holidays.
- **No native volume.** OTC/decentralized. Use tick-count as the activity proxy; Dukascopy's volume is a
  bid/ask liquidity estimate (millions base ccy); HistData volume = 0. Never mix the semantics.
- **Bid/ask spread is first-class.** Bars carry bid/ask + spread; cost models use the actual session spread.
- **Pip is pair-aware:** 0.0001 most pairs, **0.01 JPY pairs** (the #1 off-by-100 bug); pipette = 1/10 pip.
- **Carry = swap/rollover** (interest-rate differential), daily at 17:00 NY, **TRIPLE on Wednesday** (T+2
  settlement skips the weekend). No perpetual funding.

## 5. Instrument universe tiers (see `config/universes/`)

- **u7** majors: EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD, USDCHF, NZDUSD.
- **u28** + crosses (EUR/GBP/JPY crosses + AUD/CAD/NZD/CHF crosses) + XAUUSD, XAGUSD.
- **u52** + exotics (USD/EUR/GBP vs TRY, ZAR, MXN, SGD, HKD, NOK, SEK, PLN, HUF, CZK, CNH, THB) + XPT/XPD.
- Rough liquidity ranking (BIS-consistent): EUR/USD (~23%) ≫ USD/JPY (~13%) > GBP/USD (~9%) > AUD/USD,
  USD/CAD, USD/CHF > EUR/JPY, EUR/GBP > NZD/USD > other crosses > exotics.
