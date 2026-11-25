"""Cross-check built views against public market APIs the build never read.

    mlsys crosscheck --data-root data/u10 --universe u10 --start 2020-01-01 --end 2020-12-31

Three channels, each independent of the Vision files the pipeline fetched:

* **Binance REST klines** (``api.binance.com``, the exchange's public market
  data API), every time cadence, bar by bar. OHLC must equal the
  venue's; base, quote and taker-buy volumes must match (1e-8 / 1e-6
  relative, the kline gate's tolerances). Our ``n_trades`` counts aggTrades
  and a kline counts the trades inside them, so ours must be <= the venue's.
  A venue kline outage (traded minutes the venue's klines lack) is found on
  1m and every coarser bar that contains such a minute is reported with it,
  not as a mismatch. Dollar bars have no venue counterpart: each must sit
  inside the REST 1m high/low envelope of the minutes it spans, and their
  notional must add up to the REST quote volume over those minutes.
* **Binance USD-M REST funding history**: every bronze funding event equals
  the venue's, and every gold ``funding_rate`` equals the last venue event at
  or before the bar's close (the funding lag is 0).
* **Another venue** (OKX; KuCoin for a coin OKX did not list): daily closes in
  the same quote currency agree to within a cross-venue spread, and the
  return correlation peaks at lag 0. A bar stamped one period off shows up
  as a peak at +-1. BTC and ETH are also compared on 1h.

Responses are cached under ``<data-root>/crosscheck/`` (atomic parquet with
sidecars), so a rerun does not refetch a finished month. This is not a stage
of ``run``: a year of u10 is several thousand requests. The summary is also
written to ``<data-root>/crosscheck/report.json`` for ``inspect --report``.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from pipeline.calendar import iter_months, month_last
from pipeline.constants import DAY_MS, TIME_WIDTH_MS
from pipeline.fetch_crypto import http_get
from pipeline.io import atomic_write_json, atomic_write_parquet
from pipeline.partitions import load_partitions
from pipeline.validate_run import Report, clip_frame, range_ms

BINANCE_REST = "https://api.binance.com/api/v3/klines"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
OKX_HISTORY = "https://www.okx.com/api/v5/market/history-candles"
KUCOIN_CANDLES = "https://api.kucoin.com/api/v1/market/candles"

REST_INTERVAL = {
    "time_1m": "1m", "time_5m": "5m", "time_30m": "30m", "time_1h": "1h", "time_2h": "2h",
    "time_4h": "4h", "time_6h": "6h", "time_8h": "8h", "time_1d": "1d",
}
REST_LIMIT = 1000
PAUSE_S = 0.05
WORKERS = 6  # months fetched in parallel: a few calls/s x weight 2 stays far under 6000 weight/min

PRICE_REL = 1e-9
VOLUME_REL = 1e-8
QUOTE_REL = 1e-6
SCATTER_FAIL = 1e-4

VENUE_MEDIAN_WARN = 0.005  # median |close / venue close - 1|
VENUE_MEDIAN_FAIL = 0.02
VENUE_DAY_WARN = 0.05  # a single day further apart than this is listed
VENUE_HOURLY_ASSETS = ("BTCUSDT", "ETHUSDT")

KLINE_COLS = ["ts_open_ms", "ts_ms", "open", "high", "low", "close", "volume", "volume_usd",
              "n_trades", "buy_volume", "buy_usd"]


def _utc(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def _ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def _get_json(url: str):
    status, body, err = http_get(url, timeout=60)
    time.sleep(PAUSE_S)
    if status != 200:
        raise RuntimeError(f"{url} -> HTTP {status or err}")
    return json.loads(body)


def _settled(last_day: date) -> bool:
    """A period is cached only once it ended more than a day ago."""
    return last_day < datetime.now(tz=timezone.utc).date() - timedelta(days=1)


class _Live:
    """``--refresh``: every response is fetched from the venue again, even when a
    cached copy exists. The cached copy is compared with the live one and is
    never overwritten, so the run also proves the cache still equals the venue."""

    def __init__(self) -> None:
        import threading
        self.on = False
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.fetched = 0
        self.had_cache = 0
        self.identical = 0
        self.differ: list[str] = []


LIVE = _Live()


def _cached(path: Path, fetch, params: dict, settled: bool) -> pl.DataFrame:
    if LIVE.on:
        df = fetch()
        same = None
        if path.exists():
            same = pl.read_parquet(path).equals(df)
        elif settled:
            atomic_write_parquet(df, path, schema_name="crosscheck_v1", params=params)
        with LIVE.lock:
            LIVE.fetched += 1
            if same is not None:
                LIVE.had_cache += 1
                if same:
                    LIVE.identical += 1
                else:
                    LIVE.differ.append(f"{path.parent.name}/{path.name}")
        return df
    if path.exists():
        return pl.read_parquet(path)
    df = fetch()
    if settled:
        atomic_write_parquet(df, path, schema_name="crosscheck_v1", params=params)
    return df


# ---------------------------------------------------------------- Binance REST klines


def parse_rest_klines(rows: list, width_ms: int) -> pl.DataFrame:
    """Binance kline arrays -> our bar columns, stamped at the close."""
    if not rows:
        return pl.DataFrame(schema={c: (pl.Int64 if c in ("ts_open_ms", "ts_ms", "n_trades") else pl.Float64)
                                    for c in KLINE_COLS})
    return pl.DataFrame({
        "ts_open_ms": [int(r[0]) for r in rows],
        "ts_ms": [int(r[0]) + width_ms for r in rows],
        "open": [float(r[1]) for r in rows],
        "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows],
        "close": [float(r[4]) for r in rows],
        "volume": [float(r[5]) for r in rows],
        "volume_usd": [float(r[7]) for r in rows],
        "n_trades": [int(r[8]) for r in rows],
        "buy_volume": [float(r[9]) for r in rows],
        "buy_usd": [float(r[10]) for r in rows],
    }, schema_overrides={"ts_open_ms": pl.Int64, "ts_ms": pl.Int64, "n_trades": pl.Int64})


def rest_klines_month(root: Path, asset: str, bar_type: str, year: int, month: int) -> pl.DataFrame:
    interval = REST_INTERVAL[bar_type]
    width = TIME_WIDTH_MS[bar_type]
    first = date(year, month, 1)
    last = month_last(year, month)
    path = root / "crosscheck" / "binance_rest" / interval / f"{asset}_{year:04d}-{month:02d}.parquet"

    def fetch() -> pl.DataFrame:
        rows: list = []
        cursor, stop = _ms(first), _ms(last + timedelta(days=1)) - 1
        while cursor <= stop:
            url = (f"{BINANCE_REST}?symbol={asset}&interval={interval}"
                   f"&startTime={cursor}&endTime={stop}&limit={REST_LIMIT}")
            page = _get_json(url)
            rows += page
            if len(page) < REST_LIMIT:
                break
            cursor = int(page[-1][0]) + width
        return parse_rest_klines(rows, width)

    return _cached(path, fetch, {"asset": asset, "interval": interval, "month": f"{year}-{month:02d}"},
                   _settled(last))


def rest_klines(root: Path, asset: str, bar_type: str, start: date, end: date) -> pl.DataFrame:
    from concurrent.futures import ThreadPoolExecutor
    months = list(iter_months(start, end))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        parts = list(pool.map(lambda ym: rest_klines_month(root, asset, bar_type, *ym), months))
    df = pl.concat([p for p in parts if p.height], how="vertical") if any(p.height for p in parts) else parts[0]
    return clip_frame(df, start, end)


def compare_bars(
    ours: pl.DataFrame,
    ref: pl.DataFrame,
    width_ms: int,
    outage_minutes: set[int] | None = None,
) -> dict:
    """Bar-by-bar comparison. ``outage_minutes`` are 1m close stamps the venue's
    klines lack; a coarser bar containing one is reported, not judged."""
    ref_traded = ref.filter(pl.col("volume") > 0)
    j = ours.select(KLINE_COLS).join(ref.select(KLINE_COLS), on="ts_ms", how="full",
                                     coalesce=True, suffix="_r")
    both = j.filter(pl.col("open").is_not_null() & pl.col("open_r").is_not_null())
    only_ours = j.filter(pl.col("open_r").is_null())["ts_ms"].to_list()
    missing = j.filter(pl.col("open").is_null()).join(ref_traded.select("ts_ms"), on="ts_ms")["ts_ms"].to_list()
    empty_ref = ref.height - ref_traded.height

    def off(col: str, rel: float) -> pl.Expr:
        a, b = pl.col(col), pl.col(f"{col}_r")
        return (a - b).abs() > rel * pl.max_horizontal(a.abs(), b.abs(), pl.lit(1e-12))

    bad = both.filter(
        off("open", PRICE_REL) | off("high", PRICE_REL) | off("low", PRICE_REL) | off("close", PRICE_REL)
        | off("volume", VOLUME_REL) | off("buy_volume", VOLUME_REL)
        | off("volume_usd", QUOTE_REL) | off("buy_usd", QUOTE_REL)
    )
    # Bars touched by a venue outage: on 1m the holes and their two edge minutes,
    # on a coarser cadence any bar whose span holds an outage minute.
    if width_ms == 60_000:
        holes = set(only_ours)
        touched = holes | {t - 60_000 for t in holes} | {t + 60_000 for t in holes}
    else:
        touched = {((t - 1) // width_ms + 1) * width_ms for t in (outage_minutes or set())}
    trades_over = both.filter(
        (pl.col("n_trades") > pl.col("n_trades_r")) & ~pl.col("ts_ms").is_in(sorted(touched))
    ).height
    outage_bad = bad.filter(pl.col("ts_ms").is_in(sorted(touched))) if touched else bad.head(0)
    scattered = bad.filter(~pl.col("ts_ms").is_in(sorted(touched))) if touched else bad
    unexplained_only = [t for t in only_ours if t not in touched]
    return {
        "compared": both.height,
        "venue_empty_candles": empty_ref,
        "only_ours": len(only_ours),
        "only_ours_unexplained": len(unexplained_only),
        "missing": len(missing),
        "missing_sample": [str(_utc(t)) for t in missing[:3]],
        "outage_bars": outage_bad.height,
        "scattered": scattered.height,
        "scattered_sample": scattered.select(
            ["ts_ms", "close", "close_r", "volume", "volume_r", "buy_volume", "buy_volume_r"]
        ).head(3).to_dicts(),
        "n_trades_over_venue": trades_over,
        "holes": sorted(touched) if width_ms == 60_000 else [],
        "first_hole": str(_utc(min(only_ours) - width_ms)) if only_ours else None,
    }


def _judge_bars(stats: dict) -> str:
    n = max(stats["compared"], 1)
    bad = stats["scattered"] + stats["only_ours_unexplained"]
    if stats["missing"] or stats["n_trades_over_venue"] or bad / n > SCATTER_FAIL:
        return "FAIL"
    if bad or stats["outage_bars"] or stats["only_ours"]:
        return "WARN"
    return "PASS"


def _bars_detail(stats: dict) -> str:
    d = (f"{stats['compared']} bars equal-checked; {stats['scattered']} differ outside outages, "
         f"{stats['missing']} venue-traded bars missing from ours, "
         f"{stats['n_trades_over_venue']} with more aggTrades than venue trades")
    if stats["only_ours"]:
        d += (f"; venue kline outage: {stats['only_ours']} of our bars absent from the venue "
              f"(first {stats['first_hole']} UTC) and {stats['outage_bars']} bars around it differ")
    elif stats["outage_bars"]:
        d += f"; {stats['outage_bars']} bars contain a venue-outage minute and differ (expected)"
    if stats["scattered"]:
        d += f"; e.g. {stats['scattered_sample'][:2]}"
    if stats["missing"]:
        d += f"; missing e.g. {stats['missing_sample']}"
    return d


# ---------------------------------------------------------------- dollar bars


def check_dollar_against_minutes(dollar: pl.DataFrame, minutes: pl.DataFrame, outage: set[int]) -> dict:
    """Each dollar bar inside the venue 1m high/low envelope of the minutes it spans;
    notional equal to the venue quote volume over those minutes."""
    d = dollar.sort("ts_ms")
    m = minutes.filter(pl.col("volume") > 0).sort("ts_open_ms")
    if d.is_empty() or m.is_empty():
        return {"bars": d.height, "outside": 0, "outage_bars": 0, "notional_ours": 0.0,
                "notional_venue": 0.0, "slack": 0.0}
    base = int(m["ts_open_ms"][0]) // 60_000
    span = int(m["ts_open_ms"][-1]) // 60_000 - base + 1
    hi = np.full(span + 1, -np.inf)
    lo = np.full(span + 1, np.inf)
    quote = np.zeros(span + 1)
    idx = m["ts_open_ms"].to_numpy() // 60_000 - base
    hi[idx] = m["high"].to_numpy()
    lo[idx] = m["low"].to_numpy()
    quote[idx] = m["volume_usd"].to_numpy()
    # A venue outage minute has no candle: our 1m close stamps t -> minute t - 60s.
    outage_idx = np.array(sorted((t - 60_000) // 60_000 - base for t in outage), dtype=np.int64)
    s = np.clip(d["ts_open_ms"].to_numpy() // 60_000 - base, 0, span - 1)
    e = np.clip(d["ts_ms"].to_numpy() // 60_000 - base, 0, span - 1)
    bounds = np.empty(2 * len(s), dtype=np.int64)
    bounds[0::2], bounds[1::2] = s, e + 1
    env_hi = np.maximum.reduceat(hi, bounds)[0::2]
    env_lo = np.minimum.reduceat(lo, bounds)[0::2]
    tol = PRICE_REL * np.abs(env_hi)
    outside = (d["high"].to_numpy() > env_hi + tol) | (d["low"].to_numpy() < env_lo - tol)
    near_outage = np.zeros(len(s), dtype=bool)
    if outage_idx.size:
        k = np.searchsorted(outage_idx, s)
        k = np.clip(k, 0, outage_idx.size - 1)
        near_outage = (outage_idx[k] >= s) & (outage_idx[k] <= e)
    # Notional: the first bar opens on the year's first trade, so its minute is
    # whole; the last bar closes mid-minute, so the rest of that minute is slack.
    first_i, last_i = int(s[0]), int(e[-1])
    venue = float(quote[first_i:last_i + 1].sum())
    return {
        "bars": d.height,
        "outside": int((outside & ~near_outage).sum()),
        "outage_bars": int((outside & near_outage).sum()),
        "outside_sample": [str(_utc(int(t))) for t in d.filter(pl.Series(outside & ~near_outage))["ts_ms"].head(3)],
        "notional_ours": float(d["volume_usd"].sum()),
        "notional_venue": venue,
        "slack": float(quote[last_i]),
    }


# ---------------------------------------------------------------- funding


def rest_funding(root: Path, asset: str, start: date, end: date) -> pl.DataFrame:
    path = root / "crosscheck" / "binance_fapi" / "funding" / f"{asset}_{start}_{end}.parquet"

    def fetch() -> pl.DataFrame:
        rows: list = []
        cursor, stop = _ms(start), _ms(end + timedelta(days=1)) - 1
        while cursor <= stop:
            page = _get_json(f"{BINANCE_FUNDING}?symbol={asset}&startTime={cursor}&endTime={stop}&limit=1000")
            rows += page
            if len(page) < 1000:
                break
            cursor = int(page[-1]["fundingTime"]) + 1
        return pl.DataFrame({
            "ts_ms": [int(r["fundingTime"]) for r in rows],
            "funding_rate": [float(r["fundingRate"]) for r in rows],
        }, schema={"ts_ms": pl.Int64, "funding_rate": pl.Float64})

    return _cached(path, fetch, {"asset": asset, "start": str(start), "end": str(end)}, _settled(end))


def _bronze_funding(root: Path, asset: str, start: date, end: date) -> pl.DataFrame:
    files = sorted((root / "bronze" / "funding" / "um" / asset).glob("*.parquet"))
    if not files:
        return pl.DataFrame(schema={"ts_ms": pl.Int64, "funding_rate": pl.Float64})
    df = pl.concat([pl.read_parquet(f, columns=["ts_ms", "funding_rate"]) for f in files])
    lo, hi = _ms(start), _ms(end + timedelta(days=1))
    return df.filter((pl.col("ts_ms") >= lo) & (pl.col("ts_ms") < hi)).unique("ts_ms").sort("ts_ms")


def compare_funding(bronze: pl.DataFrame, venue: pl.DataFrame) -> dict:
    j = bronze.join(venue, on="ts_ms", how="full", coalesce=True, suffix="_v")
    both = j.filter(pl.col("funding_rate").is_not_null() & pl.col("funding_rate_v").is_not_null())
    differ = both.filter((pl.col("funding_rate") - pl.col("funding_rate_v")).abs() > 1e-12)
    return {
        "events": both.height,
        "differ": differ.height,
        "only_bronze": j.filter(pl.col("funding_rate_v").is_null()).height,
        "only_venue": j.filter(pl.col("funding_rate").is_null()).height,
        "sample": differ.head(3).to_dicts(),
    }


def compare_gold_funding(gold: pl.DataFrame, venue: pl.DataFrame) -> dict:
    """Gold ``funding_rate`` must be the last venue event at or before the bar's close."""
    if "funding_rate" not in gold.columns or venue.is_empty():
        return {"bars": 0, "differ": 0}
    g = gold.select(["ts_ms", "funding_rate"]).sort("ts_ms")
    exp = g.join_asof(venue.sort("ts_ms").rename({"funding_rate": "expected"}), on="ts_ms", strategy="backward")
    differ = exp.filter(
        (pl.col("funding_rate").is_null() != pl.col("expected").is_null())
        | ((pl.col("funding_rate") - pl.col("expected")).abs() > 1e-12)
    )
    return {"bars": g.height, "differ": differ.height, "sample": differ.head(3).to_dicts()}


# ---------------------------------------------------------------- another venue


def _okx_candles(inst: str, bar: str, start: date, end: date, width_ms: int) -> pl.DataFrame:
    rows: list = []
    lo, after = _ms(start), _ms(end + timedelta(days=1))
    while True:
        page = _get_json(f"{OKX_HISTORY}?instId={inst}&bar={bar}&after={after}&limit=100").get("data", [])
        if not page:
            break
        rows += [r for r in page if int(r[0]) >= lo]
        oldest = min(int(r[0]) for r in page)
        if oldest <= lo:
            break
        after = oldest
    return pl.DataFrame({
        "ts_ms": [int(r[0]) + width_ms for r in rows],
        "close_v": [float(r[4]) for r in rows],
    }, schema={"ts_ms": pl.Int64, "close_v": pl.Float64}).unique("ts_ms").sort("ts_ms")


def _kucoin_candles(sym: str, kind: str, start: date, end: date, width_ms: int) -> pl.DataFrame:
    rows: list = []
    lo, hi = _ms(start) // 1000, _ms(end + timedelta(days=1)) // 1000
    cursor_end = hi
    while cursor_end > lo:
        page = _get_json(f"{KUCOIN_CANDLES}?type={kind}&symbol={sym}&startAt={lo}&endAt={cursor_end}").get("data", [])
        if not page:
            break
        rows += page
        oldest = min(int(r[0]) for r in page)
        if oldest <= lo or len(page) < 1500:
            break
        cursor_end = oldest
    return pl.DataFrame({  # KuCoin rows: time, open, close, high, low, volume, turnover
        "ts_ms": [int(r[0]) * 1000 + width_ms for r in rows],
        "close_v": [float(r[2]) for r in rows],
    }, schema={"ts_ms": pl.Int64, "close_v": pl.Float64}).unique("ts_ms").sort("ts_ms")


def venue_closes(root: Path, asset: str, bar_type: str, start: date, end: date) -> tuple[str, pl.DataFrame]:
    base = asset.removesuffix("USDT")
    width = TIME_WIDTH_MS[bar_type]
    okx_bar = {"time_1d": "1Dutc", "time_1h": "1H"}[bar_type]
    kc_kind = {"time_1d": "1day", "time_1h": "1hour"}[bar_type]
    for venue, fetch in (
        ("okx", lambda: _okx_candles(f"{base}-USDT", okx_bar, start, end, width)),
        ("kucoin", lambda: _kucoin_candles(f"{base}-USDT", kc_kind, start, end, width)),
    ):
        path = root / "crosscheck" / venue / bar_type / f"{asset}_{start}_{end}.parquet"
        try:
            df = _cached(path, fetch, {"asset": asset, "venue": venue, "bar": bar_type}, _settled(end))
        except (RuntimeError, ValueError, KeyError):  # unknown instrument or an error body: try the next venue
            continue
        if df.height:
            return venue, df
    return "none", pl.DataFrame(schema={"ts_ms": pl.Int64, "close_v": pl.Float64})


def compare_venue(ours: pl.DataFrame, venue: pl.DataFrame) -> dict:
    j = ours.select(["ts_ms", "close"]).join(venue, on="ts_ms", how="inner").sort("ts_ms")
    if j.height < 10:
        return {"n": j.height}
    dev = (j["close"] / j["close_v"] - 1.0).abs().to_numpy()
    r = np.diff(np.log(j["close"].to_numpy()))
    rv = np.diff(np.log(j["close_v"].to_numpy()))

    def corr(k: int) -> float:
        a, b = (r[: len(r) - k], rv[k:]) if k >= 0 else (r[-k:], rv[: len(rv) + k])
        return float(np.corrcoef(a, b)[0, 1])

    lags = {k: corr(k) for k in (-1, 0, 1)}
    far = j.filter(pl.Series(dev > VENUE_DAY_WARN))
    worst = int(np.argmax(dev))
    return {
        "n": j.height,
        "median_dev": float(np.median(dev)),
        "p99_dev": float(np.quantile(dev, 0.99)),
        "max_dev": float(dev[worst]),
        "max_dev_at": str(_utc(int(j["ts_ms"][worst]))),
        "far_days": far.height,
        "corr_lag": lags,
    }


def _judge_venue(s: dict) -> str:
    if s.get("n", 0) < 10:
        return "WARN"
    peak = max(s["corr_lag"], key=s["corr_lag"].get)
    if s["median_dev"] > VENUE_MEDIAN_FAIL or peak != 0:
        return "FAIL"
    if s["median_dev"] > VENUE_MEDIAN_WARN or s["far_days"]:
        return "WARN"
    return "PASS"


# ---------------------------------------------------------------- driver


# ---------------------------------------------------------------- FX: ECB reference rates

# The ECB euro foreign exchange reference rates: one official rate a business day per
# currency, units per EUR, set at 2.15 pm C.E.T. (Frankfurt local time, CET or CEST).
ECB_EXR = "https://data-api.ecb.europa.eu/service/data/EXR/D.{ccys}.EUR.SP00.A?startPeriod={start}&endPeriod={end}&format=csvdata"
ECB_CURRENCIES = ("USD", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD")
ECB_FIX_HM = (14, 15)
ECB_OFFSETS_MIN = (-60, -30, -15, 0, 15, 30, 60)
# Rounding of the published rates (4-5 digits) and of a cross is about one pip; the
# concertation is a few minutes wide. Median |gap| at the fixing: PASS <= 5 pips,
# FAIL above 10 (a 0.1% price-scale error is about 11 pips).
ECB_MEDIAN_WARN_PIPS = 5.0
ECB_MEDIAN_FAIL_PIPS = 10.0
# Days per daylight-saving regime needed to judge that regime's lag test on its own.
ECB_MIN_REGIME_DAYS = 3
# Our mid at the fixing must come from a bar that closed at most this long before it.
ECB_MAX_STALE_MS = 5 * 60_000


def parse_ecb_exr_csv(text: str) -> pl.DataFrame:
    """ECB SDMX csv -> (ccy, day, per_eur)."""
    import io
    schema = {"ccy": pl.Utf8, "day": pl.Date, "per_eur": pl.Float64}
    if not text.strip():
        return pl.DataFrame(schema=schema)
    raw = pl.read_csv(io.StringIO(text), infer_schema_length=0)
    if not {"CURRENCY", "TIME_PERIOD", "OBS_VALUE"} <= set(raw.columns):
        return pl.DataFrame(schema=schema)
    return raw.select([
        pl.col("CURRENCY").alias("ccy"),
        pl.col("TIME_PERIOD").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("day"),
        pl.col("OBS_VALUE").cast(pl.Float64, strict=False).alias("per_eur"),
    ]).drop_nulls().sort(["ccy", "day"])


def ecb_reference(root: Path, start: date, end: date) -> pl.DataFrame:
    """ECB reference rates for the majors' currencies over ``[start, end]`` (cached when settled)."""
    path = root / "crosscheck" / "ecb" / f"EXR_{start}_{end}.parquet"

    def fetch() -> pl.DataFrame:
        url = ECB_EXR.format(ccys="+".join(ECB_CURRENCIES), start=start.isoformat(), end=end.isoformat())
        status, body, err = http_get(url, timeout=60)
        time.sleep(PAUSE_S)
        if status != 200:
            raise RuntimeError(f"{url} -> HTTP {status or err}")
        # A range with no ECB business day (a weekend) is answered 200 with no body.
        return parse_ecb_exr_csv((body or b"").decode("utf-8", errors="replace"))

    return _cached(path, fetch, {"source": "ecb EXR", "start": str(start), "end": str(end)}, _settled(end))


def ecb_pair_rate(ref: pl.DataFrame, pair: str) -> pl.DataFrame:
    """(day, rate) of ``pair`` in quote units per base unit, from the per-EUR rates."""
    base, quote = pair[:3], pair[3:]

    def leg(ccy: str) -> pl.DataFrame | None:
        if ccy == "EUR":
            return None
        part = ref.filter(pl.col("ccy") == ccy).select(["day", pl.col("per_eur").alias(ccy)])
        return part if part.height else pl.DataFrame(schema={"day": pl.Date, ccy: pl.Float64})

    b, q = leg(base), leg(quote)
    if q is None:  # XXXEUR: EUR per base = 1 / (base per EUR)
        return b.select(["day", (1.0 / pl.col(base)).alias("rate")])
    if b is None:  # EURXXX
        return q.select(["day", pl.col(quote).alias("rate")])
    return b.join(q, on="day", how="inner").select(["day", (pl.col(quote) / pl.col(base)).alias("rate")])


def fixing_ms(day: date) -> int:
    from zoneinfo import ZoneInfo
    local = datetime(day.year, day.month, day.day, *ECB_FIX_HM, tzinfo=ZoneInfo("Europe/Berlin"))
    return int(local.timestamp() * 1000)


def fixing_regime(day: date) -> str:
    """CET (UTC+1) or CEST (UTC+2) at the fixing: a clock error often hits one only."""
    from zoneinfo import ZoneInfo
    local = datetime(day.year, day.month, day.day, *ECB_FIX_HM, tzinfo=ZoneInfo("Europe/Berlin"))
    return "CEST" if local.utcoffset() == timedelta(hours=2) else "CET"


def _offset_medians(gaps_by_offset: dict[int, list[float]]) -> dict[int, float]:
    return {o: float(np.median(np.abs(np.asarray(g)))) for o, g in gaps_by_offset.items() if g}


def _shifted(medians: dict[int, float]) -> int | None:
    """The offset (minutes) that matches clearly better than the fixing itself, if any."""
    if 0 not in medians:
        return None
    best = min(medians, key=medians.get)
    if abs(best) > 15 and medians[best] < 0.5 * medians[0]:
        return best
    return None


def compare_reference(minutes: pl.DataFrame, ref: pl.DataFrame, pip: float) -> dict:
    """Our 1m mid at each fixing instant (the last bar closing at or before it, at most
    5 minutes stale) against the reference rate, in pips, at the fixing and at offsets.

    The offsets are the lag test: a timestamp or daylight-saving error shows up as a
    smaller gap at a non-zero offset than at the fixing itself.
    """
    m = minutes.sort("ts_ms")
    ts = m["ts_ms"].to_numpy()
    close = m["close"].to_numpy().astype(float)
    days = ref["day"].to_list()
    rates = ref["rate"].to_numpy().astype(float)
    by_offset: dict[int, list[float]] = {o: [] for o in ECB_OFFSETS_MIN}
    by_regime: dict[str, dict[int, list[float]]] = {r: {o: [] for o in ECB_OFFSETS_MIN} for r in ("CET", "CEST")}
    worst = (0.0, None)
    for day, rate in zip(days, rates):
        if not np.isfinite(rate) or rate <= 0:
            continue
        regime = fixing_regime(day)
        for off in ECB_OFFSETS_MIN:
            t = fixing_ms(day) + off * 60_000
            i = int(np.searchsorted(ts, t, side="right")) - 1
            if i < 0 or t - int(ts[i]) > ECB_MAX_STALE_MS:
                continue
            gap = (close[i] - rate) / pip
            by_offset[off].append(gap)
            by_regime[regime][off].append(gap)
            if off == 0 and abs(gap) > abs(worst[0]):
                worst = (gap, str(day))
    at0 = np.abs(np.asarray(by_offset[0]))
    medians = _offset_medians(by_offset)
    regimes = {}
    for regime, gaps in by_regime.items():
        rm = _offset_medians(gaps)
        if gaps[0]:
            regimes[regime] = {"days": len(gaps[0]), "median_pips": rm.get(0),
                               "median_by_offset_min": {o: round(v, 2) for o, v in rm.items()},
                               "shifted_to_min": _shifted(rm) if len(gaps[0]) >= ECB_MIN_REGIME_DAYS else None}
    return {"days": int(at0.size), "median_pips": float(np.median(at0)) if at0.size else None,
            "p90_pips": float(np.quantile(at0, 0.9)) if at0.size else None,
            "worst_pips": worst[0], "worst_day": worst[1],
            "median_by_offset_min": {o: round(v, 2) for o, v in medians.items()},
            "best_offset_min": min(medians, key=medians.get) if medians else None,
            "shifted_to_min": _shifted(medians),
            "regimes": regimes,
            "bias_pips": float(np.median(np.asarray(by_offset[0]))) if by_offset[0] else None}


def _judge_reference(s: dict) -> str:
    """Level on all days; the lag test on all days and on each daylight-saving regime
    alone, since a clock error confined to CET or to CEST days hides in the pooled median."""
    if not s["days"]:
        return "WARN"
    med = s["median_pips"]
    shifted = s.get("shifted_to_min") is not None or any(
        r.get("shifted_to_min") is not None for r in (s.get("regimes") or {}).values())
    if med > ECB_MEDIAN_FAIL_PIPS or shifted:
        return "FAIL"
    if med > ECB_MEDIAN_WARN_PIPS:
        return "WARN"
    return "PASS"


def fx_crosscheck(root: Path, pairs: list[str], start: date, end: date, report: Report, summary: dict) -> None:
    """Every FX pair's 1m mid at the ECB fixing against the ECB reference rate."""
    from pipeline.config import fx_pip

    try:
        ref = ecb_reference(root, start, end)
    except Exception as exc:  # noqa: BLE001  the reference is down: report it, keep the run alive
        report.add("ecb_reference", "WARN", f"ECB reference rates unavailable: {exc}")
        return
    for pair in pairs:
        print(f"[crosscheck] {pair}", flush=True)
        ours = load_partitions(root / "gold" / "time_1m", pair, columns=["ts_ms", "close"])
        ours = clip_frame(ours, start, end) if ours is not None else None
        if ours is None or ours.is_empty():
            report.add("ecb_reference", "WARN", f"{pair}: no 1m gold to compare")
            continue
        s = compare_reference(ours, ecb_pair_rate(ref, pair), fx_pip(pair))
        sev = _judge_reference(s)
        detail = (f"{pair}: {s['days']} ECB fixing days (14:15 Frankfurt), median |mid - ECB| "
                  f"{(s['median_pips'] or 0):.2f} pips, p90 {(s['p90_pips'] or 0):.2f}, median bias "
                  f"{(s['bias_pips'] or 0):+.2f}, worst {s['worst_pips']:+.2f} on {s['worst_day']}; median by "
                  f"offset (min) {s['median_by_offset_min']}; by regime "
                  + ", ".join(f"{k} {v['days']}d {(v['median_pips'] or 0):.2f}p"
                              + (f" SHIFTED {v['shifted_to_min']:+d}m" if v.get("shifted_to_min") is not None else "")
                              for k, v in s["regimes"].items()))
        report.add("ecb_reference", sev, detail)
        summary[pair]["ecb_reference"] = s | {"severity": sev}
        print(f"  ecb       {sev}  {detail}", flush=True)


GOLD_COLS = [*KLINE_COLS, "funding_rate"]


def _gold(root: Path, bar_type: str, asset: str, start: date, end: date) -> pl.DataFrame | None:
    df = load_partitions(root / "gold" / bar_type, asset, columns=GOLD_COLS)
    if df is None or df.is_empty():
        return None
    return clip_frame(df, start, end).sort("ts_ms")


def crosscheck(
    root: Path,
    assets: list[str],
    start: date,
    end: date,
    bar_types: list[str] | None = None,
    venue: bool = True,
    funding: bool = True,
    refresh: bool = False,
    market: str = "crypto",
) -> Report:
    """See the module docstring. ``refresh`` fetches every response live (``LIVE``).
    ``market="fx"`` checks FX pairs against the ECB reference rates instead."""
    LIVE.on = bool(refresh)
    LIVE.reset()
    try:
        return _crosscheck(root, assets, start, end, bar_types, venue, funding, market)
    finally:
        LIVE.on = False


def _crosscheck(
    root: Path,
    assets: list[str],
    start: date,
    end: date,
    bar_types: list[str] | None,
    venue: bool,
    funding: bool,
    market: str = "crypto",
) -> Report:
    report = Report()
    summary: dict = defaultdict(dict)
    started = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    wanted = list(bar_types or [*REST_INTERVAL, "dollar"])
    time_types = [bt for bt in REST_INTERVAL if bt in wanted or (bt == "time_1m" and "dollar" in wanted)]
    if market == "fx":
        fx_crosscheck(root, assets, start, end, report, summary)
    for asset in (assets if market != "fx" else []):
        print(f"[crosscheck] {asset}", flush=True)
        outage: set[int] = set()
        minutes: pl.DataFrame | None = None
        frames: dict[str, pl.DataFrame] = {}
        for bt in time_types:
            ours = _gold(root, bt, asset, start, end)
            if ours is None:
                report.add(f"rest_{bt}", "WARN", f"{asset}: no gold to compare")
                continue
            lo_day = _utc(int(ours["ts_open_ms"].min())).date()
            ref = rest_klines(root, asset, bt, max(start, lo_day), end)
            frames[bt] = ours
            stats = compare_bars(ours, ref, TIME_WIDTH_MS[bt], outage)
            if bt == "time_1m":
                outage = set(stats["holes"])
                minutes = ref
            sev = _judge_bars(stats)
            if bt in wanted:
                report.add(f"rest_{bt}", sev, f"{asset}: {_bars_detail(stats)}")
            summary[asset][bt] = {k: v for k, v in stats.items() if k != "holes"} | {"severity": sev}
            print(f"  {bt:<9} {sev}  compared={stats['compared']} scattered={stats['scattered']} "
                  f"outage={stats['outage_bars']} only_ours={stats['only_ours']} missing={stats['missing']}",
                  flush=True)
        if "dollar" in wanted and minutes is not None:
            dollar = _gold(root, "dollar", asset, start, end)
            if dollar is not None:
                s = check_dollar_against_minutes(dollar, minutes, outage)
                gap = abs(s["notional_ours"] - s["notional_venue"])
                ok_notional = gap <= s["slack"] + QUOTE_REL * s["notional_venue"]
                # Outage minutes are missing (holes) or partial (edges) at the venue; ours has them whole.
                if not ok_notional and outage and "time_1m" in frames:
                    o = frames["time_1m"].filter(pl.col("ts_ms").is_in(sorted(outage))).select(["ts_ms", "volume_usd"])
                    v = minutes.filter(pl.col("ts_ms").is_in(sorted(outage))).select(["ts_ms", "volume_usd"])
                    extra = float(o["volume_usd"].sum()) - float(v["volume_usd"].sum())
                    ok_notional = abs(gap - extra) <= s["slack"] + QUOTE_REL * s["notional_venue"]
                    s["outage_notional"] = extra
                sev = "FAIL" if s["outside"] or not ok_notional else ("WARN" if s["outage_bars"] else "PASS")
                report.add("rest_dollar", sev,
                           f"{asset}: {s['bars']} dollar bars; {s['outside']} outside the venue 1m envelope"
                           f" ({s['outage_bars']} more beside an outage); notional {s['notional_ours']:.6g} vs venue"
                           f" {s['notional_venue']:.6g} (diff {gap:.4g}, slack {s['slack']:.4g}"
                           + (f", outage minutes {s['outage_notional']:.4g}" if "outage_notional" in s else "") + ")")
                summary[asset]["dollar"] = s | {"severity": sev}
                print(f"  dollar    {sev}  bars={s['bars']} outside={s['outside']} gap={gap:.4g}", flush=True)
        if funding:
            vf = rest_funding(root, asset, start, end)
            if vf.is_empty():
                report.add("rest_funding", "WARN", f"{asset}: venue has no funding history in range")
            else:
                f = compare_funding(_bronze_funding(root, asset, start, end), vf)
                g = {bt: compare_gold_funding(df, vf) for bt, df in frames.items()}
                g_bad = {bt: v["differ"] for bt, v in g.items() if v["differ"]}
                sev = "FAIL" if f["differ"] or f["only_venue"] or f["only_bronze"] or g_bad else "PASS"
                report.add("rest_funding", sev,
                           f"{asset}: {f['events']} funding events equal-checked, {f['differ']} differ, "
                           f"{f['only_venue']} only at venue, {f['only_bronze']} only in bronze; gold funding_rate "
                           f"as-of on {sum(v['bars'] for v in g.values())} bars, differing {g_bad or 0}"
                           + (f"; e.g. {f['sample'][:2]}" if f["differ"] else ""))
                summary[asset]["funding"] = f | {"gold": g, "severity": sev}
                print(f"  funding   {sev}  events={f['events']} differ={f['differ']} gold_bad={g_bad}", flush=True)
        if venue:
            for bt in ("time_1d", "time_1h"):
                if bt == "time_1h" and asset not in VENUE_HOURLY_ASSETS:
                    continue
                ours = frames.get(bt)
                if ours is None:
                    continue
                lo_day = _utc(int(ours["ts_open_ms"].min())).date()
                name, ref = venue_closes(root, asset, bt, max(start, lo_day), end)
                s = compare_venue(ours, ref)
                sev = _judge_venue(s)
                if s.get("n", 0) < 10:
                    detail = f"{asset}: only {s.get('n', 0)} overlapping closes on {name}"
                else:
                    lag = {k: round(v, 3) for k, v in s["corr_lag"].items()}
                    detail = (f"{asset} vs {name}: {s['n']} closes, median |dev| {s['median_dev']:.3%}, "
                              f"p99 {s['p99_dev']:.2%}, max {s['max_dev']:.2%} at {s['max_dev_at']}; "
                              f"{s['far_days']} beyond {VENUE_DAY_WARN:.0%}; return corr by lag {lag}")
                report.add(f"venue_{bt}", sev, detail)
                summary[asset][f"venue_{bt}"] = s | {"venue": name, "severity": sev}
                print(f"  venue {bt:<8} {sev}  {detail}", flush=True)
    live = None
    if LIVE.on:
        live = {"started": started, "fetched": LIVE.fetched, "had_cache": LIVE.had_cache,
                "identical": LIVE.identical, "differ": LIVE.differ[:20]}
        sev = "FAIL" if LIVE.differ else "PASS"
        report.add("live_refetch", sev,
                   f"{LIVE.fetched} responses fetched live from the venues starting {started}; "
                   f"{LIVE.had_cache} had a copy cached by an earlier run and {LIVE.identical} of them are "
                   f"identical" + (f"; differ: {LIVE.differ[:5]}" if LIVE.differ else ""))
    out = root / "crosscheck" / "report.json"
    atomic_write_json(out, {
        "start": str(start), "end": str(end), "assets": assets,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "live": live,
        "checks": [{"name": c.name, "severity": c.severity, "detail": c.detail} for c in report.checks],
        "per_asset": summary,
    })
    return report
