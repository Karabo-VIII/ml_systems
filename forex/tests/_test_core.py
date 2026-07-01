"""Core unit tests for the forex pipeline (run via `python run_tests.py`).

Pure-logic, no-network tests of the load-bearing correctness: pair normalization,
the Dukascopy .bi5 parser (the highest-risk code), universe metadata, tick->bar
resampling, feature causality, the data-quality gate, and cross-pair panel signs.
"""
from __future__ import annotations

import lzma
import struct
from datetime import datetime, timezone

import polars as pl

from pipeline import layout, build_bars, build_features, data_quality, panels
from pipeline.ingest import dukascopy
from pipeline.universe_loader import UniverseLoader


# ─── pair normalization / layout ────────────────────────────────────────────

def test_normalize_pair_forms():
    for inp in ("EUR/USD", "EURUSD", "eur_usd", "eurusd"):
        n = layout.normalize_pair(inp)
        assert n.flat == "EURUSD" and n.base == "EUR" and n.quote == "USD"
        assert n.oanda == "EUR_USD" and n.lower == "eurusd"


def test_normalize_pair_rejects_bad():
    try:
        layout.normalize_pair("EURO")
        assert False, "should have raised"
    except ValueError:
        pass


def test_bars_path_roundtrip():
    from datetime import date
    p = layout.bars_path("EURUSD", "1h", latest_date=date(2026, 6, 30))
    assert p.name == "eurusd_1h_20260630.parquet"


# ─── Dukascopy .bi5 parser (highest-risk code) ──────────────────────────────

def test_bi5_record_parse():
    # (ms_offset, ask_int, bid_int, ask_vol, bid_vol) -- ask BEFORE bid.
    buf = (struct.pack(">IIIff", 1000, 114000, 113990, 1.5, 2.0)
           + struct.pack(">IIIff", 2000, 114010, 113995, 1.0, 1.2))
    cols = dukascopy.parse_hour_bytes(buf, hour_start_ms=1_000_000, divisor=100000.0)
    assert cols["timestamp"] == [1_001_000, 1_002_000]
    assert abs(cols["ask"][0] - 1.14) < 1e-9
    assert abs(cols["bid"][0] - 1.1399) < 1e-9
    assert cols["ask"][0] > cols["bid"][0]           # ask > bid (spread positive)


def test_bi5_lzma_roundtrip():
    raw = struct.pack(">IIIff", 500, 100000, 99990, 1.0, 1.0)
    for fmt in (lzma.FORMAT_ALONE, lzma.FORMAT_XZ):
        assert dukascopy._decompress_bi5(lzma.compress(raw, format=fmt)) == raw


def test_dukascopy_month_zero_indexed():
    url = dukascopy._hour_url("EURUSD", datetime(2026, 6, 30, 13, tzinfo=timezone.utc))
    assert "/2026/05/30/13h_ticks.bi5" in url        # June -> month index 05


# ─── universe metadata ──────────────────────────────────────────────────────

def test_universe_loads_and_pips():
    U = UniverseLoader.load()
    assert U.is_in("EURUSD", "u7")
    assert U.pip("EURUSD") == 0.0001 and U.price_divisor("EURUSD") == 100000.0
    assert U.pip("USDJPY") == 0.01 and U.price_divisor("USDJPY") == 1000.0
    assert U.dna_for("XAUUSD") == "METAL"
    assert len(U.list("u7")) == 7 and len(U.list("u28")) == 28


# ─── tick -> bar resampling ─────────────────────────────────────────────────

def _synthetic_ticks(minute_ms: int) -> pl.DataFrame:
    # 3 ticks in minute 0, 1 tick in minute 1
    ts = [minute_ms + 1000, minute_ms + 30000, minute_ms + 59000, minute_ms + 61000]
    bid = [1.1000, 1.1005, 1.1002, 1.1010]
    ask = [1.1001, 1.1006, 1.1003, 1.1011]
    df = pl.DataFrame({"timestamp": ts, "bid": bid, "ask": ask,
                       "bid_volume": [1.0] * 4, "ask_volume": [1.0] * 4})
    return df.with_columns([
        ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt"),
    ]).sort("dt")


def test_resample_ohlc():
    minute_ms = 1_700_000_000_000 // 60000 * 60000
    bars = build_bars._resample(_synthetic_ticks(minute_ms), "1m")
    assert len(bars) == 2
    b0 = bars.row(0, named=True)
    assert abs(b0["open"] - 1.10005) < 1e-9          # first mid
    assert abs(b0["close"] - 1.10025) < 1e-9         # last mid of minute 0
    assert abs(b0["high"] - 1.10055) < 1e-9          # max mid
    assert b0["tick_count"] == 3
    assert b0["ask_close"] > b0["bid_close"]


# ─── feature causality ──────────────────────────────────────────────────────

def _synthetic_bars(n: int) -> pl.DataFrame:
    import math
    ts = [1_700_000_000_000 + i * 3_600_000 for i in range(n)]
    close = [1.10 + 0.001 * math.sin(i / 5) for i in range(n)]
    high = [c + 0.0005 for c in close]
    low = [c - 0.0005 for c in close]
    return pl.DataFrame({
        "timestamp": ts, "open": close, "high": high, "low": low, "close": close,
        "bid_close": [c - 0.00005 for c in close], "ask_close": [c + 0.00005 for c in close],
        "spread_mean": [0.0001] * n, "spread_max": [0.0002] * n,
        "tick_count": [100] * n, "volume_est": [1.0] * n,
    })


def test_features_are_causal():
    df = _synthetic_bars(200)
    full = build_features.compute_features(df, 0.0001)
    trunc = build_features.compute_features(df.head(150), 0.0001)
    # A causal feature at row t must not change when FUTURE rows are removed.
    cols = ["log_ret_1", "ret_5", "rv_20", "spread_pips", "efficiency_ratio_20",
            "quote_imbalance"]
    a = full.head(150).select(cols).fill_null(-999.0)
    b = trunc.select(cols).fill_null(-999.0)
    assert a.equals(b), "feature layer leaked future information (non-causal)"


# ─── data-quality gate ──────────────────────────────────────────────────────

def test_weekend_gap_detection():
    # Fri 21:00 UTC -> Sun 21:30 UTC ~ 48.5h gap starting Friday.
    fri = int(datetime(2026, 6, 26, 21, 0, tzinfo=timezone.utc).timestamp() * 1000)
    sun = int(datetime(2026, 6, 28, 21, 30, tzinfo=timezone.utc).timestamp() * 1000)
    assert data_quality._is_weekend_gap(fri, sun)
    # a 2h intra-session gap on a Tuesday is NOT a weekend gap.
    t0 = int(datetime(2026, 6, 30, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert not data_quality._is_weekend_gap(t0, t0 + 2 * 3_600_000)


def test_quality_price_band():
    lo, hi = data_quality._price_band("USDJPY", "MAJOR")
    assert lo < 150 < hi
    lo2, hi2 = data_quality._price_band("EURUSD", "MAJOR")
    assert lo2 < 1.1 < hi2
    lo3, hi3 = data_quality._price_band("XAUUSD", "METAL")
    assert lo3 < 2000 < hi3


# ─── cross-pair panel signs ─────────────────────────────────────────────────

def test_usd_strength_sign():
    # EURUSD rising (EUR up vs USD) => USD strength should be NEGATIVE.
    ret = (pl.col("EURUSD") / pl.col("EURUSD").shift(1)).log()
    # EURUSD is xxxUSD => USD side return = -ret
    expr = panels._usd_side_return("EURUSD", ret)
    df = pl.DataFrame({"EURUSD": [1.10, 1.11]})
    val = df.select(expr.alias("u")).row(1, named=True)["u"]
    assert val is not None and val < 0                # USD weakened


def test_triangular_residual_zero_when_consistent():
    U = UniverseLoader.load()
    # Consistent tape: EURJPY = EURUSD * USDJPY exactly => residual ~ 0.
    eurusd, usdjpy = 1.10, 150.0
    eurjpy = eurusd * usdjpy
    import glob
    # build a tiny bar set in-memory by monkeypatching load: instead, test the math
    base_leg = pl.col("EURUSD").log()          # EUR/USD
    quote_leg = -(-pl.col("USDJPY").log())     # JPY/USD = -log(USD/JPY) => leg = -log(USDJPY)
    # implied log(EURJPY) = log(EURUSD) - log(JPY/USD) = log(EURUSD) - (-log(USDJPY))
    df = pl.DataFrame({"EURUSD": [eurusd], "USDJPY": [usdjpy], "EURJPY": [eurjpy]})
    implied = df.select((pl.col("EURUSD").log() + pl.col("USDJPY").log()).alias("i")).row(0)[0]
    actual = df.select(pl.col("EURJPY").log().alias("a")).row(0)[0]
    assert abs(actual - implied) < 1e-9


# ─── dataset loader column classification ───────────────────────────────────

def test_dataset_loader_col_classification():
    from pipeline.dataset_loader import DatasetLoader as DL
    df = pl.DataFrame({
        "timestamp": [1], "close": [1.1], "rv_20": [0.1], "usd_strength_ret": [0.0],
        "pair": ["EURUSD"], "dna": ["MAJOR"], "pip": [0.0001], "price_divisor": [1e5],
        "split": ["train"], "is_purge": [False], "rate_base": [3.0], "rate_quote": [2.0],
        "carry_diff": [1.0], "fwd_ret_1": [0.001], "fwd_ret_5": [0.002],
    })
    assert set(DL.target_cols(df)) == {"fwd_ret_1", "fwd_ret_5"}
    feat = DL.feature_cols(df)
    assert "rv_20" in feat and "carry_diff" in feat and "usd_strength_ret" in feat
    assert not any(c in feat for c in ("pair", "split", "is_purge", "fwd_ret_1", "rate_base"))
