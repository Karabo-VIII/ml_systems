"""Shared builders for offline tests. No network."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl

from pipeline.config import Windows
from pipeline.io import atomic_write_parquet
from pipeline.paths import funding_path, metrics_path, trades_path


def small_windows(**overrides) -> Windows:
    base = dict(
        z=20, fast=10, vol_cluster=10, frac_diff=30, frac_diff_d=0.4,
        hurst=20, kurtosis=15, efficiency=5, perm_entropy=20, perm_m=3,
        kyle=15, yz=10, cs_block=5, funding_mom=8, hawkes_span=8, ma=20,
        horizons=(1, 4, 16, 64),
    )
    base.update(overrides)
    return Windows(**base)


def make_trades(
    n: int,
    *,
    start: datetime,
    step: timedelta,
    price0: float = 100.0,
    qty: float = 0.5,
    seed: int = 0,
    asset_noise: float = 0.001,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    prices = [price0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1.0 + float(rng.normal(0, asset_noise))))
    rows_ts = []
    t = start
    for i in range(n):
        rows_ts.append(t)
        t += step
    ts_ms = [int(x.replace(tzinfo=timezone.utc).timestamp() * 1000) for x in rows_ts]
    qty_arr = qty * (0.5 + rng.random(n))
    # A few large prints so whale / liq proxy are not a flat zero.
    qty_arr[:: max(n // 5, 1)] *= 50
    maker = np.array([i % 3 == 0 for i in range(n)])
    return pl.DataFrame({
        "agg_trade_id": np.arange(1, n + 1, dtype=np.int64),
        "ts_us": np.array(ts_ms, dtype=np.int64) * 1000,
        "ts_ms": np.array(ts_ms, dtype=np.int64),
        "price": np.array(prices, dtype=np.float64),
        "qty": qty_arr.astype(np.float64),
        "is_buyer_maker": maker,
    }).sort(["ts_us", "agg_trade_id"])


def write_bronze_day(root, asset: str, day: date, trades: pl.DataFrame) -> None:
    path = trades_path(root, asset, day)
    atomic_write_parquet(
        trades, path, schema_name="trades_v1",
        params={"asset": asset, "day": day.isoformat()},
        sort_cols=["ts_us", "agg_trade_id"],
    )


def write_funding(root, asset: str, rows: list[tuple[int, float]]) -> None:
    df = pl.DataFrame({
        "ts_ms": [r[0] for r in rows],
        "funding_rate": [r[1] for r in rows],
        "funding_interval_hours": [8] * len(rows),
    })
    ts0 = rows[0][0]
    dt = datetime.fromtimestamp(ts0 / 1000, tz=timezone.utc)
    path = funding_path(root, asset, dt.year, dt.month)
    atomic_write_parquet(
        df, path, schema_name="funding_v1",
        params={"asset": asset}, sort_cols=["ts_ms"],
    )


def write_metrics(root, asset: str, day: date, rows: list[tuple[int, float]]) -> None:
    df = pl.DataFrame({
        "ts_ms": [r[0] for r in rows],
        "sum_open_interest_value": [r[1] for r in rows],
    })
    path = metrics_path(root, asset, day)
    atomic_write_parquet(
        df, path, schema_name="metrics_v1",
        params={"asset": asset, "day": day.isoformat()}, sort_cols=["ts_ms"],
    )
