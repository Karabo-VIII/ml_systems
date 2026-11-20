"""Daily panels from trades and from small bronze series.

A day-D aggregate is knowable at (D+1) 00:00:00 UTC. ``event_time_ms`` is
D 00:00:00 UTC and the caller adds ``lag_ms = 86_400_000``.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import numpy as np
import polars as pl

from pipeline.constants import DAY_MS

DAY_MS_I = DAY_MS


def day_start_ms(day: date) -> int:
    dt = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _eta(counts: np.ndarray) -> float:
    """Hawkes branching ratio. Null (nan) when the formula is not defined.

    v1 wrote 0.0 for sub-Poisson days. That constant is not written here.
    """
    if len(counts) < 30:
        return float("nan")
    mu = float(np.mean(counts))
    var = float(np.var(counts, ddof=1))
    if mu <= 0 or var <= mu:
        return float("nan")
    eta = 1.0 - math.sqrt(mu / var)
    return float(min(0.99, max(0.0, eta)))


def _minute_counts(ts_ms: np.ndarray, mask: np.ndarray, day_ms: int) -> np.ndarray:
    """1440 bins covering [day_ms, day_ms + 1 day), zeros included."""
    counts = np.zeros(1440, dtype=np.float64)
    if mask.any():
        minutes = ((ts_ms[mask] - day_ms) // 60_000).astype(np.int64)
        minutes = minutes[(minutes >= 0) & (minutes < 1440)]
        np.add.at(counts, minutes, 1.0)
    return counts


def summarize_day(
    trades: pl.DataFrame,
    bars_1m: pl.DataFrame,
    *,
    asset: str,
    day: date,
    whale_usd: float,
    liq_usd: float,
) -> dict:
    """One row of whale, liquidation-proxy, realized-vol, and Hawkes features."""
    event = day_start_ms(day)
    row: dict = {
        "asset": asset,
        "event_time_ms": event,
        "wh_large_count": 0,
        "wh_large_usd": 0.0,
        "wh_large_buy_usd": 0.0,
        "wh_large_sell_usd": 0.0,
        "wh_max_print_usd": 0.0,
        "liq_proxy_count": 0,
        "liq_proxy_usd": 0.0,
        "liq_proxy_buy_usd": 0.0,
        "liq_proxy_sell_usd": 0.0,
        "rv_realized": None,
        "rv_bpv": None,
        "rv_jv": None,
        "rv_jump_frac": None,
        "hbr_eta": None,
        "hbr_eta_buy": None,
        "hbr_eta_sell": None,
        "hbr_eta_imbalance": None,
        "hbr_n_trades": 0,
    }
    if trades is not None and not trades.is_empty():
        t = trades.with_columns((pl.col("price") * pl.col("qty")).alias("_usd"))
        usd = t["_usd"].to_numpy()
        maker = t["is_buyer_maker"].to_numpy().astype(bool)
        large = usd >= whale_usd
        proxy = usd >= liq_usd
        row["wh_large_count"] = int(large.sum())
        row["wh_large_usd"] = float(usd[large].sum()) if large.any() else 0.0
        row["wh_large_buy_usd"] = float(usd[large & ~maker].sum()) if large.any() else 0.0
        row["wh_large_sell_usd"] = float(usd[large & maker].sum()) if large.any() else 0.0
        row["wh_max_print_usd"] = float(usd.max()) if len(usd) else 0.0
        row["liq_proxy_count"] = int(proxy.sum())
        row["liq_proxy_usd"] = float(usd[proxy].sum()) if proxy.any() else 0.0
        row["liq_proxy_buy_usd"] = float(usd[proxy & ~maker].sum()) if proxy.any() else 0.0
        row["liq_proxy_sell_usd"] = float(usd[proxy & maker].sum()) if proxy.any() else 0.0
        row["hbr_n_trades"] = int(t.height)
        ts = t["ts_ms"].to_numpy()
        row["hbr_eta"] = _eta(_minute_counts(ts, np.ones(len(ts), dtype=bool), event))
        row["hbr_eta_buy"] = _eta(_minute_counts(ts, ~maker, event))
        row["hbr_eta_sell"] = _eta(_minute_counts(ts, maker, event))
        if row["hbr_eta_buy"] == row["hbr_eta_buy"] and row["hbr_eta_sell"] == row["hbr_eta_sell"]:
            row["hbr_eta_imbalance"] = row["hbr_eta_buy"] - row["hbr_eta_sell"]

    if bars_1m is not None and bars_1m.height >= 3:
        close = bars_1m.sort("ts_ms")["close"].to_numpy().astype(float)
        prev = close[:-1]
        ok = (prev > 0) & (close[1:] > 0)
        r = np.full(len(close) - 1, np.nan)
        r[ok] = np.log(close[1:][ok] / prev[ok])
        r = r[np.isfinite(r)]
        if len(r) >= 2:
            rv = float(np.sum(r ** 2))
            bpv = float((math.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
            jv = max(rv - bpv, 0.0)
            row["rv_realized"] = rv
            row["rv_bpv"] = bpv
            row["rv_jv"] = jv
            row["rv_jump_frac"] = (jv / rv) if rv > 0 else None
    # NaN -> None for parquet nulls
    for k, v in list(row.items()):
        if isinstance(v, float) and not math.isfinite(v):
            row[k] = None
    return row


def rows_to_frame(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={"asset": pl.Utf8, "event_time_ms": pl.Int64})
    return pl.DataFrame(rows).sort(["asset", "event_time_ms"])


def funding_daily(funding: pl.DataFrame, asset: str) -> pl.DataFrame:
    if funding is None or funding.is_empty():
        return pl.DataFrame(schema={
            "asset": pl.Utf8, "event_time_ms": pl.Int64,
            "funding_day_mean": pl.Float64, "funding_day_sum": pl.Float64,
            "funding_day_last": pl.Float64,
        })
    df = funding.with_columns(((pl.col("ts_ms") // DAY_MS_I) * DAY_MS_I).alias("event_time_ms"))
    # Settlement count is usually 3 and would be a constant feature. Keep the
    # rate statistics, which are the actual signal.
    out = df.group_by("event_time_ms", maintain_order=True).agg([
        pl.col("funding_rate").mean().alias("funding_day_mean"),
        pl.col("funding_rate").sum().alias("funding_day_sum"),
        pl.col("funding_rate").sort_by("ts_ms").last().alias("funding_day_last"),
    ])
    return out.with_columns(pl.lit(asset).alias("asset")).sort("event_time_ms")


def metrics_daily(metrics: pl.DataFrame, asset: str) -> pl.DataFrame:
    """Mean of the numeric columns Vision actually shipped, prefixed ``s3_``.

    ``long_short_ratio`` is not created. A column that is absent stays absent.
    """
    if metrics is None or metrics.is_empty():
        return pl.DataFrame(schema={"asset": pl.Utf8, "event_time_ms": pl.Int64})
    numeric = [
        c for c, dt in zip(metrics.columns, metrics.dtypes)
        if c != "ts_ms" and dt in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]
    if not numeric:
        return pl.DataFrame(schema={"asset": pl.Utf8, "event_time_ms": pl.Int64})
    df = metrics.with_columns(((pl.col("ts_ms") // DAY_MS_I) * DAY_MS_I).alias("event_time_ms"))
    aggs = [pl.col(c).mean().alias(f"s3_{c}_mean") for c in numeric]
    out = df.group_by("event_time_ms", maintain_order=True).agg(aggs)
    return out.with_columns(pl.lit(asset).alias("asset")).sort("event_time_ms")


def basis_daily(spot_1d: pl.DataFrame, perp_1d: pl.DataFrame, asset: str) -> pl.DataFrame:
    """Basis from true daily closes. Null when either close is missing. Never 0-filled."""
    if spot_1d is None or perp_1d is None or spot_1d.is_empty() or perp_1d.is_empty():
        return pl.DataFrame(schema={"asset": pl.Utf8, "event_time_ms": pl.Int64, "bs_basis": pl.Float64})
    # A 1d bar labelled at the close (next midnight) describes the day that opened one day earlier.
    spot = spot_1d.select([
        (pl.col("ts_ms") - DAY_MS_I).alias("event_time_ms"),
        pl.col("close").alias("_spot"),
    ])
    perp = perp_1d.select([
        (pl.col("ts_ms") - DAY_MS_I).alias("event_time_ms"),
        pl.col("close").alias("_perp"),
    ])
    joined = spot.join(perp, on="event_time_ms", how="inner")
    joined = joined.with_columns(
        pl.when(pl.col("_spot") > 0).then((pl.col("_perp") - pl.col("_spot")) / pl.col("_spot")).otherwise(None).alias("bs_basis")
    )
    return joined.select(["event_time_ms", "bs_basis"]).with_columns(pl.lit(asset).alias("asset"))
