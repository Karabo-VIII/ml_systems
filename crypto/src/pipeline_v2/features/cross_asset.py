"""Cross-asset features. Null when fewer than two assets are knowable. Ranks need three.

Values are not filled with 0 on a one-asset run.
"""

from __future__ import annotations

import polars as pl

from pipeline_v2.align.asof import asof_join_knowable
from pipeline_v2.constants import DOLLAR_PEER_TOLERANCE_MS, TIME_WIDTH_MS

XD_COLUMNS = (
    "xd_btc_return",
    "xd_btc_volatility",
    "xd_funding_spread",
    "xd_cross_return_mean",
    "xd_cross_vol_mean",
    "xd_ma_distance",
    "xd_momentum_rank",
)


def _prep(df: pl.DataFrame, fast: int, ma: int) -> pl.DataFrame:
    out = df.sort("ts_ms").with_columns([
        pl.col("close").pct_change().alias("_ret"),
        pl.col("close").pct_change().rolling_std(window_size=fast, min_samples=fast).alias("_vol"),
        (
            (pl.col("close") - pl.col("close").rolling_mean(window_size=ma, min_samples=ma))
            / pl.col("close").rolling_mean(window_size=ma, min_samples=ma)
        ).alias("_ma"),
    ])
    if "funding_rate" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("funding_rate"))
    return out


def _horizontal_mean(cols: list[str]) -> pl.Expr:
    if not cols:
        return pl.lit(None, dtype=pl.Float64)
    num = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in cols])
    den = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Float64) for c in cols])
    return pl.when(den > 0).then(num / den).otherwise(None)


def add_cross_asset(
    frames: dict[str, pl.DataFrame],
    *,
    bar_type: str,
    fast: int = 20,
    ma: int = 20,
) -> dict[str, pl.DataFrame]:
    """Join peer state onto each asset. ``ma`` defaults small so callers pass the real window."""
    if not frames:
        return frames
    prepped = {a: _prep(df, fast, ma) for a, df in frames.items() if df is not None and not df.is_empty()}
    if not prepped:
        return frames
    tolerance = TIME_WIDTH_MS.get(bar_type)
    if tolerance is None:
        tolerance = DOLLAR_PEER_TOLERANCE_MS
    # Exact bucket only for time bars: tolerance just under one width, so a
    # missing hour does not silently reuse the previous hour.
    tol = tolerance - 1 if bar_type in TIME_WIDTH_MS else tolerance

    btc = prepped.get("BTCUSDT")
    out: dict[str, pl.DataFrame] = {}
    for asset, base in prepped.items():
        df = base
        peer_ret: list[str] = []
        peer_vol: list[str] = []
        peer_ma: list[str] = []
        peer_fund: list[str] = []
        for other, other_df in prepped.items():
            if other == asset:
                continue
            feat = other_df.select([
                "ts_ms",
                pl.col("_ret").alias(f"_pr_{other}"),
                pl.col("_vol").alias(f"_pv_{other}"),
                pl.col("_ma").alias(f"_pm_{other}"),
                pl.col("funding_rate").alias(f"_pf_{other}"),
            ])
            df = asof_join_knowable(
                df, feat,
                [f"_pr_{other}", f"_pv_{other}", f"_pm_{other}", f"_pf_{other}"],
                publication_lag_ms=0,
                tolerance_ms=tol,
            )
            peer_ret.append(f"_pr_{other}")
            peer_vol.append(f"_pv_{other}")
            peer_ma.append(f"_pm_{other}")
            peer_fund.append(f"_pf_{other}")

        n_peers = (
            pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int64) for c in peer_ret])
            if peer_ret else pl.lit(0)
        )
        n_with_self = n_peers + pl.col("_ret").is_not_null().cast(pl.Int64)
        enough = n_with_self >= 2
        enough_rank = n_with_self >= 3

        if btc is None or asset == "BTCUSDT":
            # BTC's own return is not a cross-asset feature. The pass-through
            # exists for the other names.
            btc_ret = pl.lit(None, dtype=pl.Float64)
            btc_vol = pl.lit(None, dtype=pl.Float64)
        else:
            btc_ret = pl.col("_pr_BTCUSDT") if "_pr_BTCUSDT" in peer_ret else pl.lit(None, dtype=pl.Float64)
            btc_vol = pl.col("_pv_BTCUSDT") if "_pv_BTCUSDT" in peer_vol else pl.lit(None, dtype=pl.Float64)

        # Rank: 1 + number of peers with a strictly higher return. Null below 3 names.
        if peer_ret:
            n_higher = pl.sum_horizontal([
                (pl.col(c) > pl.col("_ret")).fill_null(False).cast(pl.Float64) for c in peer_ret
            ])
        else:
            n_higher = pl.lit(None, dtype=pl.Float64)

        df = df.with_columns([
            pl.when(enough).then(btc_ret).otherwise(None).alias("xd_btc_return"),
            pl.when(enough).then(btc_vol).otherwise(None).alias("xd_btc_volatility"),
            pl.when(enough).then(pl.col("funding_rate") - _horizontal_mean(peer_fund)).otherwise(None).alias("xd_funding_spread"),
            pl.when(enough).then(_horizontal_mean(peer_ret)).otherwise(None).alias("xd_cross_return_mean"),
            pl.when(enough).then(_horizontal_mean(peer_vol)).otherwise(None).alias("xd_cross_vol_mean"),
            pl.when(enough).then(pl.col("_ma") - _horizontal_mean(peer_ma)).otherwise(None).alias("xd_ma_distance"),
            pl.when(enough_rank & pl.col("_ret").is_not_null()).then(1.0 + n_higher).otherwise(None).alias("xd_momentum_rank"),
        ])
        drop = [c for c in df.columns if c.startswith("_pr_") or c.startswith("_pv_") or c.startswith("_pm_") or c.startswith("_pf_") or c in ("_ret", "_vol", "_ma")]
        out[asset] = df.drop(drop)
    # Assets that were empty stay as they were.
    for asset, df in frames.items():
        if asset not in out:
            out[asset] = df
    return out
