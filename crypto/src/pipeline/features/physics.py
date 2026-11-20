"""On-bar physics. Formulas follow the v50 set. Warmup and missing joins stay null.

``norm_oi_change`` is computed only when ``include_oi`` is set. ``norm_bar_duration``
is not emitted on fixed time bars.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from pipeline.constants import MARKET_COLUMNS
from pipeline.contract import Windows
from pipeline.features.rolling import (
    frac_diff,
    hurst_rs,
    kyle_lambda,
    permutation_entropy,
    rolling_z,
    to_series,
)

PHYSICS_FEATURES = (
    "norm_deviation",
    "norm_fd_close",
    "norm_vpin",
    "norm_flow_imbalance",
    "norm_vol_cluster",
    "norm_funding",
    "norm_tick_count",
    "norm_log_volume",
    "norm_hl_spread",
    "hurst_regime",
    "norm_return_1",
    "norm_return_4",
    "norm_return_16",
    "norm_spread_bps",
    "norm_ma_distance",
    "norm_whale",
    "norm_efficiency",
    "norm_return_kurtosis",
    "norm_funding_momentum",
    "norm_hawkes_intensity",
    "norm_hawkes_buy_intensity",
    "norm_hawkes_sell_intensity",
    "norm_hawkes_imbalance",
    "norm_momentum_accel",
    "norm_vol_price_corr",
    "norm_vol_ratio",
    "norm_flow_persistence",
    "norm_yz_volatility",
    "norm_cs_spread",
    "norm_perm_entropy",
    "norm_kyle_lambda",
)

OI_FEATURES = ("norm_oi_change", "norm_oi_price_divergence")


def _log_ratio(num: str, den: str) -> pl.Expr:
    n = pl.col(num)
    d = pl.col(den)
    return pl.when((n > 0) & (d > 0)).then((n / d).log()).otherwise(None)


def _corr_roll(x: str, y: str, window: int, out: str) -> pl.Expr:
    """Trailing Pearson correlation. Null until the window is full. No zero-fill."""
    xs = pl.col(x)
    ys = pl.col(y)
    mx = xs.rolling_mean(window_size=window, min_samples=window)
    my = ys.rolling_mean(window_size=window, min_samples=window)
    cov = ((xs - mx) * (ys - my)).rolling_mean(window_size=window, min_samples=window)
    sx = xs.rolling_std(window_size=window, min_samples=window)
    sy = ys.rolling_std(window_size=window, min_samples=window)
    corr = cov / (sx * sy + 1e-12)
    return pl.when(sx.is_null() | sy.is_null() | (sx < 1e-12) | (sy < 1e-12)).then(None).otherwise(corr.clip(-1.0, 1.0)).alias(out)


def compute_physics(
    df: pl.DataFrame,
    windows: Windows,
    *,
    include_oi: bool = False,
    include_bar_duration: bool | None = None,
) -> pl.DataFrame:
    if df.is_empty():
        return df
    bar_type = df["bar_type"][0] if "bar_type" in df.columns else ""
    if include_bar_duration is None:
        include_bar_duration = not str(bar_type).startswith("time_")

    df = df.sort("ts_ms")
    if "funding_rate" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("funding_rate"))

    w = windows
    df = df.with_columns([
        pl.col("close").pct_change().alias("_ret"),
        ((pl.col("buy_volume") - pl.col("sell_volume")) / pl.col("volume")).alias("_flow"),
        ((pl.col("buy_volume") - pl.col("sell_volume")).abs() / pl.col("volume")).alias("_vpin"),
        (pl.col("close").ewm_mean(span=12, adjust=False) - pl.col("close").ewm_mean(span=26, adjust=False))
        .truediv(pl.col("close").ewm_mean(span=26, adjust=False))
        .alias("_dev"),
        pl.when(pl.col("n_trades") > 0).then(pl.col("volume") / pl.col("n_trades")).otherwise(None).alias("_whale"),
        pl.when(pl.col("volume") > 0).then(pl.col("volume").log()).otherwise(None).alias("_log_vol"),
        pl.when(pl.col("n_trades") > 0).then(pl.col("n_trades").cast(pl.Float64)).otherwise(None).alias("_ticks"),
    ])
    # Rogers-Satchell, null when a price is not positive.
    rs_var = (
        _log_ratio("high", "close") * _log_ratio("high", "open")
        + _log_ratio("low", "close") * _log_ratio("low", "open")
    )
    df = df.with_columns(rs_var.alias("_rs_var"))
    df = df.with_columns(
        pl.when(pl.col("_rs_var").is_null()).then(None)
        .when(pl.col("_rs_var") < 0).then(pl.lit(0.0))
        .otherwise(pl.col("_rs_var"))
        .sqrt()
        .alias("_rs")
    )
    df = df.with_columns([
        pl.col("_dev").rolling_std(window_size=w.vol_cluster, min_samples=w.vol_cluster).alias("_vol_cluster"),
        (pl.col("close").shift(1) - pl.col("close").shift(1 + 4))
        .truediv(pl.col("close").shift(1 + 4))
        .alias("_ret4"),
        (pl.col("close").shift(1) - pl.col("close").shift(1 + 16))
        .truediv(pl.col("close").shift(1 + 16))
        .alias("_ret16"),
        pl.col("_ret").shift(1).alias("_ret1"),
        pl.when((pl.col("buy_volume") + pl.col("sell_volume")) > 0)
        .then(
            ((pl.col("high") - pl.col("low")) / pl.col("close"))
            * (1.0 + ((pl.col("buy_volume") - pl.col("sell_volume")).abs() / (pl.col("buy_volume") + pl.col("sell_volume"))))
            * 10_000.0
        )
        .otherwise(None)
        .alias("_spread_bps"),
        (
            (pl.col("close") - pl.col("close").shift(w.efficiency)).abs()
            / ((pl.col("close") - pl.col("close").shift(1)).abs().rolling_sum(window_size=w.efficiency, min_samples=w.efficiency) + 1e-12)
        ).alias("_eff"),
        (pl.col("funding_rate") - pl.col("funding_rate").rolling_mean(window_size=w.funding_mom, min_samples=w.funding_mom)).alias("_fund_mom"),
        (
            (pl.col("close") - pl.col("close").rolling_mean(window_size=w.ma, min_samples=w.ma))
            / pl.col("close").rolling_mean(window_size=w.ma, min_samples=w.ma)
        ).alias("_ma_dist"),
    ])
    # Duration and Hawkes. Duration is almost constant on time bars, so the
    # feature is omitted there; the rate-versus-EMA Hawkes signal is not.
    dur_s = pl.col("ts_ms").diff() / 1000.0
    df = df.with_columns(dur_s.alias("_dur_s"))
    df = df.with_columns([
        pl.when(pl.col("_dur_s") > 0).then(pl.col("_dur_s").log()).otherwise(None).alias("_bar_dur"),
        pl.when(pl.col("_dur_s") > 0).then(pl.col("n_trades") / pl.col("_dur_s")).otherwise(None).alias("_tick_rate"),
        pl.when(pl.col("_dur_s") > 0).then(pl.col("buy_volume") / pl.col("_dur_s")).otherwise(None).alias("_buy_rate"),
        pl.when(pl.col("_dur_s") > 0).then(pl.col("sell_volume") / pl.col("_dur_s")).otherwise(None).alias("_sell_rate"),
    ])
    df = df.with_columns([
        (pl.col("_tick_rate") / pl.col("_tick_rate").ewm_mean(span=w.hawkes_span, adjust=False, ignore_nulls=True) - 1.0).alias("_h_all"),
        (pl.col("_buy_rate") / pl.col("_buy_rate").ewm_mean(span=w.hawkes_span, adjust=False, ignore_nulls=True) - 1.0).alias("_h_buy"),
        (pl.col("_sell_rate") / pl.col("_sell_rate").ewm_mean(span=w.hawkes_span, adjust=False, ignore_nulls=True) - 1.0).alias("_h_sell"),
    ])
    df = df.with_columns((pl.col("_h_buy") - pl.col("_h_sell")).alias("_h_imb"))
    df = df.with_columns((pl.col("_ret4") - pl.col("_ret4").shift(4)).alias("_mom_acc"))

    # Kurtosis of returns. Null while the window is incomplete.
    k = w.kurtosis
    ret = pl.col("_ret")
    mu = ret.rolling_mean(window_size=k, min_samples=k)
    sd = ret.rolling_std(window_size=k, min_samples=k)
    df = df.with_columns(
        (((ret - mu).pow(4)).rolling_mean(window_size=k, min_samples=k) / sd.pow(4) - 3.0).alias("_kurt")
    )

    # Vol term structure on Rogers-Satchell.
    df = df.with_columns(
        (pl.col("_rs").ewm_mean(span=10, adjust=False, ignore_nulls=True)
         / pl.col("_rs").ewm_mean(span=max(w.fast, 2), adjust=False, ignore_nulls=True)
         - 1.0).alias("_vol_ratio")
    )
    # The long leg of the ratio is a slower EMA. Recompute with span 50 when fast is small.
    df = df.with_columns(
        (pl.col("_rs").ewm_mean(span=10, adjust=False, ignore_nulls=True)
         / (pl.col("_rs").ewm_mean(span=50, adjust=False, ignore_nulls=True) + 1e-12)
         - 1.0).alias("_vol_ratio")
    )

    df = df.with_columns([
        pl.col("_ret").abs().alias("_abs_ret"),
        pl.col("_flow").shift(1).alias("_flow_lag"),
    ])
    df = df.with_columns([
        _corr_roll("_log_vol", "_abs_ret", w.fast, "_vp_corr"),
        _corr_roll("_flow", "_flow_lag", w.fast, "_flow_pers"),
    ])

    # Yang-Zhang.
    n_yz = w.yz
    yz_k = 0.34 / (1.34 + (n_yz + 1) / (n_yz - 1)) if n_yz > 1 else 0.34
    df = df.with_columns([
        pl.when((pl.col("open") > 0) & (pl.col("close").shift(1) > 0))
        .then((pl.col("open") / pl.col("close").shift(1)).log())
        .otherwise(None)
        .alias("_yz_on"),
        pl.when((pl.col("close") > 0) & (pl.col("open") > 0))
        .then((pl.col("close") / pl.col("open")).log())
        .otherwise(None)
        .alias("_yz_oc"),
    ])
    df = df.with_columns(
        (
            pl.col("_yz_on").pow(2).rolling_mean(window_size=n_yz, min_samples=n_yz)
            + yz_k * pl.col("_yz_oc").pow(2).rolling_mean(window_size=n_yz, min_samples=n_yz)
            + (1.0 - yz_k) * pl.when(pl.col("_rs_var") < 0).then(0.0).otherwise(pl.col("_rs_var")).rolling_mean(window_size=n_yz, min_samples=n_yz)
        ).clip(lower_bound=0.0).sqrt().alias("_yz")
    )

    # Corwin-Schultz alpha over a trailing block. Null when the range is zero.
    b = w.cs_block
    h_block = pl.col("high").rolling_max(window_size=b, min_samples=b)
    l_block = pl.col("low").rolling_min(window_size=b, min_samples=b)
    h_prev = pl.col("high").shift(b).rolling_max(window_size=b, min_samples=b)
    l_prev = pl.col("low").shift(b).rolling_min(window_size=b, min_samples=b)
    beta = (
        pl.when((h_block > l_block) & (l_block > 0) & (h_prev > l_prev) & (l_prev > 0))
        .then(((h_block / l_block).log().pow(2) + (h_prev / l_prev).log().pow(2)) / 2.0)
        .otherwise(None)
    )
    h2 = pl.col("high").rolling_max(window_size=2 * b, min_samples=2 * b)
    l2 = pl.col("low").rolling_min(window_size=2 * b, min_samples=2 * b)
    gamma = pl.when((h2 > l2) & (l2 > 0)).then((h2 / l2).log().pow(2)).otherwise(None)
    sqrt2 = math.sqrt(2.0)
    denom = 3.0 - 2.0 * sqrt2
    alpha = ((2.0 * beta).sqrt() - beta.sqrt()) / denom - (gamma / denom).sqrt()
    df = df.with_columns(alpha.alias("_cs"))

    # Numpy features on the close path. They see only the past.
    close = df["close"].to_numpy().astype(np.float64)
    log_close = np.where(close > 0, np.log(close), np.nan)
    log_ret = np.full(len(close), np.nan)
    log_ret[1:] = np.diff(log_close)
    fd = frac_diff(log_close, d=w.frac_diff_d, window=w.frac_diff)
    hurst = hurst_rs(log_ret, window=w.hurst)
    ret_np = df["_ret"].to_numpy().astype(np.float64)
    pent = permutation_entropy(ret_np, window=w.perm_entropy, m=w.perm_m)
    signed = df["signed_usd"].to_numpy().astype(np.float64) if "signed_usd" in df.columns else np.zeros(len(close))
    kyle = kyle_lambda(close, signed, window=w.kyle)
    df = df.with_columns([
        to_series("_fd", fd),
        to_series("_hurst", hurst),
        to_series("_pent", pent),
        to_series("_kyle", kyle),
    ])

    zmap = {
        "norm_deviation": ("_dev", w.z),
        "norm_fd_close": ("_fd", w.z),
        "norm_vpin": ("_vpin", w.z),
        "norm_flow_imbalance": ("_flow", w.z),
        "norm_vol_cluster": ("_vol_cluster", w.z),
        "norm_funding": ("funding_rate", w.z),
        "norm_tick_count": ("_ticks", w.z),
        "norm_log_volume": ("_log_vol", w.z),
        "norm_hl_spread": ("_rs", w.z),
        "hurst_regime": ("_hurst", w.z),
        "norm_return_1": ("_ret1", w.z),
        "norm_return_4": ("_ret4", w.z),
        "norm_return_16": ("_ret16", w.z),
        "norm_spread_bps": ("_spread_bps", w.z),
        "norm_ma_distance": ("_ma_dist", w.z),
        "norm_whale": ("_whale", w.z),
        "norm_efficiency": ("_eff", w.z),
        "norm_return_kurtosis": ("_kurt", w.z),
        "norm_funding_momentum": ("_fund_mom", w.z),
        "norm_hawkes_intensity": ("_h_all", w.z),
        "norm_hawkes_buy_intensity": ("_h_buy", w.z),
        "norm_hawkes_sell_intensity": ("_h_sell", w.z),
        "norm_hawkes_imbalance": ("_h_imb", w.z),
        "norm_momentum_accel": ("_mom_acc", w.z),
        "norm_vol_price_corr": ("_vp_corr", w.z),
        "norm_vol_ratio": ("_vol_ratio", w.z),
        "norm_flow_persistence": ("_flow_pers", w.z),
        "norm_yz_volatility": ("_yz", w.z),
        "norm_cs_spread": ("_cs", w.z),
        "norm_perm_entropy": ("_pent", w.z),
        "norm_kyle_lambda": ("_kyle", w.z),
    }
    df = df.with_columns([rolling_z(src, window).alias(name) for name, (src, window) in zmap.items()])

    # Fractional-difference z-score is not meaningful until both windows have passed.
    fd_warmup = w.frac_diff + w.z
    hurst_warmup = w.hurst + w.z
    df = df.with_row_index("_i")
    df = df.with_columns([
        pl.when(pl.col("_i") < fd_warmup).then(None).otherwise(pl.col("norm_fd_close")).alias("norm_fd_close"),
        pl.when(pl.col("_i") < hurst_warmup).then(None).otherwise(pl.col("hurst_regime")).alias("hurst_regime"),
    ])

    if include_bar_duration:
        df = df.with_columns(rolling_z("_bar_dur", w.z).alias("norm_bar_duration"))

    if include_oi and "oi_value" in df.columns:
        med = pl.col("oi_value").rolling_median(window_size=50, min_samples=20)
        capped = pl.when(pl.col("oi_value").is_not_null() & med.is_not_null() & (pl.col("oi_value") > 5.0 * med)).then(5.0 * med).otherwise(pl.col("oi_value"))
        df = df.with_columns(capped.alias("_oi_cap"))
        df = df.with_columns(pl.col("_oi_cap").pct_change().alias("_oi_chg"))
        df = df.with_columns(rolling_z("_oi_chg", w.z).alias("norm_oi_change"))
        df = df.with_columns(
            (pl.col("_oi_chg").abs().ewm_mean(span=10, adjust=False, ignore_nulls=True)
             / (pl.col("_ret").abs().ewm_mean(span=10, adjust=False, ignore_nulls=True) + 1e-12)
             - 1.0).alias("_oi_div")
        )
        df = df.with_columns(rolling_z("_oi_div", w.z).alias("norm_oi_price_divergence"))

    # Regime label. Null during the moving-average warmup. Not a training feature.
    ma_std = pl.col("_ma_dist").rolling_std(window_size=w.ma, min_samples=w.ma)
    df = df.with_columns(ma_std.alias("_ma_std"))
    df = df.with_columns(
        pl.when(pl.col("_ma_dist").is_null() | pl.col("_ma_std").is_null()).then(None)
        .when(pl.col("_ma_dist") > 0.5 * pl.col("_ma_std")).then(pl.lit(2, dtype=pl.Int8))
        .when(pl.col("_ma_dist") < -0.5 * pl.col("_ma_std")).then(pl.lit(0, dtype=pl.Int8))
        .otherwise(pl.lit(1, dtype=pl.Int8))
        .alias("regime_label")
    )

    lookback = w.max_lookback
    df = df.with_columns((pl.col("_i") < lookback).alias("is_warmup"))

    keep = [c for c in df.columns if c in MARKET_COLUMNS or c in PHYSICS_FEATURES or c in OI_FEATURES or c in ("norm_bar_duration", "regime_label", "is_warmup", "funding_rate", "oi_value")]
    # Preserve market columns that were on the input even if not in the constant tuple.
    for c in df.columns:
        if c.startswith("_"):
            continue
        if c not in keep:
            # drop raw helpers only
            if c in ("funding_rate", "oi_value"):
                keep.append(c)
    return df.select(keep)
