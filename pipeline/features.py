"""Causal rolling helpers. Warmup stays null. Nothing is filled with 0."""

from __future__ import annotations

import math
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl


def rolling_z(col: str, window: int) -> pl.Expr:
    """Trailing z-score. Null until ``window`` finite observations. Null when sigma is ~0."""
    x = pl.col(col)
    mu = x.rolling_mean(window_size=window, min_samples=window)
    sd = x.rolling_std(window_size=window, min_samples=window)
    z = (x - mu) / (sd + 1e-5)
    return pl.when(sd.is_null() | (sd < 1e-12) | x.is_null()).then(None).otherwise(z.clip(-5.0, 5.0))


def frac_diff(series: np.ndarray, d: float, window: int) -> np.ndarray:
    """Fractional difference. ``out[i]`` uses ``series[i-window:i]`` only (excludes i)."""
    n = len(series)
    out = np.full(n, np.nan)
    if window < 1 or n <= window:
        return out
    w = np.empty(window, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, window):
        w[k] = -w[k - 1] / k * (d - k + 1)
    weights = np.ascontiguousarray(w[::-1])
    values = np.asarray(series, dtype=np.float64)
    if not np.isfinite(values).all():
        for i in range(window, n):
            chunk = values[i - window:i]
            if np.isfinite(chunk).all():
                out[i] = float(np.dot(weights, chunk))
        return out
    view = np.lib.stride_tricks.sliding_window_view(values, window)
    # view[i-window] = values[i-window:i] for i in [window, n)
    out[window:] = view[: n - window] @ weights
    return out


def hurst_rs(log_returns: np.ndarray, window: int) -> np.ndarray:
    """Rescaled-range Hurst on a trailing window of log returns. Null during warmup."""
    n = len(log_returns)
    out = np.full(n, np.nan)
    if n <= window:
        return out
    values = np.asarray(log_returns, dtype=np.float64)
    log_w = math.log(window)
    for i in range(window, n):
        chunk = values[i - window:i]
        if not np.isfinite(chunk).all():
            continue
        if np.max(chunk) == np.min(chunk):
            out[i] = 0.5
            continue
        mean = chunk.mean()
        z = np.cumsum(chunk - mean)
        r = float(np.max(z) - np.min(z))
        s = float(np.std(chunk))
        if s < 1e-12 or r <= 0:
            out[i] = 0.5
        else:
            out[i] = math.log(r / s) / log_w
    return out


def _pattern3(a: float, b: float, c: float) -> int:
    """Ordinal pattern of a triple. Ties keep the earlier index smaller."""
    if a <= b:
        if b <= c:
            return 0
        if a <= c:
            return 1
        return 2
    if a <= c:
        return 3
    if b <= c:
        return 4
    return 5


def permutation_entropy(returns: np.ndarray, window: int, m: int = 3) -> np.ndarray:
    """Normalized permutation entropy in [0, 1]. Null until the window is full.

    The value at t uses returns in ``[t-window+1, t]`` only.
    """
    n = len(returns)
    out = np.full(n, np.nan)
    if m != 3 or window < m or n < window:
        return out
    values = np.asarray(returns, dtype=np.float64)
    log_nf = math.log(6.0)
    for t in range(window - 1, n):
        counts = [0, 0, 0, 0, 0, 0]
        left = t - window + 1
        total = 0
        last_start = t - m + 1
        for i in range(left, last_start + 1):
            a, b, c = values[i], values[i + 1], values[i + 2]
            if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
                continue
            counts[_pattern3(a, b, c)] += 1
            total += 1
        if total == 0:
            continue
        entropy = 0.0
        for c in counts:
            if c:
                p = c / total
                entropy -= p * math.log(p)
        out[t] = entropy / log_nf
    return out


def kyle_lambda(close: np.ndarray, signed_usd: np.ndarray, window: int) -> np.ndarray:
    """Trailing correlation of log return with signed USD. Null during warmup."""
    n = len(close)
    out = np.full(n, np.nan)
    log_ret = np.full(n, np.nan)
    for i in range(1, n):
        prev = close[i - 1]
        if prev > 0 and math.isfinite(prev) and math.isfinite(close[i]) and close[i] > 0:
            log_ret[i] = math.log(close[i] / prev)
    signed = np.asarray(signed_usd, dtype=np.float64)
    for t in range(window, n):
        lr = log_ret[t - window:t]
        sv = signed[t - window:t]
        mask = np.isfinite(lr) & np.isfinite(sv)
        if int(mask.sum()) < window:
            continue
        x = sv[mask]
        y = lr[mask]
        x = x - x.mean()
        y = y - y.mean()
        denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
        if denom > 1e-15:
            out[t] = float(np.dot(x, y) / denom)
    return out


def to_series(name: str, values: np.ndarray) -> pl.Series:
    s = pl.Series(name, values)
    return s.fill_nan(None)

"""On-bar physics. Formulas follow the v50 set. Warmup and missing joins stay null.

``norm_oi_change`` is computed only when ``include_oi`` is set. ``norm_bar_duration``
is not emitted on fixed time bars.
"""


import math

import numpy as np
import polars as pl

from pipeline.constants import MARKET_COLUMNS
from pipeline.config import Windows

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

"""Daily panels from trades and from small bronze series.

A day-D aggregate is knowable at (D+1) 00:00:00 UTC. ``event_time_ms`` is
D 00:00:00 UTC and the caller adds ``lag_ms = 86_400_000``.
"""


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


# Causal whale size: a log-histogram of per-trade notional, one day at a time.
# Bins cover [$1, $100M] in log10 space (16 per decade) plus under/overflow.
WHALE_LOG_LO = 0.0
WHALE_LOG_HI = 8.0
WHALE_BINS_PER_DECADE = 16
WHALE_N_BINS = int((WHALE_LOG_HI - WHALE_LOG_LO) * WHALE_BINS_PER_DECADE)


def trade_notional_hist(usd: np.ndarray | None) -> np.ndarray:
    """Integer counts, length ``WHALE_N_BINS + 2`` (underflow, bins, overflow)."""
    hist = np.zeros(WHALE_N_BINS + 2, dtype=np.int64)
    if usd is None or len(usd) == 0:
        return hist
    x = np.asarray(usd, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return hist
    scale = WHALE_N_BINS / (WHALE_LOG_HI - WHALE_LOG_LO)
    idx = np.floor((np.log10(x) - WHALE_LOG_LO) * scale).astype(np.int64) + 1
    np.add.at(hist, np.clip(idx, 0, WHALE_N_BINS + 1), 1)
    return hist


def quantile_from_hists(hists: list[np.ndarray], q: float) -> float | None:
    """Approximate ``q``-quantile of the pooled histograms, in USD."""
    if not hists:
        return None
    hist = np.sum(np.stack(hists, axis=0), axis=0)
    total = int(hist.sum())
    if total <= 0:
        return None
    target = float(q) * total
    cum = 0.0
    width = (WHALE_LOG_HI - WHALE_LOG_LO) / WHALE_N_BINS
    for i, n in enumerate(hist):
        cum += n
        if cum >= target:
            if i == 0:
                return 10.0 ** WHALE_LOG_LO
            if i >= WHALE_N_BINS + 1:
                return 10.0 ** WHALE_LOG_HI
            lo = WHALE_LOG_LO + (i - 1) * width
            return float(10.0 ** (lo + 0.5 * width))
    return 10.0 ** WHALE_LOG_HI


def whale_threshold(history: list[np.ndarray], spec) -> float:
    """Size for a new UTC day from complete days already in ``history``. Causal."""
    usable = [h for h in history if h is not None and int(np.asarray(h).sum()) > 0]
    if not usable:
        return float(spec.bootstrap)
    value = quantile_from_hists(usable[-int(spec.window_days):], float(spec.quantile))
    return float(value) if value is not None else float(spec.bootstrap)


def load_whale_hists(root, asset: str, before: date, window_days: int) -> list[np.ndarray]:
    """Histograms of complete trade days strictly before ``before``, oldest first."""
    from datetime import timedelta
    from pipeline.paths import trades_path

    out: list[np.ndarray] = []
    day = before - timedelta(days=1)
    oldest = before - timedelta(days=int(window_days) * 4 + 30)
    while day >= oldest and len(out) < int(window_days):
        path = trades_path(root, asset, day)
        if path.exists():
            usd = (
                pl.scan_parquet(path)
                .select((pl.col("price") * pl.col("qty")).alias("u"))
                .collect()["u"]
                .to_numpy()
            )
            out.append(trade_notional_hist(usd))
        day -= timedelta(days=1)
    out.reverse()
    return out


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
        "wh_threshold_usd": float(whale_usd),
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

"""Cross-asset features. Null when fewer than two assets are knowable. Ranks need three.

Values are not filled with 0 on a one-asset run.
"""


import polars as pl

from pipeline.align import asof_join_knowable
from pipeline.constants import DOLLAR_PEER_TOLERANCE_MS, TIME_WIDTH_MS

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


# Sound subset of the old EURUSD 1h manifest. Windows are in bars of the
# frame they are computed on. Warmup is null. Nothing is filled with 0.
FX_FEATURES = (
    "log_ret_1",
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_50",
    "rv_10", "rv_20", "rv_50",
    "parkinson_10", "parkinson_20", "parkinson_50",
    "garman_klass_20",
    "atr_pips_10", "atr_pips_20", "atr_pips_50",
    "spread_pips", "rel_spread_bps", "spread_z_100",
    "tick_intensity_z_100",
    "efficiency_ratio_20", "efficiency_ratio_50",
    "is_tokyo", "is_london", "is_newyork", "is_ldn_ny_overlap",
    "weekend_gap_pips",
    "usd_strength_ret",
)

_LN2 = math.log(2.0)
_TOKYO = ZoneInfo("Asia/Tokyo")
_LONDON = ZoneInfo("Europe/London")
_NY = ZoneInfo("America/New_York")


def _session_bounds(day, zone: ZoneInfo, start_h: int, end_h: int) -> tuple[int, int]:
    start = datetime(day.year, day.month, day.day, start_h, 0, tzinfo=zone)
    end = datetime(day.year, day.month, day.day, end_h, 0, tzinfo=zone)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _overlaps(open_ms: int, close_ms: int, start_ms: int, end_ms: int) -> bool:
    return open_ms < end_ms and close_ms > start_ms


def _session_flags(ts_open: list[int], ts_close: list[int]) -> dict[str, list[int]]:
    """1 when the bar interval overlaps the local session. DST follows the zone."""
    tokyo, london, ny = [], [], []
    for open_ms, close_ms in zip(ts_open, ts_close):
        start_day = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc).date()
        hit_t = hit_l = hit_n = False
        for delta in (-1, 0, 1):
            day = start_day + timedelta(days=delta)
            # Tokyo 09:00-18:00 JST is 00:00-09:00 UTC, and Japan does not use DST.
            t0 = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
            hit_t = hit_t or _overlaps(open_ms, close_ms, t0, t0 + 9 * 3_600_000)
            l0, l1 = _session_bounds(day, _LONDON, 8, 17)
            n0, n1 = _session_bounds(day, _NY, 8, 17)
            hit_l = hit_l or _overlaps(open_ms, close_ms, l0, l1)
            hit_n = hit_n or _overlaps(open_ms, close_ms, n0, n1)
        tokyo.append(int(hit_t))
        london.append(int(hit_l))
        ny.append(int(hit_n))
    overlap = [int(a and b) for a, b in zip(london, ny)]
    return {
        "is_tokyo": tokyo,
        "is_london": london,
        "is_newyork": ny,
        "is_ldn_ny_overlap": overlap,
    }


def _weekend_gap_pips(ts_open: list[int], ts_close: list[int], open_: list[float], close: list[float], pip: float) -> list[float | None]:
    """Price gap, in pips, on the first bar after a hole that contains a Saturday.

    Knowable at that bar. Other bars, including ordinary missing minutes, stay null.
    """
    out: list[float | None] = [None] * len(ts_open)
    for i in range(1, len(ts_open)):
        if ts_open[i] - ts_close[i - 1] < 24 * 3_600_000:
            continue
        t = ts_close[i - 1]
        saturday = False
        while t < ts_open[i]:
            if datetime.fromtimestamp(t / 1000.0, tz=timezone.utc).weekday() == 5:
                saturday = True
                break
            t += 6 * 3_600_000
        if saturday and open_[i] is not None and close[i - 1] is not None:
            out[i] = (open_[i] - close[i - 1]) / pip
    return out


def _pos_log_ratio(num: str, den: str) -> pl.Expr:
    n = pl.col(num)
    d = pl.col(den)
    return pl.when((n > 0) & (d > 0)).then((n / d).log()).otherwise(None)


def _rolling_sqrt_mean(expr: pl.Expr, window: int) -> pl.Expr:
    mu = expr.rolling_mean(window_size=window, min_samples=window)
    return pl.when(mu.is_null() | (mu < 0)).then(None).otherwise(mu.sqrt())


def compute_fx_features(bars: pl.DataFrame, pip: float) -> pl.DataFrame:
    """Causal FX features. ``usd_strength_ret`` is filled by ``add_usd_strength``."""
    if bars.is_empty():
        return bars
    df = bars.sort("ts_ms")
    df = df.with_columns([
        pl.when((pl.col("close") > 0) & (pl.col("close").shift(1) > 0))
        .then((pl.col("close") / pl.col("close").shift(1)).log())
        .otherwise(None)
        .alias("log_ret_1"),
        *[(pl.col("close") / pl.col("close").shift(k) - 1.0).alias(f"ret_{k}") for k in (1, 3, 5, 10, 20, 50)],
    ])
    hl = _pos_log_ratio("high", "low")
    co = _pos_log_ratio("close", "open")
    prev_c = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_c).abs(),
        (pl.col("low") - prev_c).abs(),
    )
    df = df.with_columns([
        *[pl.col("ret_1").rolling_std(window_size=n, min_samples=n).alias(f"rv_{n}") for n in (10, 20, 50)],
        *[
            _rolling_sqrt_mean(hl.pow(2) / (4.0 * _LN2), n).alias(f"parkinson_{n}")
            for n in (10, 20, 50)
        ],
        _rolling_sqrt_mean(
            0.5 * hl.pow(2) - (2.0 * _LN2 - 1.0) * co.pow(2), 20,
        ).alias("garman_klass_20"),
        *[
            (tr.rolling_mean(window_size=n, min_samples=n) / pip).alias(f"atr_pips_{n}")
            for n in (10, 20, 50)
        ],
        (pl.col("spread_mean") / pip).alias("spread_pips"),
        pl.when(pl.col("close") > 0)
        .then(pl.col("spread_mean") / pl.col("close") * 1e4)
        .otherwise(None)
        .alias("rel_spread_bps"),
        rolling_z("spread_mean", 100).alias("spread_z_100"),
        rolling_z("n_trades", 100).alias("tick_intensity_z_100"),
    ])
    for n in (20, 50):
        change = (pl.col("close") - pl.col("close").shift(n)).abs()
        path = pl.col("close").diff().abs().rolling_sum(window_size=n, min_samples=n)
        df = df.with_columns(
            pl.when(path.is_null() | (path <= 0) | change.is_null()).then(None).otherwise(change / path).alias(f"efficiency_ratio_{n}")
        )
    flags = _session_flags(df["ts_open_ms"].to_list(), df["ts_ms"].to_list())
    gap = _weekend_gap_pips(
        df["ts_open_ms"].to_list(),
        df["ts_ms"].to_list(),
        df["open"].to_list(),
        df["close"].to_list(),
        pip,
    )
    df = df.with_columns([
        pl.Series(name, values) for name, values in flags.items()
    ]).with_columns([
        pl.Series("weekend_gap_pips", gap),
        pl.lit(None, dtype=pl.Float64).alias("usd_strength_ret"),
    ])
    return df


def add_usd_strength(frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Mean of USD-leg returns across the pairs present at each timestamp.

    USD as the base (USDJPY) contributes ``+ret``. USD as the quote (EURUSD)
    contributes ``-ret``. One pair is not an index: the column stays null.
    A cross with no USD leg is left null.
    """
    signed = []
    for pair, df in frames.items():
        if df.is_empty() or "ret_1" not in df.columns:
            continue
        if pair.startswith("USD"):
            sign = 1.0
        elif pair.endswith("USD"):
            sign = -1.0
        else:
            continue
        signed.append(df.select(["ts_ms", (pl.col("ret_1") * sign).alias(pair)]))
    if len(signed) < 2:
        return frames
    wide = signed[0]
    for part in signed[1:]:
        wide = wide.join(part, on="ts_ms", how="full", coalesce=True)
    names = [c for c in wide.columns if c != "ts_ms"]
    wide = wide.with_columns(
        pl.mean_horizontal([pl.col(c) for c in names]).alias("usd_strength_ret")
    ).select(["ts_ms", "usd_strength_ret"])
    out = {}
    for pair, df in frames.items():
        if df.is_empty():
            out[pair] = df
            continue
        joined = df.drop("usd_strength_ret") if "usd_strength_ret" in df.columns else df
        out[pair] = joined.join(wide, on="ts_ms", how="left")
    return out
