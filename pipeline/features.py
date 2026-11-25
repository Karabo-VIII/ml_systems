"""Causal rolling helpers. Warmup stays null. Nothing is filled with 0."""

from __future__ import annotations

import math
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl


# A rolling standard deviation at or below this fraction of the window's scale
# (|mean| + std) is a flat window, not a measurement.
SD_FLOOR_REL = 1e-10
# Permutation entropy ranks returns rounded to this many decimals (far below one tick).
PERM_DECIMALS = 12
# A z-score needs this share of its window to be defined values (4/5, kept in
# integers: 0.8 * 45 is 36.000000000000004 in floating point and would round up).
Z_MIN_NUM, Z_MIN_DEN = 4, 5


def z_min_samples(window: int) -> int:
    """Defined values a ``window``-row z-score needs: ceil(4/5 of the window), at least 2."""
    return max(2, -(-Z_MIN_NUM * int(window) // Z_MIN_DEN))


def rolling_z(col: str, window: int) -> pl.Expr:
    """Trailing z-score ``(x - mean) / std``, clipped to +-5.

    Mean and sample std are taken over the defined values among the last
    ``window`` rows, and at least 4/5 of them must be defined. An
    occasional undefined input (the efficiency ratio of a flat path) therefore
    blanks only its own row, not the ``window`` rows after it. No additive
    epsilon (2.4.x divided by ``std + 1e-5``, which shrank the z-scores of
    series whose std is of that order). Null where ``x`` is null and when the
    window is flat.
    """
    x = pl.col(col).cast(pl.Float64).fill_nan(None)
    need = z_min_samples(window)
    mu = x.rolling_mean(window_size=window, min_samples=need)
    sd = x.rolling_std(window_size=window, min_samples=need)
    flat = (sd.is_null() | sd.is_nan() | (sd <= SD_FLOOR_REL * (mu.abs() + sd))).fill_null(True)
    z = ((x - mu) / sd).clip(-5.0, 5.0)
    return pl.when(flat | x.is_null()).then(None).otherwise(z)


def ewm_finite(values: np.ndarray, span: float, length: int) -> np.ndarray:
    """Exponentially weighted mean over exactly the last ``length`` values.

    Weights ``(1 - alpha)^k`` for ``k = 0`` (newest) to ``length - 1``, normalised
    to sum to one, ``alpha = 2 / (span + 1)``. NaN until ``length`` values exist
    and wherever the window holds a NaN. A recursive EMA has infinite memory, so
    its value depended on where a partition's warm-up frame began (at 1d by up to
    1.7 z); this one depends on the last ``length`` rows only.
    """
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    out = np.full(n, np.nan)
    length = int(length)
    if length < 1 or n < length:
        return out
    alpha = 2.0 / (float(span) + 1.0)
    w = (1.0 - alpha) ** np.arange(length, dtype=np.float64)  # newest first
    w /= w.sum()
    # np.convolve(x, w, "valid")[j] = sum_k w[k] * x[j + length - 1 - k]
    out[length - 1:] = np.convolve(x, w, mode="valid")
    return out


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den`` where the denominator is positive, NaN elsewhere."""
    out = np.full(len(num), np.nan)
    ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out


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
    """Rescaled-range Hurst on the ``window`` log returns before ``i`` (excludes ``i``).

    Null during warmup and on a flat window, where R/S is undefined (2.4.x wrote 0.5).
    """
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
        mean = chunk.mean()
        z = np.cumsum(chunk - mean)
        r = float(np.max(z) - np.min(z))
        s = float(np.std(chunk))
        if s > SD_FLOOR_REL * (abs(mean) + s) and r > 0:
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
    # Rounded so that returns equal up to float error tie exactly: otherwise the
    # pattern of two equal returns (common on a tick grid) is decided by rounding
    # noise in how the return was computed.
    values = np.round(np.asarray(returns, dtype=np.float64), PERM_DECIMALS)
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
    """Kyle's lambda: the OLS slope of log return on signed USD volume (taker buy
    minus taker sell notional) over the ``window`` bars before ``t`` (excludes ``t``).

    Price impact per USD of net aggressive flow. 2.4.x returned the correlation,
    which drops the scale ratio sd(return) / sd(flow) that makes it an impact.
    Null during warmup and when the flow does not vary over the window.
    """
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
        if not (np.isfinite(lr).all() and np.isfinite(sv).all()):
            continue
        x = sv - sv.mean()
        y = lr - lr.mean()
        sxx = float(np.dot(x, x))
        if sxx > (SD_FLOOR_REL * (abs(float(sv.mean())) + math.sqrt(sxx / window))) ** 2 * window:
            out[t] = float(np.dot(x, y)) / sxx
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
    "norm_tick_count",
    "norm_log_volume",
    "norm_rs_volatility",
    "hurst_regime",
    "norm_return_1",
    "norm_return_4",
    "norm_return_16",
    "norm_ma_distance",
    "norm_whale",
    "norm_efficiency",
    "norm_return_kurtosis",
    "norm_hawkes_intensity",
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
    """Trailing Pearson correlation over the last ``window`` rows.

    Both series are centred on the same window's own means (2.4.x centred each
    product on per-row trailing means, which is not a correlation). Null until
    the window is full, when it holds a null, or when either series is flat.
    """
    xs = pl.col(x).cast(pl.Float64).fill_nan(None)
    ys = pl.col(y).cast(pl.Float64).fill_nan(None)
    r = pl.rolling_corr(xs, ys, window_size=window, min_samples=window)
    flat = pl.lit(False)
    for s in (xs, ys):
        mu = s.rolling_mean(window_size=window, min_samples=window)
        sd = s.rolling_std(window_size=window, min_samples=window)
        flat = flat | sd.is_null() | (sd <= SD_FLOOR_REL * (mu.abs() + sd))
    bad = (flat | r.is_null() | r.is_nan()).fill_null(True)
    return pl.when(bad).then(None).otherwise(r.clip(-1.0, 1.0)).alias(out)


def compute_physics(
    df: pl.DataFrame,
    windows: Windows,
    *,
    include_oi: bool = False,
    include_bar_duration: bool | None = None,
) -> pl.DataFrame:
    """On-bar physics. Every input is a trailing window (or a chain of two) that
    ``windows.input_window()`` accounts for, then a ``z``-bar z-score."""
    if df.is_empty():
        return df
    bar_type = df["bar_type"][0] if "bar_type" in df.columns else ""
    if include_bar_duration is None:
        include_bar_duration = not str(bar_type).startswith("time_")

    df = df.sort("ts_ms")
    if "funding_rate" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("funding_rate"))

    w = windows
    close_np = df["close"].to_numpy().astype(np.float64)
    # MACD-style deviation from finite-memory EMAs (spans 12 and 26).
    ema_fast = ewm_finite(close_np, 12, w.ema)
    ema_slow = ewm_finite(close_np, 26, w.ema)
    vol = pl.col("volume")
    df = df.with_columns([
        pl.col("close").pct_change().alias("_ret"),
        pl.when(vol > 0).then((pl.col("buy_volume") - pl.col("sell_volume")) / vol).otherwise(None).alias("_flow"),
        pl.when(vol > 0).then((pl.col("buy_volume") - pl.col("sell_volume")).abs() / vol).otherwise(None).alias("_vpin"),
        to_series("_dev", _ratio(ema_fast - ema_slow, ema_slow)),
        pl.when(pl.col("n_trades") > 0).then(pl.col("volume") / pl.col("n_trades")).otherwise(None).alias("_whale"),
        pl.when(vol > 0).then(vol.log()).otherwise(None).alias("_log_vol"),
        pl.when(pl.col("n_trades") > 0).then(pl.col("n_trades").cast(pl.Float64)).otherwise(None).alias("_ticks"),
    ])
    # Rogers-Satchell per-bar variance (>= 0 for valid OHLC; a float-error negative is 0).
    rs_var = (
        _log_ratio("high", "close") * _log_ratio("high", "open")
        + _log_ratio("low", "close") * _log_ratio("low", "open")
    )
    df = df.with_columns(rs_var.alias("_rs_var"))
    df = df.with_columns(
        pl.when(pl.col("_rs_var").is_null()).then(None)
        .when(pl.col("_rs_var") < 0).then(pl.lit(0.0))
        .otherwise(pl.col("_rs_var"))
        .alias("_rs_var")
    )
    df = df.with_columns(pl.col("_rs_var").sqrt().alias("_rs"))
    path = (pl.col("close") - pl.col("close").shift(1)).abs().rolling_sum(window_size=w.efficiency, min_samples=w.efficiency)
    df = df.with_columns([
        pl.col("_dev").rolling_std(window_size=w.vol_cluster, min_samples=w.vol_cluster).alias("_vol_cluster"),
        (pl.col("close").shift(1) - pl.col("close").shift(1 + 4))
        .truediv(pl.col("close").shift(1 + 4))
        .alias("_ret4"),
        (pl.col("close").shift(1) - pl.col("close").shift(1 + 16))
        .truediv(pl.col("close").shift(1 + 16))
        .alias("_ret16"),
        pl.col("_ret").shift(1).alias("_ret1"),
        # Kaufman efficiency: net move over path length. Undefined on a flat path.
        pl.when(path > 0)
        .then((pl.col("close") - pl.col("close").shift(w.efficiency)).abs() / path)
        .otherwise(None)
        .alias("_eff"),
        (
            (pl.col("close") - pl.col("close").rolling_mean(window_size=w.ma, min_samples=w.ma))
            / pl.col("close").rolling_mean(window_size=w.ma, min_samples=w.ma)
        ).alias("_ma_dist"),
    ])
    # Duration and Hawkes. Duration is almost constant on time bars, so the
    # feature is omitted there; the rate-versus-EMA Hawkes signal is not. The
    # rate is trades (or volume) per second since the previous bar's close.
    dur_s = pl.col("ts_ms").diff() / 1000.0
    df = df.with_columns(dur_s.alias("_dur_s"))
    df = df.with_columns([
        pl.when(pl.col("_dur_s") > 0).then(pl.col("_dur_s").log()).otherwise(None).alias("_bar_dur"),
        pl.when(pl.col("_dur_s") > 0).then(pl.col("n_trades") / pl.col("_dur_s")).otherwise(None).alias("_tick_rate"),
        pl.when(pl.col("_dur_s") > 0).then(pl.col("buy_volume") / pl.col("_dur_s")).otherwise(None).alias("_buy_rate"),
        pl.when(pl.col("_dur_s") > 0).then(pl.col("sell_volume") / pl.col("_dur_s")).otherwise(None).alias("_sell_rate"),
    ])
    surge = {}
    for name, col in (("_h_all", "_tick_rate"), ("_h_buy", "_buy_rate"), ("_h_sell", "_sell_rate")):
        rate = df[col].to_numpy().astype(np.float64)
        surge[name] = _ratio(rate, ewm_finite(rate, w.hawkes_span, w.ema)) - 1.0
    df = df.with_columns([
        to_series("_h_all", surge["_h_all"]),
        to_series("_h_imb", surge["_h_buy"] - surge["_h_sell"]),
        (pl.col("_ret4") - pl.col("_ret4").shift(4)).alias("_mom_acc"),
    ])

    # Excess kurtosis of returns over one window of k bars (population moments,
    # all centred on that window's mean). Null while the window is incomplete.
    k = w.kurtosis
    df = df.with_columns(
        pl.col("_ret").rolling_kurtosis(window_size=k, fisher=True, bias=True, min_samples=k).alias("_kurt")
    )

    # Vol term structure on Rogers-Satchell: fast (span 10) over slow (slow_span) EMA, minus one.
    rs_np = df["_rs"].to_numpy().astype(np.float64)
    vol_ratio = _ratio(ewm_finite(rs_np, 10, w.ema), ewm_finite(rs_np, w.slow_span, w.ema)) - 1.0
    df = df.with_columns([
        to_series("_vol_ratio", vol_ratio),
        pl.col("_ret").abs().alias("_abs_ret"),
        pl.col("_flow").shift(1).alias("_flow_lag"),
    ])
    df = df.with_columns([
        _corr_roll("_log_vol", "_abs_ret", w.fast, "_vp_corr"),
        _corr_roll("_flow", "_flow_lag", w.fast, "_flow_pers"),
    ])

    # Yang-Zhang (2000): sample variances of the open jump and the open-to-close
    # return plus the Rogers-Satchell mean, k = 0.34 / (1.34 + (n + 1) / (n - 1)).
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
            pl.col("_yz_on").rolling_var(window_size=n_yz, min_samples=n_yz)
            + yz_k * pl.col("_yz_oc").rolling_var(window_size=n_yz, min_samples=n_yz)
            + (1.0 - yz_k) * pl.col("_rs_var").rolling_mean(window_size=n_yz, min_samples=n_yz)
        ).clip(lower_bound=0.0).sqrt().alias("_yz")
    )

    # Corwin-Schultz (2012) spread from two adjacent blocks of b bars:
    # beta = ln(H1/L1)^2 + ln(H2/L2)^2 (the sum over both blocks), gamma = ln(H12/L12)^2,
    # alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2)),
    # S = 2 (e^alpha - 1) / (1 + e^alpha). A negative S is kept (no floor at 0).
    # A block without range is a defined input (ln(H/L) = 0; a flat 2b-bar
    # stretch gives S = 0); only a non-positive price is undefined.
    b = w.cs_block
    h_block = pl.col("high").rolling_max(window_size=b, min_samples=b)
    l_block = pl.col("low").rolling_min(window_size=b, min_samples=b)
    h_prev = pl.col("high").shift(b).rolling_max(window_size=b, min_samples=b)
    l_prev = pl.col("low").shift(b).rolling_min(window_size=b, min_samples=b)
    beta = (
        pl.when((l_block > 0) & (l_prev > 0))
        .then((h_block / l_block).log().pow(2) + (h_prev / l_prev).log().pow(2))
        .otherwise(None)
    )
    h2 = pl.col("high").rolling_max(window_size=2 * b, min_samples=2 * b)
    l2 = pl.col("low").rolling_min(window_size=2 * b, min_samples=2 * b)
    gamma = pl.when(l2 > 0).then((h2 / l2).log().pow(2)).otherwise(None)
    denom = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = ((2.0 * beta).sqrt() - beta.sqrt()) / denom - (gamma / denom).sqrt()
    df = df.with_columns((2.0 * (alpha.exp() - 1.0) / (1.0 + alpha.exp())).alias("_cs"))

    # Numpy features on the close path. They see only the past.
    log_close = np.where(close_np > 0, np.log(np.where(close_np > 0, close_np, 1.0)), np.nan)
    log_ret = np.full(len(close_np), np.nan)
    log_ret[1:] = np.diff(log_close)
    fd = frac_diff(log_close, d=w.frac_diff_d, window=w.frac_diff)
    hurst = hurst_rs(log_ret, window=w.hurst)
    ret_np = df["_ret"].to_numpy().astype(np.float64)
    pent = permutation_entropy(ret_np, window=w.perm_entropy, m=w.perm_m)
    if "signed_usd" in df.columns:
        signed = df["signed_usd"].to_numpy().astype(np.float64)
    else:
        signed = np.full(len(close_np), np.nan)
    kyle = kyle_lambda(close_np, signed, window=w.kyle)
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
        "norm_tick_count": ("_ticks", w.z),
        "norm_log_volume": ("_log_vol", w.z),
        "norm_rs_volatility": ("_rs", w.z),
        "hurst_regime": ("_hurst", w.z),
        "norm_return_1": ("_ret1", w.z),
        "norm_return_4": ("_ret4", w.z),
        "norm_return_16": ("_ret16", w.z),
        "norm_ma_distance": ("_ma_dist", w.z),
        "norm_whale": ("_whale", w.z),
        "norm_efficiency": ("_eff", w.z),
        "norm_return_kurtosis": ("_kurt", w.z),
        "norm_hawkes_intensity": ("_h_all", w.z),
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
        m = w.oi_median
        med = pl.col("oi_value").rolling_median(window_size=m, min_samples=min(20, m))
        capped = pl.when(pl.col("oi_value").is_not_null() & med.is_not_null() & (pl.col("oi_value") > 5.0 * med)).then(5.0 * med).otherwise(pl.col("oi_value"))
        df = df.with_columns(capped.alias("_oi_cap"))
        df = df.with_columns(pl.col("_oi_cap").pct_change().alias("_oi_chg"))
        df = df.with_columns(rolling_z("_oi_chg", w.z).alias("norm_oi_change"))
        oi_abs = df["_oi_chg"].abs().to_numpy().astype(np.float64)
        ret_abs = df["_abs_ret"].to_numpy().astype(np.float64)
        oi_div = _ratio(ewm_finite(oi_abs, 10, w.ema), ewm_finite(ret_abs, 10, w.ema)) - 1.0
        df = df.with_columns(to_series("_oi_div", oi_div))
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
# Bins cover [$1, $100M] in log10 space (64 per decade, 3.7% wide) plus
# under/overflow; the quantile is interpolated inside its bin.
WHALE_LOG_LO = 0.0
WHALE_LOG_HI = 8.0
WHALE_BINS_PER_DECADE = 64
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
    """``q``-quantile of the pooled histograms, in USD.

    The bin that holds the ``q * total``-th trade is found from the cumulative
    counts, and the value is interpolated log-linearly inside that bin (trades
    spread evenly in log size), so the error is a fraction of one 3.7% bin.
    """
    if not hists:
        return None
    hist = np.sum(np.stack(hists, axis=0), axis=0).astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return None
    target = float(q) * total
    cum = np.cumsum(hist)
    i = int(np.searchsorted(cum, target, side="left"))
    if i <= 0:
        return 10.0 ** WHALE_LOG_LO
    if i >= WHALE_N_BINS + 1:
        return 10.0 ** WHALE_LOG_HI
    width = (WHALE_LOG_HI - WHALE_LOG_LO) / WHALE_N_BINS
    below = float(cum[i - 1])
    frac = min(1.0, max(0.0, (target - below) / hist[i]))
    lo = WHALE_LOG_LO + (i - 1) * width
    return float(10.0 ** (lo + frac * width))


def whale_threshold(history: list[np.ndarray], spec) -> float | None:
    """Size for a new UTC day from complete days already in ``history``. Causal.

    ``None`` with no history (an asset's first day): there is no size, and the
    day's ``wh_*`` columns are null rather than measured against a static table.
    """
    usable = [h for h in history if h is not None and int(np.asarray(h).sum()) > 0]
    if not usable:
        return None
    value = quantile_from_hists(usable[-int(spec.window_days):], float(spec.quantile))
    return None if value is None else float(value)


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


WH_COLUMNS = (
    "wh_large_count", "wh_large_usd", "wh_large_buy_usd", "wh_large_sell_usd",
    "wh_max_print_usd", "wh_threshold_usd",
)


def summarize_day(
    trades: pl.DataFrame,
    bars_1m: pl.DataFrame,
    *,
    asset: str,
    day: date,
    whale_usd: float | None,
) -> dict:
    """One row of whale, realized-vol, and Hawkes-branching features for UTC day D.

    ``whale_usd`` is the day's causal whale size; ``None`` (no history yet) leaves
    every ``wh_*`` column null. Counts and sums of large prints are real zeros
    when the day had trades but none that large.
    """
    event = day_start_ms(day)
    row: dict = {
        "asset": asset,
        "event_time_ms": event,
        "wh_large_count": None,
        "wh_large_usd": None,
        "wh_large_buy_usd": None,
        "wh_large_sell_usd": None,
        "wh_max_print_usd": None,
        "wh_threshold_usd": None if whale_usd is None else float(whale_usd),
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
        row["wh_max_print_usd"] = float(usd.max())
        if whale_usd is not None:
            large = usd >= whale_usd
            row["wh_large_count"] = int(large.sum())
            row["wh_large_usd"] = float(usd[large].sum()) if large.any() else 0.0
            row["wh_large_buy_usd"] = float(usd[large & ~maker].sum()) if large.any() else 0.0
            row["wh_large_sell_usd"] = float(usd[large & maker].sum()) if large.any() else 0.0
        row["hbr_n_trades"] = int(t.height)
        ts = t["ts_ms"].to_numpy()
        row["hbr_eta"] = _eta(_minute_counts(ts, np.ones(len(ts), dtype=bool), event))
        row["hbr_eta_buy"] = _eta(_minute_counts(ts, ~maker, event))
        row["hbr_eta_sell"] = _eta(_minute_counts(ts, maker, event))
        if row["hbr_eta_buy"] == row["hbr_eta_buy"] and row["hbr_eta_sell"] == row["hbr_eta_sell"]:
            row["hbr_eta_imbalance"] = row["hbr_eta_buy"] - row["hbr_eta_sell"]

    row.update(realized_day(bars_1m))
    # NaN -> None for parquet nulls
    for k, v in list(row.items()):
        if isinstance(v, float) and not math.isfinite(v):
            row[k] = None
    return row


RV_COLUMNS = ("rv_realized", "rv_bpv", "rv_jv", "rv_jump_frac")


def realized_day(bars_1m: pl.DataFrame | None) -> dict:
    """Realized variance of one UTC day from its 1m closes (log returns between
    consecutive bars): ``rv_realized`` = sum r^2, bipower ``rv_bpv`` =
    pi/2 sum |r_t||r_t-1|, jump ``rv_jv`` = max(rv - bpv, 0), ``rv_jump_frac``.
    Null with fewer than three bars. Shared by crypto (trade closes) and FX (mids)."""
    out: dict = {c: None for c in RV_COLUMNS}
    if bars_1m is None or bars_1m.height < 3:
        return out
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
        out.update({"rv_realized": rv, "rv_bpv": bpv, "rv_jv": jv,
                    "rv_jump_frac": (jv / rv) if rv > 0 else None})
    return out


FX_DAILY_SCHEMA: dict[str, pl.DataType] = {
    "asset": pl.Utf8,
    "event_time_ms": pl.Int64,
    **{c: pl.Float64 for c in RV_COLUMNS},
    "n_bars": pl.Int64,
}


def fx_daily_panel(bars_1m: pl.DataFrame, asset: str) -> pl.DataFrame:
    """Per UTC day (by bar open) realized variance of the 1m mids, plus the bar
    count, since an FX day can be partial (the Sunday reopen, the Friday close)."""
    if bars_1m is None or bars_1m.is_empty():
        return pl.DataFrame(schema=FX_DAILY_SCHEMA)
    rows = []
    keyed = bars_1m.sort("ts_ms").with_columns((pl.col("ts_open_ms") // DAY_MS).alias("_day"))
    for (day,), part in keyed.group_by("_day", maintain_order=True):
        rows.append({"asset": asset, "event_time_ms": int(day) * DAY_MS, **realized_day(part), "n_bars": part.height})
    return pl.DataFrame(rows, schema=FX_DAILY_SCHEMA).sort("event_time_ms")


DAILY_PANEL_SCHEMA: dict[str, pl.DataType] = {
    "asset": pl.Utf8,
    "event_time_ms": pl.Int64,
    "wh_large_count": pl.Int64,
    "wh_large_usd": pl.Float64,
    "wh_large_buy_usd": pl.Float64,
    "wh_large_sell_usd": pl.Float64,
    "wh_max_print_usd": pl.Float64,
    "wh_threshold_usd": pl.Float64,
    "rv_realized": pl.Float64,
    "rv_bpv": pl.Float64,
    "rv_jv": pl.Float64,
    "rv_jump_frac": pl.Float64,
    "hbr_eta": pl.Float64,
    "hbr_eta_buy": pl.Float64,
    "hbr_eta_sell": pl.Float64,
    "hbr_eta_imbalance": pl.Float64,
    "hbr_n_trades": pl.Int64,
}


def rows_to_frame(rows: list[dict]) -> pl.DataFrame:
    """The daily panel with fixed dtypes, so a month whose only day has no whale
    size still writes typed (all-null) columns. A key outside the schema is an
    error: polars would drop it silently."""
    if not rows:
        return pl.DataFrame(schema={"asset": pl.Utf8, "event_time_ms": pl.Int64})
    unknown = sorted({k for row in rows for k in row} - set(DAILY_PANEL_SCHEMA))
    if unknown:
        raise ValueError(f"daily panel columns {unknown} are not in DAILY_PANEL_SCHEMA")
    return pl.DataFrame(rows, schema=DAILY_PANEL_SCHEMA).sort(["asset", "event_time_ms"])


def funding_daily(funding: pl.DataFrame, asset: str) -> pl.DataFrame:
    if funding is None or funding.is_empty():
        return pl.DataFrame(schema={
            "asset": pl.Utf8, "event_time_ms": pl.Int64,
            "funding_day_mean": pl.Float64, "funding_day_sum": pl.Float64,
            "funding_day_last": pl.Float64,
        })
    # Settlement count is usually 3 and would be a constant feature. Keep the
    # rate statistics, which are the actual signal.
    days, sums, counts, last = _per_day(funding, ["funding_rate"])
    s, n = sums["funding_rate"], counts["funding_rate"]
    return pl.DataFrame({
        "event_time_ms": days,
        "funding_day_mean": np.where(n > 0, s / np.maximum(n, 1), np.nan),
        "funding_day_sum": np.where(n > 0, s, np.nan),
        "funding_day_last": last["funding_rate"],
    }).with_columns([pl.col(c).fill_nan(None) for c in ("funding_day_mean", "funding_day_sum", "funding_day_last")]
                    ).with_columns(pl.lit(asset).alias("asset")).sort("event_time_ms")


def _per_day(df: pl.DataFrame, cols: list[str]):
    """Per-UTC-day sums, counts of defined values, and last defined values.

    The sums are correctly rounded (``math.fsum``), so they do not depend on row
    order or on the frame's layout. Polars' group-by can sum the same group along a
    different path from one run to the next (seen on the s3_* means, 1e-16 apart),
    which breaks the byte-identical rebuild the contract promises. The last value
    is the last defined one in time order; equal timestamps keep the input order,
    which the readers fix with a stable sort.
    """
    df = df.with_columns(((pl.col("ts_ms") // DAY_MS_I) * DAY_MS_I).alias("_day")).sort(
        ["_day", "ts_ms"], maintain_order=True)
    keys = df["_day"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(keys)]
    sums, counts, last = {}, {}, {}
    for c in cols:
        v = df[c].cast(pl.Float64).to_numpy()
        ok = np.isfinite(v)
        s = np.empty(len(starts))
        lv = np.full(len(starts), np.nan)
        for k, (a, b) in enumerate(zip(starts, ends)):
            s[k] = math.fsum(v[a:b][ok[a:b]])
            idx = np.flatnonzero(ok[a:b])
            if idx.size:
                lv[k] = v[a + idx[-1]]
        sums[c] = s
        counts[c] = np.add.reduceat(ok.astype(np.int64), starts) if len(starts) else np.zeros(0, np.int64)
        last[c] = lv
    return keys[starts], sums, counts, last


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
    days, sums, counts, _last = _per_day(metrics, numeric)
    out = pl.DataFrame({
        "event_time_ms": days,
        **{f"s3_{c}_mean": np.where(counts[c] > 0, sums[c] / np.maximum(counts[c], 1), np.nan) for c in numeric},
    })
    out = out.with_columns([pl.col(f"s3_{c}_mean").fill_nan(None) for c in numeric])
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


def _fold_sum(exprs: list[pl.Expr]) -> pl.Expr:
    """Left-to-right sum in the order given, a null counting as 0.0.

    polars' sum_horizontal changes its rounding with the number of columns, so one
    more all-null column (a peer that lists later in the month's cohort) moved the
    cross-asset means by an ulp. A left fold adds that column as an exact 0.0.
    """
    return pl.fold(acc=pl.lit(0.0, dtype=pl.Float64), function=lambda acc, x: acc + x,
                   exprs=[e.cast(pl.Float64).fill_null(0.0) for e in exprs])


def _horizontal_mean(cols: list[str]) -> pl.Expr:
    """Mean of the defined values among ``cols``, summed in the order given."""
    if not cols:
        return pl.lit(None, dtype=pl.Float64)
    num = _fold_sum([pl.col(c) for c in cols])
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
        # Peers in name order: the means are then summed in the same order whatever
        # order the frames arrived in.
        for other in sorted(prepped):
            other_df = prepped[other]
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

        # Percentile rank in [0, 1]: the share of the other names with a lower return,
        # ties counting half (1 = best of the cohort). Dividing by the names present
        # keeps it comparable when SOL and AVAX join mid-year. Null below 3 names.
        if peer_ret:
            n_lower = pl.sum_horizontal([
                (pl.col(c) < pl.col("_ret")).fill_null(False).cast(pl.Float64) for c in peer_ret
            ])
            n_equal = pl.sum_horizontal([
                (pl.col(c) == pl.col("_ret")).fill_null(False).cast(pl.Float64) for c in peer_ret
            ])
            pct_rank = (n_lower + 0.5 * n_equal) / (n_with_self - 1).cast(pl.Float64)
        else:
            pct_rank = pl.lit(None, dtype=pl.Float64)

        df = df.with_columns([
            pl.when(enough).then(btc_ret).otherwise(None).alias("xd_btc_return"),
            pl.when(enough).then(btc_vol).otherwise(None).alias("xd_btc_volatility"),
            pl.when(enough).then(pl.col("funding_rate") - _horizontal_mean(peer_fund)).otherwise(None).alias("xd_funding_spread"),
            pl.when(enough).then(_horizontal_mean(peer_ret)).otherwise(None).alias("xd_cross_return_mean"),
            pl.when(enough).then(_horizontal_mean(peer_vol)).otherwise(None).alias("xd_cross_vol_mean"),
            pl.when(enough).then(pl.col("_ma") - _horizontal_mean(peer_ma)).otherwise(None).alias("xd_ma_distance"),
            pl.when(enough_rank & pl.col("_ret").is_not_null()).then(pct_rank).otherwise(None).alias("xd_momentum_rank"),
        ])
        drop = [c for c in df.columns if c.startswith("_pr_") or c.startswith("_pv_") or c.startswith("_pm_") or c.startswith("_pf_") or c in ("_ret", "_vol", "_ma")]
        out[asset] = df.drop(drop)
    # Assets that were empty stay as they were.
    for asset, df in frames.items():
        if asset not in out:
            out[asset] = df
    return out


# FX features. Windows are in bars of the frame they are computed on. Warmup is
# null. Nothing is filled with 0. The two z-scores use a 100-bar window, cut on
# coarse cadences so no FX lookback spans more than 90 calendar days (1d: 63).
FX_FEATURES = (
    "log_ret_1",
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_50",
    "rv_10", "rv_20", "rv_50",
    "parkinson_10", "parkinson_20", "parkinson_50",
    "garman_klass_20",
    "atr_pips_10", "atr_pips_20", "atr_pips_50",
    "spread_pips", "rel_spread_bps", "spread_z",
    "tick_intensity_z",
    "efficiency_ratio_20", "efficiency_ratio_50",
    "is_tokyo", "is_london", "is_newyork", "is_ldn_ny_overlap",
    "weekend_gap_pips",
    "base_strength_ret", "quote_strength_ret",
)

# The FX week is Sunday 17:00 to Friday 17:00 America/New_York: 120 of 168 hours.
FX_TRADING_SHARE = 5.0 / 7.0
FX_Z_WINDOW = 100
# The longest fixed reach besides the z-scores: ret_50, rv_50, atr_pips_50 and
# efficiency_ratio_50 read the bar and the 50 before it.
FX_FIXED_REACH = 51

_LN2 = math.log(2.0)
_TOKYO = ZoneInfo("Asia/Tokyo")
_LONDON = ZoneInfo("Europe/London")
_NY = ZoneInfo("America/New_York")


def fx_cap_bars(bar_type: str, max_lookback_days: int = 90) -> int:
    """FX bars that fit in ``max_lookback_days`` calendar days (trading time only)."""
    from pipeline.constants import TIME_WIDTH_MS
    return int(max_lookback_days * DAY_MS * FX_TRADING_SHARE // TIME_WIDTH_MS[bar_type])


def fx_z_window(bar_type: str, max_lookback_days: int = 90) -> int:
    """Window of ``spread_z`` and ``tick_intensity_z``: 100 bars, cut to fit the cap (1d: 63)."""
    return max(2, min(FX_Z_WINDOW, fx_cap_bars(bar_type, max_lookback_days) - 1))


def fx_lookback(bar_type: str, max_lookback_days: int = 90) -> int:
    """Rows (the bar and those before it) that the longest FX feature reads.

    The 50-bar windows are part of the feature names and are not cut: a cap too
    short to hold them on a cadence is a configuration error, not a silent overrun.
    """
    cap = fx_cap_bars(bar_type, max_lookback_days)
    if FX_FIXED_REACH > cap:
        raise ValueError(
            f"windows.max_lookback_days={max_lookback_days} holds {cap} FX bars on {bar_type}, "
            f"fewer than the {FX_FIXED_REACH} the 50-bar FX features read"
        )
    return max(FX_FIXED_REACH, fx_z_window(bar_type, max_lookback_days))


def fx_embargo_bars(bar_type: str, horizons: tuple[int, ...], *, calendar_floor_days: int = 7,
                    max_lookback_days: int = 90) -> int:
    """FX embargo in bars of this cadence: ``max(horizon, lookback)``, and at least
    ``calendar_floor_days`` of calendar time (five trading days in seven)."""
    from pipeline.constants import TIME_WIDTH_MS
    width = TIME_WIDTH_MS[bar_type]
    floor_bars = max(1, math.ceil(calendar_floor_days * DAY_MS * FX_TRADING_SHARE / width))
    return max(max(horizons), fx_lookback(bar_type, max_lookback_days), floor_bars)


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


def compute_fx_features(bars: pl.DataFrame, pip: float, *, max_lookback_days: int = 90) -> pl.DataFrame:
    """Causal FX features. The currency strengths are filled by ``add_currency_strength``."""
    if bars.is_empty():
        return bars
    df = bars.sort("ts_ms")
    bar_type = str(df["bar_type"][0]) if "bar_type" in df.columns else "time_1m"
    zw = fx_z_window(bar_type, max_lookback_days)
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
        rolling_z("spread_mean", zw).alias("spread_z"),
        rolling_z("n_trades", zw).alias("tick_intensity_z"),
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
        pl.Series("weekend_gap_pips", gap, dtype=pl.Float64),
        pl.lit(None, dtype=pl.Float64).alias("base_strength_ret"),
        pl.lit(None, dtype=pl.Float64).alias("quote_strength_ret"),
    ])
    return df


FX_CURRENCIES = ("USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF")


def add_currency_strength(frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Base and quote currency strength of every pair, from the USD pairs present.

    Each currency's log return against USD at a bar comes from its USD pair
    (``log_ret_1`` of XXXUSD, or minus that of USDXXX); USD's own is 0. With N
    currencies present at the bar, the strength of currency C is its mean log
    return against the other N - 1: ``(N * r_C - sum_X r_X) / (N - 1)``. A pair
    gets ``base_strength_ret`` and ``quote_strength_ret``, so one of them is the
    USD strength for every major. Null below three currencies (one pair is not a
    basket) and for a currency absent at the bar. Crosses add no currency.
    """
    legs = []
    for pair, df in frames.items():
        if df.is_empty() or "log_ret_1" not in df.columns or len(pair) != 6:
            continue
        base, quote = pair[:3], pair[3:]
        if quote == "USD" and base != "USD":
            ccy, sign = base, 1.0
        elif base == "USD" and quote != "USD":
            ccy, sign = quote, -1.0
        else:
            continue
        legs.append(df.select(["ts_ms", (pl.col("log_ret_1") * sign).alias(ccy)]))
    out = {}
    if not legs:
        return {p: (df if df.is_empty() else df.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("base_strength_ret"),
            pl.lit(None, dtype=pl.Float64).alias("quote_strength_ret"),
        ])) for p, df in frames.items()}
    wide = legs[0]
    for part in legs[1:]:
        wide = wide.join(part, on="ts_ms", how="full", coalesce=True)
    # Currency order is fixed, so the basket sums the same way whatever order the pairs came in.
    ccys = sorted(c for c in wide.columns if c != "ts_ms")
    # USD against itself is 0 by definition, at every bar some USD pair printed.
    wide = wide.with_columns(
        pl.when(pl.any_horizontal([pl.col(c).is_not_null() for c in ccys])).then(0.0).otherwise(None).alias("USD")
    )
    present = [*ccys, "USD"]
    n = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Float64) for c in present])
    total = _fold_sum([pl.col(c) for c in present])
    strength = wide.select([
        "ts_ms",
        *[
            pl.when(pl.col(c).is_not_null() & (n >= 3)).then((n * pl.col(c) - total) / (n - 1.0)).otherwise(None).alias(f"_str_{c}")
            for c in present
        ],
    ])
    for pair, df in frames.items():
        if df.is_empty():
            out[pair] = df
            continue
        base, quote = pair[:3], pair[3:]
        cols = [c for c in ("base_strength_ret", "quote_strength_ret") if c in df.columns]
        joined = df.drop(cols).join(strength, on="ts_ms", how="left")
        null = pl.lit(None, dtype=pl.Float64)
        joined = joined.with_columns([
            (pl.col(f"_str_{base}") if f"_str_{base}" in strength.columns else null).alias("base_strength_ret"),
            (pl.col(f"_str_{quote}") if f"_str_{quote}" in strength.columns else null).alias("quote_strength_ret"),
        ])
        out[pair] = joined.drop([c for c in joined.columns if c.startswith("_str_")])
    return out
