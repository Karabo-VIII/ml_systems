"""
V4 Feature Engineering — SOTA Microstructure Physics

Features (34 base dimensions, 30 legacy + 4 SOTA):
  -- Legacy (0-12) --
  0: norm_deviation       — Volatility regime (EMA spread)
  1: norm_fd_close        — Fractional differentiation (stationary trend memory)
  2: norm_vpin            — Volume-synchronized probability of informed trading
  3: norm_flow_imbalance  — Buy/sell volume delta
  4: norm_vol_cluster     — Volatility of volatility
  5: norm_funding         — Funding rate (positioning sentiment)
  6: norm_tick_count      — Liquidity activity proxy
  7: norm_log_volume      — Absolute volume (log-scaled)
  8: norm_hl_spread       — Rogers-Satchell realized volatility (drift-independent, ~6x CC efficiency)
  9: hurst_regime         — Mean-reversion vs trending (R/S statistic)
 10: norm_oi_change       — Open interest rate of change
 11: norm_return_1        — Lagged 1-bar return
 12: norm_spread_bps      — Effective bid-ask spread proxy
  -- Extended (13-17) --
 13: norm_ma_distance     — Distance from SMA(200), medium-term trend regime
 14: norm_whale           — Average trade size = volume/tick_count (institutional flow proxy)
 15: norm_efficiency      — Price efficiency ratio (trending vs choppy)
 16: norm_return_4        — Lagged 4-bar cumulative return (momentum/mean-reversion)
 17: norm_return_16       — Lagged 16-bar cumulative return (medium-term momentum)
  -- Tier 1 (18-20): orthogonal replacements --
 18: norm_return_kurtosis — Rolling excess kurtosis (distribution shape)
 19: norm_bar_duration    — Log time between dollar bars (activity clock speed)
 20: norm_funding_momentum— Funding rate of change (leverage dynamics)
  -- Hawkes (21-24): trade clustering dynamics --
 21: norm_hawkes_intensity     — Tick rate vs EMA (self-excitation signal)
 22: norm_hawkes_buy_intensity — Buy-side clustering (informed buying acceleration)
 23: norm_hawkes_sell_intensity— Sell-side clustering (liquidation/distribution)
 24: norm_hawkes_imbalance     — Buy - sell clustering (directional clustering)
  -- Tier 2 (25-29): IC-boosting dynamics --
 25: norm_momentum_accel       — Second derivative of price (trend acceleration vs fading)
 26: norm_vol_price_corr       — Volume-price correlation (accumulation/distribution)
 27: norm_vol_ratio            — Volatility term structure (short/long vol ratio)
 28: norm_flow_persistence     — Flow autocorrelation (institutional campaign detection)
 29: norm_oi_price_divergence  — OI building while price flat (spring loading)
  -- SOTA (30-33) — institutional-grade additions --
 30: norm_yz_volatility       — Yang-Zhang volatility (MVUE, adds overnight-jump to RS)
 31: norm_cs_spread           — Corwin-Schultz spread (principled H/L bid-ask estimator)
 32: norm_perm_entropy        — Permutation entropy (complexity/predictability measure)
 33: norm_kyle_lambda         — Kyle's lambda (price impact per dollar of order flow)

  SOTA equivalences (old features KEPT, new features ADDED):
    norm_yz_volatility  upgrades  norm_hl_spread (feature 8) — adds overnight correction
    norm_cs_spread      upgrades  norm_spread_bps (feature 12) — principled vs HL-range proxy
    norm_perm_entropy   NEW — no equivalent in existing features
    norm_kyle_lambda    NEW — no equivalent in existing features

Labels (non-feature):
  regime_label            — SMA(200) regime classification (0=bear, 1=neutral, 2=bull)

Targets (multi-horizon):
  target_return_1   — Next-bar return (raw)
  target_return_4   — 4-bar cumulative return (raw)
  target_return_16  — 16-bar cumulative return (raw)
  target_return_64  — 64-bar cumulative return (raw)
  target_voladj_1   — Vol-normalized 1-bar return (symlog, for TwoHot training)
  target_voladj_4   — Vol-normalized 4-bar return (symlog)
  target_voladj_16  — Vol-normalized 16-bar return (symlog)
  target_voladj_64  — Vol-normalized 64-bar return (symlog)
  target_return_50  — 50-bar risk-adjusted return (for agent reward)
  target_vol_20     — 20-bar forward volatility

Changes from V303:
  - FracDiff uses rolling normalization (no look-ahead bias)
  - OI data integration (was fetched but unused)
  - Lagged return as explicit feature (autoregressive signal)
  - Effective spread proxy from aggTrade data
  - Multi-horizon targets for world model regularization

V51 Upgrades (2026-03-08):
  - Rogers-Satchell volatility replaces simple (H-L)/O in norm_hl_spread
  - Multi-horizon lagged returns: norm_return_4 (4-bar), norm_return_16 (16-bar)
  - Vol-normalized targets: target_voladj_{1,4,16,64} for TwoHot training
  - Restored dropped features: norm_whale (trade size), norm_efficiency (price efficiency)
"""

import math
import polars as pl
import numpy as np
from numba import njit
import warnings

# ── Configuration ──────────────────────────────────────────────────────────────

WINDOW_FAST = 20
WINDOW_ADAPTIVE = 200
WINDOW_REGIME = 200


# ── Numba Engine (Core Compute) ───────────────────────────────────────────────

@njit(cache=True)
def get_weights_frac_diff(d, size):
    w = np.empty(size)
    w[0] = 1.0
    for k in range(1, size):
        w[k] = -w[k-1] / k * (d - k + 1)
    return np.ascontiguousarray(w[::-1])


@njit(cache=True)
def frac_diff_fast(series, d=0.4, window=1000):
    T = len(series)
    out = np.full(T, np.nan)
    if T <= window:
        return out
    weights = get_weights_frac_diff(d, window)
    for i in range(window, T):
        window_data = series[i-window:i]
        if np.isnan(window_data).any():
            continue
        out[i] = np.dot(weights, window_data)
    return out


@njit(cache=True)
def get_rs_hurst_rolling(series, window=200):
    T = len(series)
    out = np.full(T, 0.5)
    if T <= window:
        return out
    for i in range(window, T):
        chunk = series[i-window:i]
        if np.max(chunk) == np.min(chunk):
            out[i] = 0.5
            continue
        mean = np.mean(chunk)
        z = np.cumsum(chunk - mean)
        R = np.max(z) - np.min(z)
        S = np.std(chunk)
        if S < 1e-9:
            out[i] = 0.5
        else:
            out[i] = np.log(R / S) / np.log(window)
    return out


@njit(cache=True)
def _permutation_entropy_rolling(returns, m=3, window=100):
    """Rolling permutation entropy of return series.

    For each bar, computes the Shannon entropy of ordinal patterns
    in the preceding `window` returns. Embedding dimension m=3
    gives 3!=6 possible patterns.

    Returns array of PE values normalized to [0, 1].
    """
    n = len(returns)
    out = np.full(n, 0.5)  # Default = maximum entropy (uninformative)
    n_patterns = 1
    for i in range(1, m + 1):
        n_patterns *= i  # m!

    log_mfact = np.log(n_patterns)
    if log_mfact < 1e-12:
        return out

    for t in range(window + m, n):
        # Count occurrences of each ordinal pattern in window
        counts = np.zeros(n_patterns, dtype=np.int64)
        for i in range(t - window, t - m + 1):
            # Get the rank pattern of returns[i:i+m]
            seg = returns[i:i + m]
            # Convert rank pattern to index (0 to m!-1)
            # For m=3: 6 patterns. Use insertion-sort-based ranking.
            if m == 3:
                a, b, c = seg[0], seg[1], seg[2]
                if a <= b:
                    if b <= c:
                        idx = 0  # 0,1,2
                    elif a <= c:
                        idx = 1  # 0,2,1
                    else:
                        idx = 2  # 1,2,0 -> actually 2,0,1... let me be precise
                else:  # a > b
                    if a <= c:
                        idx = 3  # 1,0,2
                    elif b <= c:
                        idx = 4  # 2,0,1
                    else:
                        idx = 5  # 2,1,0
                counts[idx] += 1
            else:
                # Generic but slow -- not used for m=3
                counts[0] += 1

        # Compute Shannon entropy
        total = 0
        for i in range(n_patterns):
            total += counts[i]
        if total == 0:
            continue

        entropy = 0.0
        for i in range(n_patterns):
            if counts[i] > 0:
                p = counts[i] / total
                entropy -= p * np.log(p)

        # Normalize to [0, 1]
        out[t] = entropy / log_mfact

    return out


@njit(cache=True)
def _kyle_lambda_rolling(close, signed_volume, window=50):
    """Rolling correlation of log_return with signed_volume.

    Returns values in [-1, 1] measuring price impact sensitivity:
    - High positive = buy pressure moves price up efficiently (liquid informed flow)
    - Near zero = volume and price disconnected (noise)
    - High negative = contrarian (sell pressure moves price up -- rare, market making)

    Uses correlation instead of OLS slope because it's inherently scale-invariant
    across different price levels ($80K BTC vs $0.17 DOGE) and volume scales.
    Raw OLS slope produced near-zero values for altcoins due to unit mismatch.
    """
    n = len(close)
    out = np.full(n, 0.0)
    # Pre-compute log returns (element-wise for numba compat)
    log_ret = np.zeros(n)
    for i in range(1, n):
        prev = close[i - 1]
        if prev > 1e-12:
            log_ret[i] = math.log(close[i] / prev)

    for t in range(window, n):
        sv_mean = 0.0
        lr_mean = 0.0
        for i in range(t - window, t):
            sv_mean += signed_volume[i]
            lr_mean += log_ret[i]
        sv_mean /= window
        lr_mean /= window

        cov = 0.0
        var_sv = 0.0
        var_lr = 0.0
        for i in range(t - window, t):
            d_sv = signed_volume[i] - sv_mean
            d_lr = log_ret[i] - lr_mean
            cov += d_sv * d_lr
            var_sv += d_sv * d_sv
            var_lr += d_lr * d_lr

        denom = math.sqrt(var_sv * var_lr)
        if denom > 1e-15:
            out[t] = cov / denom  # Pearson correlation in [-1, 1]

    return out


# ── Normalization ─────────────────────────────────────────────────────────────

def robust_normalize(df: pl.DataFrame, col_name: str, window: int = 200) -> pl.Series:
    """Rolling z-score with ±5 clipping. No look-ahead."""
    mu = df[col_name].rolling_mean(window)
    sigma = df[col_name].rolling_std(window)
    z_score = ((df[col_name] - mu) / (sigma + 1e-5)).clip(-5.0, 5.0)
    # Forward-fill warmup nulls (first ~200 bars) instead of 0.0
    # fill_null(0.0) creates synthetic "neutral" data that models can overfit to
    return z_score.fill_null(strategy="forward").fill_null(0.0)  # final fill_null for row 0


def get_regime_metric(df: pl.DataFrame) -> pl.Series:
    """Distance from rolling mean, normalized."""
    ma_regime = df["close"].rolling_mean(WINDOW_REGIME)
    dist = (df["close"] - ma_regime) / (ma_regime + 1e-9)
    mu = dist.rolling_mean(WINDOW_REGIME)
    sigma = dist.rolling_std(WINDOW_REGIME)
    return ((dist - mu) / (sigma + 1e-9)).clip(-3.0, 3.0)


# ── Targets (Multi-Horizon) ──────────────────────────────────────────────────

def add_strategic_targets(df: pl.DataFrame) -> pl.DataFrame:
    """
    Multi-horizon return targets for world model training.
    Each target is a simple percentage return over the given horizon.
    No division by future volatility (that introduces look-ahead complexity).
    """
    # Ensure returns exist
    if "returns" not in df.columns:
        df = df.with_columns(
            pl.col("close").pct_change().fill_null(0).alias("returns")
        )

    # Multi-horizon returns: cumulative return over N bars
    # target_return_h = (close[t+h] - close[t]) / close[t]
    for h in [1, 4, 16, 64]:
        col_name = f"target_return_{h}"
        if h == 1:
            df = df.with_columns(
                pl.col("close").pct_change().shift(-1)
                .clip(-0.15, 0.15).alias(col_name)
            )
        else:
            df = df.with_columns(
                ((pl.col("close").shift(-h) - pl.col("close")) / (pl.col("close") + 1e-9))
                .clip(-0.50, 0.50).alias(col_name)
            )

    # Vol-normalized targets: target_voladj_{h} = return_h / realized_vol_h (with symlog)
    # These fix the TwoHot bin collapse problem: raw h=1 returns concentrate 54.9% in 1 bin,
    # while voladj targets distribute across all 255 bins (std ~0.89 vs 0.0014).
    # Uses per-horizon realized vol as denominator. Vol floor prevents blow-up.
    vol_floor = 0.001
    for h in [1, 4, 16, 64]:
        future_vol_h = pl.col("returns").rolling_std(max(h, 2)).shift(-h)
        vol_safe = pl.when(future_vol_h < vol_floor).then(vol_floor).otherwise(future_vol_h)
        raw_ret_h = (pl.col("close").shift(-h) - pl.col("close")) / (pl.col("close") + 1e-9)
        risk_adj_h = raw_ret_h / vol_safe
        # Symlog: sign(x) * log(1 + |x|) compresses heavy tails
        symlog_h = risk_adj_h.sign() * (risk_adj_h.abs() + 1.0).log()
        df = df.with_columns(
            symlog_h.clip(-5.0, 5.0).alias(f"target_voladj_{h}")
        )

    # Risk-adjusted 50-bar return (for agent reward shaping)
    # NOTE: No fill_null -- nulls at tail handled by column-selective drop in calculate_v50_features
    # FIX: Volatility floor raised from 1e-5 to 0.001 to prevent extreme values when vol is tiny.
    # FIX: Apply symlog transform (sign(x)*log(1+|x|)) to compress extremes while preserving order,
    #       then clip at +-5.0. Previously 47% of values were clipped at +-5.0 (near-binary target).
    future_vol = pl.col("returns").rolling_std(50).shift(-50)
    vol_floor = 0.001  # Minimum volatility denominator to prevent blow-up
    raw_ret_50 = (pl.col("close").shift(-50) - pl.col("close")) / (pl.col("close") + 1e-9)
    risk_adj = raw_ret_50 / (pl.when(future_vol < vol_floor).then(vol_floor).otherwise(future_vol))
    # Symlog: sign(x) * log(1 + |x|) — compresses extreme risk-adjusted returns while preserving sign/order
    symlog_ret_50 = risk_adj.sign() * (risk_adj.abs() + 1.0).log()
    df = df.with_columns(
        symlog_ret_50.clip(-5.0, 5.0).alias("target_return_50")
    )

    # Forward volatility (for risk gates)
    df = df.with_columns(
        pl.col("returns").rolling_std(20).shift(-20).alias("target_vol_20")
    )

    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def calculate_legacy_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute base features present in V303 pipeline."""

    # Base calculations
    df = df.with_columns([
        pl.col("close").pct_change().fill_null(0).alias("returns"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("volatility_measure")
    ])

    # Raw indicators
    ema_f = pl.col("close").ewm_mean(span=12)
    ema_s = pl.col("close").ewm_mean(span=26)
    eff_num = (pl.col("close") - pl.col("close").shift(10)).abs()
    eff_den = (pl.col("close") - pl.col("close").shift(1)).abs().rolling_sum(10) + 1e-9

    df = df.with_columns([
        ((ema_f - ema_s) / ema_s).alias("raw_deviation"),
        (pl.col("volume") / (pl.col("tick_count") + 1)).alias("raw_whale"),
        (pl.col("volume") * pl.col("returns").abs()).alias("raw_vpin_proxy"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("raw_spread"),
        (eff_num / eff_den).alias("raw_efficiency")
    ])

    # Activity features
    # Rogers-Satchell realized volatility: drift-independent, ~6x efficiency vs close-to-close
    # RS_var = log(H/C)*log(H/O) + log(L/C)*log(L/O), clamped >= 0, then sqrt
    # Replaces the crude (H-L)/O estimator (V51 upgrade, 2026-03-08)
    df = df.with_columns([
        pl.col("tick_count").cast(pl.Float64).alias("raw_tick_count"),
        (pl.col("volume") + 1.0).log().alias("raw_log_volume"),
        (
            (pl.col("high") / pl.col("close")).log() * (pl.col("high") / pl.col("open")).log()
            + (pl.col("low") / pl.col("close")).log() * (pl.col("low") / pl.col("open")).log()
        ).clip(lower_bound=0.0).sqrt().fill_nan(0.0).alias("raw_hl_spread")
    ])

    # Funding
    # P5 FIX: Cap forward-fill at 36 bars (~6h at 5min/bar = 3 funding periods).
    # Bare fill_null(0.0) was infinite forward-fill: 20+ bars of "unchanged" funding
    # is actually missing data, not a real signal.
    if "funding_rate" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("funding_rate"))
    else:
        df = df.with_columns(
            pl.col("funding_rate")
            .fill_null(strategy="forward", limit=36)
            .fill_null(0.0)
        )

    # Normalize legacy features
    df = df.with_columns([
        robust_normalize(df, "raw_deviation", WINDOW_ADAPTIVE).alias("norm_deviation"),
        robust_normalize(df, "raw_whale", WINDOW_ADAPTIVE).alias("norm_whale"),
        robust_normalize(df, "raw_spread", WINDOW_ADAPTIVE).alias("norm_spread"),
        robust_normalize(df, "funding_rate", WINDOW_ADAPTIVE).alias("norm_funding"),
        robust_normalize(df, "raw_efficiency", WINDOW_ADAPTIVE).alias("norm_efficiency"),
        robust_normalize(df, "raw_vpin_proxy", WINDOW_ADAPTIVE).alias("norm_vpin")
    ])

    # Blindspot features
    df = df.with_columns([
        robust_normalize(df, "raw_tick_count", WINDOW_ADAPTIVE).alias("norm_tick_count"),
        robust_normalize(df, "raw_log_volume", WINDOW_ADAPTIVE).alias("norm_log_volume"),
        robust_normalize(df, "raw_hl_spread", WINDOW_ADAPTIVE).alias("norm_hl_spread")
    ])

    return df


# ── V4 CHIMERA PHYSICS (MASTER) ──────────────────────────────────────────────

def calculate_v50_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Complete V4 feature pipeline. Produces 18 base features + targets.

    FILL STRATEGY (audited 2026-02-21):
      - Features: forward-fill then 0.0 is acceptable for rolling warmup periods.
        The 1200-bar warmup (fd_window=1000 + WINDOW_ADAPTIVE=200) is explicitly
        nulled and dropped via drop_nulls(subset=...) at the end.
      - Targets: NEVER fill_null/fill_nan. Tail rows with null targets are dropped
        by the column-selective drop_nulls at line ~392.
      - hurst_regime: 400-bar warmup (200 Hurst + 200 z-score) explicitly nulled
        and included in drop_cols for safety.
    """

    # 1. Base + Legacy
    df = calculate_legacy_features(df)

    # 2. SOTA Flow Imbalance (from real buy/sell volume)
    has_delta = "buy_vol" in df.columns and "sell_vol" in df.columns

    if has_delta:
        df = df.with_columns([
            ((pl.col("buy_vol") - pl.col("sell_vol")) / (pl.col("volume") + 1e-9))
            .fill_nan(0).alias("raw_flow_imbalance"),
            ((pl.col("buy_vol") - pl.col("sell_vol")).abs() / (pl.col("volume") + 1e-9))
            .fill_nan(0).alias("raw_vpin_sota")
        ])
    else:
        df = df.with_columns([
            ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 1e-9))
            .fill_nan(0).alias("raw_flow_imbalance"),
            pl.col("raw_vpin_proxy").alias("raw_vpin_sota")
        ])

    df = df.with_columns([
        robust_normalize(df, "raw_flow_imbalance", WINDOW_ADAPTIVE).alias("norm_flow_imbalance"),
        robust_normalize(df, "raw_vpin_sota", WINDOW_ADAPTIVE).alias("norm_vpin")  # Overwrite proxy
    ])

    # 3. Volatility Clustering
    df = df.with_columns(
        # BUG FIX: was using norm_deviation (z-scored, clipped ±5) -- weakened signal.
        # Use raw_deviation (EMA spread) to get true volatility-of-volatility.
        pl.col("raw_deviation").rolling_std(50).fill_null(strategy="forward").fill_null(0.0).alias("raw_vol_cluster")
    )
    df = df.with_columns([
        robust_normalize(df, "raw_vol_cluster", WINDOW_ADAPTIVE).alias("norm_vol_cluster")
    ])

    # 4. FracDiff & Hurst (Numba)
    log_close = df["close"].log().fill_nan(0).fill_null(0).to_numpy()

    # A. Fractional Differentiation
    fd_window = 1000
    fd_close = frac_diff_fast(log_close, d=0.4, window=fd_window)

    # B. Hurst Regime — compute on LOG RETURNS, not raw prices
    # R/S on prices gives ~0.5 constant (unit root); on returns gives true Hurst exponent
    log_returns = np.diff(log_close, prepend=log_close[0])
    hurst_val = get_rs_hurst_rolling(log_returns, window=200)

    df = df.with_columns([
        pl.Series(name="fd_close", values=fd_close).fill_nan(0.0).fill_null(0.0),
        pl.Series(name="raw_hurst", values=hurst_val).fill_nan(0.5).fill_null(0.5)
    ])

    # FD Normalization — V4 FIX: Use rolling normalization (no look-ahead bias)
    df = df.with_columns([
        robust_normalize(df, "fd_close", WINDOW_ADAPTIVE).alias("norm_fd_close")
    ])

    # FIX: norm_fd_close has a massive spike in the first ~1000 bars because frac_diff returns NaN
    # for indices < window, which get filled to 0.0 then z-scored into an outlier spike.
    # Extend warmup to fd_window + WINDOW_ADAPTIVE so the rolling z-score also stabilizes.
    warmup_len = fd_window + WINDOW_ADAPTIVE
    warmup_mask = pl.Series(name="warmup", values=[True] * min(warmup_len, len(df)) + [False] * max(0, len(df) - warmup_len))
    df = df.with_columns(
        pl.when(warmup_mask).then(None).otherwise(pl.col("norm_fd_close")).alias("norm_fd_close")
    )

    # FIX: hurst_regime was the only feature NOT getting rolling z-score normalization (std=0.04 vs ~1.0).
    # Apply same robust_normalize as all other features. Column name kept as "hurst_regime" for backward compat.
    df = df.with_columns([
        robust_normalize(df, "raw_hurst", WINDOW_ADAPTIVE).alias("hurst_regime")
    ])

    # Null hurst_regime warmup (200 bars Hurst + 200 bars z-score = 400 bars).
    # Currently redundant (fd_close warmup=1200 > 400), but future-proof if fd_window changes.
    hurst_warmup_len = 200 + WINDOW_ADAPTIVE  # 400
    hurst_warmup_mask = pl.Series(
        name="hurst_warmup",
        values=([True] * min(hurst_warmup_len, len(df))
                + [False] * max(0, len(df) - hurst_warmup_len))
    )
    df = df.with_columns(
        pl.when(hurst_warmup_mask).then(None).otherwise(pl.col("hurst_regime")).alias("hurst_regime")
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # NEW V4 FEATURES
    # ═══════════════════════════════════════════════════════════════════════════

    # 5. Open Interest Change (norm_oi_change)
    # If OI data was joined upstream, compute rate of change
    # P4 FIX: Detect OI outlier spikes (liquidation cascades) before pct_change.
    # Median-based clipping: values > 5x rolling median are capped to prevent
    # extreme pct_change values from dominating the feature.
    if "open_interest_val" in df.columns:
        oi_filled = pl.col("open_interest_val").fill_null(strategy="forward")
        oi_median = oi_filled.rolling_median(50)
        # Cap at 5x rolling median (handles flash spikes from liquidation cascades)
        oi_capped = pl.when(oi_filled > 5.0 * oi_median).then(5.0 * oi_median).otherwise(oi_filled)
        df = df.with_columns(oi_capped.alias("_oi_capped"))
        df = df.with_columns(
            pl.col("_oi_capped")
            .pct_change().fill_null(0.0).fill_nan(0.0).clip(-1.0, 1.0)
            .alias("raw_oi_change")
        )
        df = df.drop("_oi_capped")
        df = df.with_columns([
            robust_normalize(df, "raw_oi_change", WINDOW_ADAPTIVE).alias("norm_oi_change")
        ])
    else:
        # Zero-fill if OI data not available
        df = df.with_columns(pl.lit(0.0).alias("norm_oi_change"))

    # 6. Lagged Returns (multi-horizon: 1, 4, 16 bars)
    # norm_return_1: 1-bar lag (autoregressive signal)
    # norm_return_4: 4-bar lag (short-term momentum/mean-reversion, corr -0.007 with target_r1)
    # norm_return_16: 16-bar lag (medium-term momentum, corr -0.008 with target_r1)
    # Multi-horizon returns capture mean-reversion signal the model can't compute from ret_1 alone
    df = df.with_columns(
        pl.col("returns").shift(1).fill_null(0.0).alias("raw_return_1")
    )
    df = df.with_columns([
        robust_normalize(df, "raw_return_1", WINDOW_ADAPTIVE).alias("norm_return_1")
    ])

    # 4-bar cumulative lagged return
    df = df.with_columns(
        ((pl.col("close").shift(1) - pl.col("close").shift(5))
         / (pl.col("close").shift(5) + 1e-9))
        .fill_null(0.0).fill_nan(0.0).clip(-0.5, 0.5)
        .alias("raw_return_4")
    )
    df = df.with_columns([
        robust_normalize(df, "raw_return_4", WINDOW_ADAPTIVE).alias("norm_return_4")
    ])

    # 16-bar cumulative lagged return
    df = df.with_columns(
        ((pl.col("close").shift(1) - pl.col("close").shift(17))
         / (pl.col("close").shift(17) + 1e-9))
        .fill_null(0.0).fill_nan(0.0).clip(-0.5, 0.5)
        .alias("raw_return_16")
    )
    df = df.with_columns([
        robust_normalize(df, "raw_return_16", WINDOW_ADAPTIVE).alias("norm_return_16")
    ])

    # 7. Effective Spread Proxy (norm_spread_bps)
    # Approximation: Use alternating buy/sell trade flow as spread proxy
    # When buy_vol ≈ sell_vol, spread is narrow; when skewed, spread widens
    # Alternative: use (high - low) / (2 * VWAP) as Roll spread estimate
    if has_delta:
        # Roll-inspired: |2 * midpoint_change| ≈ spread
        # But simpler: imbalance-adjusted HL range gives effective cost
        total_vol = pl.col("buy_vol") + pl.col("sell_vol") + 1e-9
        balance = (pl.col("buy_vol") - pl.col("sell_vol")).abs() / total_vol
        # When flow is balanced, spread ≈ HL range / close * 10000 (in bps)
        # When flow is skewed, effective spread increases
        df = df.with_columns(
            (((pl.col("high") - pl.col("low")) / (pl.col("close") + 1e-9)) * (1.0 + balance) * 10000)
            .fill_nan(0.0).alias("raw_spread_bps")
        )
    else:
        # Fallback: simple HL spread in basis points
        df = df.with_columns(
            ((pl.col("high") - pl.col("low")) / (pl.col("close") + 1e-9) * 10000)
            .fill_nan(0.0).alias("raw_spread_bps")
        )

    df = df.with_columns([
        robust_normalize(df, "raw_spread_bps", WINDOW_ADAPTIVE).alias("norm_spread_bps")
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 1 FEATURES (orthogonal replacements, V51b)
    # Replaced: leverage_ratio (==oi_change), imbalance_intensity (~=flow_imbalance),
    #           parkinson_vol (~=hl_spread). All were redundant after z-score normalization.
    # ═══════════════════════════════════════════════════════════════════════════

    # T1. Return Kurtosis — rolling excess kurtosis of returns (window=50)
    # Harvey & Siddique (2000): conditional kurtosis predicts future returns.
    # Fat tails (high kurtosis) precede regime changes and volatility spikes.
    # Orthogonal to vol_cluster (vol-of-vol) — kurtosis captures distribution SHAPE,
    # not magnitude. A stable market can have low vol but fat tails.
    ret_col = pl.col("returns")
    ret_mean = ret_col.rolling_mean(50)
    ret_std = ret_col.rolling_std(50) + 1e-9
    # Excess kurtosis = E[(x-mu)^4] / std^4 - 3
    # Polars doesn't have rolling_kurtosis, so compute via rolling_mean of (x-mu)^4
    # Use centered 4th moment: rolling_mean((ret - rolling_mean)^4) / rolling_std^4 - 3
    df = df.with_columns(
        (ret_col - ret_mean).pow(4).rolling_mean(50)
        .truediv(ret_std.pow(4))
        .sub(3.0)
        .fill_nan(0.0).fill_null(0.0).clip(-10.0, 50.0)
        .alias("raw_return_kurtosis")
    )
    df = df.with_columns([
        robust_normalize(df, "raw_return_kurtosis", WINDOW_ADAPTIVE).alias("norm_return_kurtosis")
    ])

    # T2. Bar Duration — log time between consecutive dollar bars (seconds)
    # Dollar bars fire on fixed volume, so inter-bar TIME varies with activity.
    # Short duration = high activity/liquidity, long = quiet/illiquid.
    # Orthogonal to tick_count (trades per bar) and log_volume (~constant on dollar bars).
    # Amihud (|ret|/vol) was degenerate on dollar bars (vol is constant by construction).
    df = df.with_columns(
        pl.col("timestamp").diff().fill_null(0).cast(pl.Float64)
        .truediv(1000.0)  # ms -> seconds
        .clip(0.1, 86400.0)  # clip to [0.1s, 24h]
        .log()
        .fill_nan(0.0).fill_null(0.0)
        .alias("raw_bar_duration")
    )
    df = df.with_columns([
        robust_normalize(df, "raw_bar_duration", WINDOW_ADAPTIVE).alias("norm_bar_duration")
    ])

    # T3. Funding Rate Momentum — rate of change of funding rate (rolling slope)
    # Current norm_funding captures the LEVEL of funding rate.
    # Funding momentum captures whether leverage is BUILDING or UNWINDING.
    # Rapidly increasing funding = crowd building leveraged positions = fragile.
    # 2026-05-19 fix: prior implementation used .diff(8).rolling_mean(16) which
    # produced a mostly-zero raw signal for assets with limited funding variability,
    # causing robust_normalize to output a degenerate distribution (std~0.10 instead
    # of the expected ~1.0). Validation check_zscore_invariants caught this across
    # all u100 assets. New implementation: trailing-window mean deviation (current
    # funding minus 30-bar trailing mean), which preserves momentum semantics
    # without the diff-of-smoothed-zeros degeneracy. NaN propagation is preserved
    # for assets without funding data (instead of silent 0.0 fill that biased the
    # robust_normalize denominator).
    if "funding_rate" in df.columns:
        df = df.with_columns(
            pl.col("funding_rate")
            .fill_null(strategy="forward", limit=36)
            .alias("_fr_filled")
        )
        df = df.with_columns(
            (pl.col("_fr_filled") - pl.col("_fr_filled").rolling_mean(30))
            .alias("raw_funding_momentum")
        ).drop("_fr_filled")
    else:
        # explicit null (not 0.0) — let robust_normalize handle warmup
        df = df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("raw_funding_momentum")
        )
    df = df.with_columns([
        robust_normalize(df, "raw_funding_momentum", WINDOW_ADAPTIVE).alias("norm_funding_momentum")
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # HAWKES INTENSITY PROXY (bar-level trade clustering signal)
    # Captures whether trade arrivals are self-exciting (clustering) or dispersing.
    # True Hawkes process models intensity lambda(t) = mu + sum(alpha * exp(-beta*(t-t_i))).
    # At bar level, we approximate this with:
    #   hawkes_intensity = tick_rate / EMA(tick_rate) - 1
    # where tick_rate = tick_count / bar_duration (trades per second).
    # Positive values = trade clustering accelerating (self-excitation).
    # Negative values = trade intensity decaying back to baseline.
    # Orthogonal to norm_tick_count (raw count) and norm_bar_duration (raw timing):
    #   - tick_count: "how many trades in this bar" (level)
    #   - bar_duration: "how long this bar took" (timing)
    #   - hawkes_intensity: "is trade rate ACCELERATING vs recent baseline" (dynamics)
    # ═══════════════════════════════════════════════════════════════════════════

    # Compute tick rate (trades per second) — needs bar_duration in seconds
    bar_dur_sec = (pl.col("timestamp").diff().fill_null(0).cast(pl.Float64) / 1000.0).clip(0.1, 86400.0)
    tick_rate = pl.col("tick_count").cast(pl.Float64) / bar_dur_sec

    df = df.with_columns(tick_rate.fill_nan(0.0).fill_null(0.0).alias("_tick_rate"))

    # Hawkes proxy: ratio of current tick_rate to its EMA (captures self-excitation)
    # EMA span=20 matches WINDOW_FAST for consistency
    df = df.with_columns(
        (pl.col("_tick_rate") / (pl.col("_tick_rate").ewm_mean(span=WINDOW_FAST) + 1e-9) - 1.0)
        .fill_nan(0.0).fill_null(0.0)
        .clip(-5.0, 10.0)
        .alias("raw_hawkes_intensity")
    )
    df = df.drop("_tick_rate")

    # Buy-side Hawkes: same but only for buyer-initiated trades
    # Captures informed buying acceleration specifically
    buy_rate = pl.col("buy_vol") / (bar_dur_sec * (pl.col("volume") + 1e-9))
    df = df.with_columns(buy_rate.fill_nan(0.0).fill_null(0.0).alias("_buy_rate"))
    df = df.with_columns(
        (pl.col("_buy_rate") / (pl.col("_buy_rate").ewm_mean(span=WINDOW_FAST) + 1e-9) - 1.0)
        .fill_nan(0.0).fill_null(0.0)
        .clip(-5.0, 10.0)
        .alias("raw_hawkes_buy_intensity")
    )
    df = df.drop("_buy_rate")

    # Sell-side Hawkes: same but only for seller-initiated trades
    # Captures informed selling acceleration (panic, liquidations, distribution)
    sell_rate = pl.col("sell_vol") / (bar_dur_sec * (pl.col("volume") + 1e-9))
    df = df.with_columns(sell_rate.fill_nan(0.0).fill_null(0.0).alias("_sell_rate"))
    df = df.with_columns(
        (pl.col("_sell_rate") / (pl.col("_sell_rate").ewm_mean(span=WINDOW_FAST) + 1e-9) - 1.0)
        .fill_nan(0.0).fill_null(0.0)
        .clip(-5.0, 10.0)
        .alias("raw_hawkes_sell_intensity")
    )
    df = df.drop("_sell_rate")

    # Hawkes Imbalance: buy clustering - sell clustering
    # Positive = buyers clustering faster than sellers (informed buying)
    # Negative = sellers clustering faster (informed selling / liquidation)
    # Orthogonal to flow_imbalance (volume level) -- this captures ACCELERATION difference
    df = df.with_columns(
        (pl.col("raw_hawkes_buy_intensity") - pl.col("raw_hawkes_sell_intensity"))
        .clip(-10.0, 10.0)
        .alias("raw_hawkes_imbalance")
    )

    df = df.with_columns([
        robust_normalize(df, "raw_hawkes_intensity", WINDOW_ADAPTIVE).alias("norm_hawkes_intensity"),
        robust_normalize(df, "raw_hawkes_buy_intensity", WINDOW_ADAPTIVE).alias("norm_hawkes_buy_intensity"),
        robust_normalize(df, "raw_hawkes_sell_intensity", WINDOW_ADAPTIVE).alias("norm_hawkes_sell_intensity"),
        robust_normalize(df, "raw_hawkes_imbalance", WINDOW_ADAPTIVE).alias("norm_hawkes_imbalance"),
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # IC-BOOSTING FEATURES (Tier 2 — dynamics, not levels)
    # These capture RELATIONSHIPS and CHANGES that per-bar level features miss.
    # The IC=0.03 ceiling exists because all prior features are point-in-time
    # levels. The model must learn temporal dynamics from raw z-scores — wasteful.
    # These features pre-compute the dynamics, freeing model capacity for patterns.
    # ═══════════════════════════════════════════════════════════════════════════

    # IC1. Momentum Acceleration — second derivative of price
    # d(return_4)/dt: is momentum BUILDING or FADING?
    # A 4-bar return that's increasing = acceleration (trend strengthening)
    # A 4-bar return that's decreasing = deceleration (reversal incoming)
    # Orthogonal to return_4 (level of momentum) — this captures its CHANGE.
    df = df.with_columns(
        (pl.col("raw_return_4") - pl.col("raw_return_4").shift(4))
        .fill_null(0.0).fill_nan(0.0).clip(-0.5, 0.5)
        .alias("raw_momentum_accel")
    )
    df = df.with_columns([
        robust_normalize(df, "raw_momentum_accel", WINDOW_ADAPTIVE).alias("norm_momentum_accel")
    ])

    # IC2. Volume-Price Divergence — rolling correlation of volume and |returns|
    # When volume increases but price barely moves = accumulation (smart money loading)
    # When volume drops but price moves a lot = distribution (low-conviction move)
    # This is a classic technical analysis signal (On-Balance Volume theory).
    # We compute rolling Pearson correlation over 20 bars.
    # Negative correlation = DIVERGENCE (accumulation/distribution phase).
    df = df.with_columns(
        pl.col("raw_log_volume").rolling_mean(WINDOW_FAST).alias("_vol_mean"),
        pl.col("returns").abs().rolling_mean(WINDOW_FAST).alias("_ret_mean"),
    )
    # Pearson correlation = cov(x,y) / (std(x) * std(y))
    # Use rolling covariance / (rolling std * rolling std)
    df = df.with_columns(
        (
            (pl.col("raw_log_volume") - pl.col("_vol_mean"))
            * (pl.col("returns").abs() - pl.col("_ret_mean"))
        ).rolling_mean(WINDOW_FAST).alias("_vp_cov"),
        pl.col("raw_log_volume").rolling_std(WINDOW_FAST).alias("_vol_std"),
        pl.col("returns").abs().rolling_std(WINDOW_FAST).alias("_ret_std"),
    )
    df = df.with_columns(
        (pl.col("_vp_cov") / (pl.col("_vol_std") * pl.col("_ret_std") + 1e-9))
        .fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0)
        .alias("raw_vol_price_corr")
    )
    df = df.drop(["_vol_mean", "_ret_mean", "_vp_cov", "_vol_std", "_ret_std"])
    df = df.with_columns([
        robust_normalize(df, "raw_vol_price_corr", WINDOW_ADAPTIVE).alias("norm_vol_price_corr")
    ])

    # IC3. Volatility Term Structure — short-term vol / long-term vol
    # Vol inversion (short > long) = stress, mean-revert expected
    # Normal (short < long) = calm, trend continuation likely
    # Uses Rogers-Satchell vol at two scales: EMA(10) and EMA(50).
    df = df.with_columns([
        pl.col("raw_hl_spread").ewm_mean(span=10).alias("_vol_short"),
        pl.col("raw_hl_spread").ewm_mean(span=50).alias("_vol_long"),
    ])
    df = df.with_columns(
        (pl.col("_vol_short") / (pl.col("_vol_long") + 1e-9) - 1.0)
        .fill_nan(0.0).fill_null(0.0).clip(-5.0, 5.0)
        .alias("raw_vol_ratio")
    )
    df = df.drop(["_vol_short", "_vol_long"])
    df = df.with_columns([
        robust_normalize(df, "raw_vol_ratio", WINDOW_ADAPTIVE).alias("norm_vol_ratio")
    ])

    # IC4. Flow Persistence — autocorrelation of order flow
    # Sustained institutional flow has high autocorrelation (buying over many bars).
    # Noise trades have ~zero autocorrelation (random direction each bar).
    # Compute: corr(flow_imbalance[t], flow_imbalance[t-1]) over rolling window.
    # High persistence = institutional campaign. Low = retail noise.
    df = df.with_columns(
        pl.col("raw_flow_imbalance").shift(1).fill_null(0.0).alias("_flow_lag1")
    )
    df = df.with_columns(
        pl.col("raw_flow_imbalance").rolling_mean(WINDOW_FAST).alias("_flow_mean"),
        pl.col("_flow_lag1").rolling_mean(WINDOW_FAST).alias("_flowlag_mean"),
    )
    df = df.with_columns(
        (
            (pl.col("raw_flow_imbalance") - pl.col("_flow_mean"))
            * (pl.col("_flow_lag1") - pl.col("_flowlag_mean"))
        ).rolling_mean(WINDOW_FAST).alias("_flow_cov"),
        pl.col("raw_flow_imbalance").rolling_std(WINDOW_FAST).alias("_flow_std"),
        pl.col("_flow_lag1").rolling_std(WINDOW_FAST).alias("_flowlag_std"),
    )
    df = df.with_columns(
        (pl.col("_flow_cov") / (pl.col("_flow_std") * pl.col("_flowlag_std") + 1e-9))
        .fill_nan(0.0).fill_null(0.0).clip(-1.0, 1.0)
        .alias("raw_flow_persistence")
    )
    df = df.drop(["_flow_lag1", "_flow_mean", "_flowlag_mean", "_flow_cov", "_flow_std", "_flowlag_std"])
    df = df.with_columns([
        robust_normalize(df, "raw_flow_persistence", WINDOW_ADAPTIVE).alias("norm_flow_persistence")
    ])

    # IC5. OI-Price Divergence — OI building while price flat (positioning buildup)
    # When OI increases but price doesn't move much = position buildup (breakout incoming)
    # When OI drops and price moves = position unwinding (trend exhaustion)
    # Captures the "spring loading" effect before large moves.
    if "open_interest_val" in df.columns or "raw_oi_change" in df.columns:
        oi_col = "raw_oi_change" if "raw_oi_change" in df.columns else "norm_oi_change"
        df = df.with_columns(
            (pl.col(oi_col).abs().ewm_mean(span=10) / (pl.col("returns").abs().ewm_mean(span=10) + 1e-9) - 1.0)
            .fill_nan(0.0).fill_null(0.0).clip(-5.0, 10.0)
            .alias("raw_oi_price_divergence")
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("raw_oi_price_divergence"))
    df = df.with_columns([
        robust_normalize(df, "raw_oi_price_divergence", WINDOW_ADAPTIVE).alias("norm_oi_price_divergence")
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # SOTA FEATURES (Tier 3 — institutional-grade replacements, V52)
    # These are ADDITIVE -- existing features kept for backward compatibility.
    # New models (V1.7+) can use these alongside or instead of their predecessors.
    #
    # NOTE: SOTA equivalences (for documentation, NOT replacement):
    #   norm_yz_volatility  -> upgrades norm_hl_spread (adds overnight jump correction)
    #   norm_cs_spread      -> upgrades norm_spread_bps (principled H/L estimator)
    #   norm_perm_entropy   -> new (predictability/complexity measure)
    #   norm_kyle_lambda    -> new (price impact per dollar of order flow)
    # ═══════════════════════════════════════════════════════════════════════════

    # SOTA1. Yang-Zhang Volatility Estimator
    # Minimum Variance Unbiased Estimator combining:
    #   - Overnight variance: var(ln(O_i / C_{i-1}))
    #   - Open-to-close variance: var(ln(C_i / O_i))
    #   - Rogers-Satchell variance: ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
    # k = 0.34 / (1.34 + (n+1)/(n-1)) where n = window size
    # More efficient than Rogers-Satchell alone because it incorporates
    # overnight gaps (open vs prior close) which RS ignores.
    # Keeps norm_hl_spread (RS) intact for backward compatibility.
    yz_window = 20
    yz_k = 0.34 / (1.34 + (yz_window + 1) / (yz_window - 1))

    # Overnight component: ln(open / prior_close)
    df = df.with_columns(
        (pl.col("open") / pl.col("close").shift(1)).log()
        .fill_nan(0.0).fill_null(0.0)
        .alias("_yz_overnight")
    )
    # Open-to-close component: ln(close / open)
    df = df.with_columns(
        (pl.col("close") / pl.col("open")).log()
        .fill_nan(0.0).fill_null(0.0)
        .alias("_yz_oc")
    )
    # Rogers-Satchell component (already computed as raw_hl_spread, but need variance not sqrt)
    df = df.with_columns(
        (
            (pl.col("high") / pl.col("close")).log() * (pl.col("high") / pl.col("open")).log()
            + (pl.col("low") / pl.col("close")).log() * (pl.col("low") / pl.col("open")).log()
        ).clip(lower_bound=0.0).fill_nan(0.0).fill_null(0.0)
        .alias("_yz_rs_var")
    )
    # Yang-Zhang variance = overnight_var + k * oc_var + (1-k) * rs_var
    df = df.with_columns(
        (
            pl.col("_yz_overnight").pow(2).rolling_mean(yz_window)  # overnight variance
            + yz_k * pl.col("_yz_oc").pow(2).rolling_mean(yz_window)  # OC variance
            + (1 - yz_k) * pl.col("_yz_rs_var").rolling_mean(yz_window)  # RS variance
        ).clip(lower_bound=0.0).sqrt()
        .fill_nan(0.0).fill_null(0.0)
        .alias("raw_yz_volatility")
    )
    df = df.drop(["_yz_overnight", "_yz_oc", "_yz_rs_var"])
    df = df.with_columns([
        robust_normalize(df, "raw_yz_volatility", WINDOW_ADAPTIVE).alias("norm_yz_volatility")
    ])

    # SOTA2. Corwin-Schultz Alpha (spread-volatility ratio)
    # The classic CS spread formula produces all-negative alpha on dollar bars
    # because intra-bar price trends contaminate the H/L range estimate.
    # Instead of converting alpha to spread (which clips to 0), we use the raw
    # alpha_cs value directly. It captures the volatility-to-spread ratio:
    #   - More negative alpha = more volatility relative to spread
    #   - Less negative (near 0) = spread dominates (illiquid conditions)
    # Computed over 10-bar aggregated blocks for robustness.
    CS_BLOCK = 10

    h_block = pl.col("high").rolling_max(CS_BLOCK)
    l_block = pl.col("low").rolling_min(CS_BLOCK)
    h_block_prev = pl.col("high").shift(CS_BLOCK).rolling_max(CS_BLOCK)
    l_block_prev = pl.col("low").shift(CS_BLOCK).rolling_min(CS_BLOCK)

    ln_hl_sq = (h_block / l_block).log().pow(2)
    ln_hl_sq_prev = (h_block_prev / l_block_prev).log().pow(2)
    beta_cs = (ln_hl_sq + ln_hl_sq_prev) / 2.0

    h_max_2block = pl.col("high").rolling_max(2 * CS_BLOCK)
    l_min_2block = pl.col("low").rolling_min(2 * CS_BLOCK)
    gamma_cs = (h_max_2block / l_min_2block).log().pow(2)

    sqrt_2 = 1.4142135623730951
    denom_cs = 3.0 - 2.0 * sqrt_2  # ~0.1716

    alpha_cs = (
        ((2.0 * beta_cs).sqrt() - beta_cs.sqrt()) / denom_cs
        - (gamma_cs / denom_cs).sqrt()
    )

    # Use alpha directly (not spread) -- meaningful variation on dollar bars
    df = df.with_columns(
        alpha_cs.fill_nan(0.0).fill_null(0.0).clip(-1.0, 0.0)
        .alias("raw_cs_spread")
    )
    df = df.with_columns([
        robust_normalize(df, "raw_cs_spread", WINDOW_ADAPTIVE).alias("norm_cs_spread")
    ])

    # SOTA3. Permutation Entropy — rolling predictability measure
    # Measures the complexity/randomness of the return series using ordinal patterns.
    # For embedding dimension m=3, there are 3!=6 possible rank orderings.
    # PE = -sum(p_i * log(p_i)) / log(m!)
    # Normalized to [0,1]: 0 = perfectly predictable, 1 = perfectly random.
    # Low entropy = exploitable structure. High entropy = noise.
    # Numba-compiled for performance on 2000-bar rolling window.
    returns_arr = df["returns"].to_numpy().astype(np.float64)
    pe_values = _permutation_entropy_rolling(returns_arr, m=3, window=100)
    df = df.with_columns(
        pl.Series(name="raw_perm_entropy", values=pe_values)
        .fill_nan(0.5).fill_null(0.5)
    )
    df = df.with_columns([
        robust_normalize(df, "raw_perm_entropy", WINDOW_ADAPTIVE).alias("norm_perm_entropy")
    ])

    # SOTA4. Kyle's Lambda — price impact coefficient
    # Measures the slope of the price-liquidity curve: how much does $1 of
    # signed order flow move the price?
    # Regression: delta_close = lambda * signed_volume + epsilon
    # High lambda = illiquid (large price impact per dollar)
    # Low lambda = liquid (absorbs order flow without price movement)
    # Computed as rolling OLS slope over 50-bar windows.
    close_arr = df["close"].to_numpy().astype(np.float64)
    if has_delta:
        signed_vol = (df["buy_vol"] - df["sell_vol"]).to_numpy().astype(np.float64)
    else:
        signed_vol = (df["volume"].to_numpy().astype(np.float64)
                      * np.sign(np.diff(close_arr, prepend=close_arr[0])))
    kyle_values = _kyle_lambda_rolling(close_arr, signed_vol, window=50)
    df = df.with_columns(
        pl.Series(name="raw_kyle_lambda", values=kyle_values)
        .fill_nan(0.0).fill_null(0.0)
    )
    df = df.with_columns([
        robust_normalize(df, "raw_kyle_lambda", WINDOW_ADAPTIVE).alias("norm_kyle_lambda")
    ])

    # 8. MA Distance + Regime Label (norm_ma_distance, regime_label)
    # SMA_200 captures medium-term trend (~17 hours at ~5min/bar)
    # Provides: (a) continuous feature for model input, (b) discrete regime label for supervision
    # No leakage: SMA uses only past 200 bars + current close (backward-looking)
    SMA_REGIME_WINDOW = 200

    sma_200 = pl.col("close").rolling_mean(SMA_REGIME_WINDOW)
    raw_ma_dist = (pl.col("close") - sma_200) / (sma_200 + 1e-9)

    df = df.with_columns([
        raw_ma_dist.alias("raw_ma_distance")
    ])

    df = df.with_columns([
        robust_normalize(df, "raw_ma_distance", WINDOW_ADAPTIVE).alias("norm_ma_distance")
    ])

    # Discrete regime: adaptive threshold using rolling std of MA distance
    ma_dist_series = df["raw_ma_distance"]
    ma_dist_std = ma_dist_series.rolling_std(SMA_REGIME_WINDOW).fill_null(strategy="forward").fill_null(1e-6)

    df = df.with_columns([
        pl.when(pl.col("raw_ma_distance") > 0.5 * ma_dist_std)
          .then(pl.lit(2))       # bullish: above SMA by > 0.5 std
          .when(pl.col("raw_ma_distance") < -0.5 * ma_dist_std)
          .then(pl.lit(0))       # bearish: below SMA by > 0.5 std
          .otherwise(pl.lit(1))  # neutral: within +/- 0.5 std band
          .cast(pl.Int8)
          .alias("regime_label")
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TARGETS (Multi-Horizon)
    # ═══════════════════════════════════════════════════════════════════════════

    df = add_strategic_targets(df)

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL CLEANUP: Select output columns and drop nulls selectively
    # ═══════════════════════════════════════════════════════════════════════════

    # Select only final output columns (drop raw intermediates that have windowed nulls)
    output_columns = [
        "timestamp", "bar_id",
        "open", "high", "low", "close", "volume", "volume_usd",
        "buy_vol", "sell_vol", "tick_count",
        # 21 base features (18 original + 3 Tier 1)
        "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
        "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
        "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",
        "norm_spread_bps", "norm_ma_distance",
        "norm_whale", "norm_efficiency",       # restored (were computed but not output)
        "norm_return_4", "norm_return_16",      # multi-horizon lagged returns
        # Tier 1 features (V51b — orthogonal replacements)
        "norm_return_kurtosis",                 # rolling excess kurtosis (distribution shape)
        "norm_bar_duration",                    # Bar duration (volume clock speed)
        "norm_funding_momentum",                # funding rate of change (leverage dynamics)
        # Hawkes intensity proxy (trade clustering dynamics)
        "norm_hawkes_intensity",                # tick rate vs EMA (self-excitation signal)
        "norm_hawkes_buy_intensity",            # buy-side clustering (informed flow acceleration)
        "norm_hawkes_sell_intensity",            # sell-side clustering (liquidation/distribution)
        "norm_hawkes_imbalance",                # buy - sell clustering (directional clustering)
        # IC-boosting features (Tier 2 — dynamics, not levels)
        "norm_momentum_accel",                  # second derivative of price (trend acceleration)
        "norm_vol_price_corr",                  # volume-price correlation (accumulation/distribution)
        "norm_vol_ratio",                       # vol term structure (short/long vol ratio)
        "norm_flow_persistence",                # flow autocorrelation (institutional campaigns)
        "norm_oi_price_divergence",             # OI building while price flat (spring loading)
        # SOTA features (Tier 3 — institutional-grade, V52)
        "norm_yz_volatility",                   # Yang-Zhang vol (upgrades norm_hl_spread)
        "norm_cs_spread",                       # Corwin-Schultz spread (upgrades norm_spread_bps)
        "norm_perm_entropy",                    # Permutation entropy (predictability measure)
        "norm_kyle_lambda",                     # Kyle's lambda (price impact coefficient)
        # Raw return targets (backward compat with V1.0-V1.5)
        "target_return_1", "target_return_4", "target_return_16", "target_return_64",
        # Vol-normalized targets (for V1.6+ TwoHot training)
        "target_voladj_1", "target_voladj_4", "target_voladj_16", "target_voladj_64",
        # Auxiliary targets
        "target_return_50", "target_vol_20",
        "regime_label",
    ]
    output_columns = [c for c in output_columns if c in df.columns]
    df = df.select(output_columns)

    # Drop rows where ANY target is null (tail rows from shift operations)
    # Also drop rows where key features are null (e.g. norm_fd_close warmup nulls).
    # Primary targets (used in training) -- null rows must be dropped
    primary_targets = ["target_return_1", "target_return_4",
                       "target_return_16", "target_return_64",
                       "target_voladj_1", "target_voladj_4",
                       "target_voladj_16", "target_voladj_64"]
    # BUG FIX: target_return_50 and target_vol_20 are auxiliary (never used in
    # training). Including them in drop_nulls trimmed 50 extra valid rows per asset.
    # They now get fill_null(0.0) instead of causing row drops.
    for aux_col in ["target_return_50", "target_vol_20"]:
        if aux_col in df.columns:
            df = df.with_columns(pl.col(aux_col).fill_null(0.0))
    drop_cols = primary_targets + ["norm_fd_close", "hurst_regime", "norm_ma_distance"]
    drop_cols = [t for t in drop_cols if t in df.columns]
    df = df.drop_nulls(subset=drop_cols)

    return df
