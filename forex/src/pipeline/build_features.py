"""Feature engineering (silver) — per-pair, causal FX features from time bars.

Reads a bar file (pair, cadence) and computes a principled, LOOK-AHEAD-SAFE
feature frame: every feature at row t uses only information available at the
close of bar t (trailing rolling windows + past shifts; no centering, no future).

Feature groups:
  returns        multi-horizon log returns + rolling z of price
  volatility     realized vol, Parkinson, Garman-Klass, ATR (pips), vol ratio
  microstructure spread (pips + bps), spread z, tick intensity, Amihud illiquidity
  session/time   hour, day-of-week, cyclical encodings, session flags, weekend-gap
  trend/regime   SMA ratio, Kaufman efficiency ratio, vol regime z

Cross-pair / USD-strength features are NOT here (they need the full panel) — see
panels.py. Targets are NOT here (they are future returns) — see make_dataset.py.

Output: data/processed/features/<cadence>/<pair>_features_<YYYYMMDD>.parquet

    python src/pipeline/build_features.py --universe u7 --cadences 1h 1d
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import polars as pl

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[1])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import cli, layout
from pipeline.dispatch import run_per_task
from pipeline.parquet_io import atomic_write_parquet
from pipeline.universe_loader import UniverseLoader

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "build_features",
    "inputs": {"upstream": "data/processed/bars/<cadence>/<pair>_<cadence>_*.parquet",
               "args": ["--pairs", "--universe", "--cadences"]},
    "outputs": {"files": "data/processed/features/<cadence>/<pair>_features_<YYYYMMDD>.parquet"},
    "invariants": {"causal_no_lookahead": True, "trailing_windows_only": True,
                   "pip_aware": True, "atomic_write": True},
    "rationale": "Silver per-pair feature layer; the gold dataset joins these.",
}

RET_HORIZONS = (1, 3, 5, 10, 20, 50)
VOL_WINDOWS = (10, 20, 50)
_LN2_4 = 4.0 * math.log(2.0)
_EPS = 1e-12

# Approximate session windows in UTC (DST caveat: London/NY edges drift +-1h;
# Tokyo has no DST). Documented approximation for v1 features.
# (start_hour_inclusive, end_hour_exclusive) on the UTC clock.
_SESSIONS = {"tokyo": (0, 9), "london": (7, 16), "newyork": (12, 21)}


def compute_features(df: pl.DataFrame, pip: float) -> pl.DataFrame:
    """Compute the causal feature frame from a sorted bar DataFrame."""
    df = df.sort("timestamp")
    close, high, low = pl.col("close"), pl.col("high"), pl.col("low")
    prev_close = close.shift(1)
    log_ret_1 = (close / prev_close).log()

    exprs: list[pl.Expr] = [
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt"),
        log_ret_1.alias("log_ret_1"),
    ]

    # --- returns (multi-horizon) ---
    for h in RET_HORIZONS:
        exprs.append((close / close.shift(h)).log().alias(f"ret_{h}"))
    # rolling z of price (trend/mean-reversion context)
    exprs.append(((close - close.rolling_mean(100)) /
                  (close.rolling_std(100) + _EPS)).alias("close_z_100"))

    # --- volatility ---
    log_hl2 = (high / low).log() ** 2
    for w in VOL_WINDOWS:
        exprs.append((log_ret_1.pow(2).rolling_sum(w)).sqrt().alias(f"rv_{w}"))
        exprs.append(((log_hl2.rolling_mean(w)) / _LN2_4).sqrt().alias(f"parkinson_{w}"))
    # Garman-Klass (per-bar, then rolling mean) — needs open.
    log_hl = (high / low).log()
    log_co = (close / pl.col("open")).log()
    gk_bar = 0.5 * log_hl.pow(2) - (2 * math.log(2) - 1) * log_co.pow(2)
    exprs.append((gk_bar.rolling_mean(20).clip(lower_bound=0.0)).sqrt().alias("garman_klass_20"))
    # ATR in pips: true range / pip, rolling mean.
    tr = pl.max_horizontal(high - low, (high - prev_close).abs(), (low - prev_close).abs())
    for w in VOL_WINDOWS:
        exprs.append((tr.rolling_mean(w) / pip).alias(f"atr_pips_{w}"))
    # vol acceleration
    exprs.append((log_ret_1.pow(2).rolling_sum(10).sqrt() /
                  (log_ret_1.pow(2).rolling_sum(50).sqrt() + _EPS)).alias("rvol_ratio_10_50"))

    # --- microstructure ---
    if "spread_mean" in df.columns:
        spread = pl.col("spread_mean")
        exprs.append((spread / pip).alias("spread_pips"))
        exprs.append((spread / close * 1e4).alias("rel_spread_bps"))
        exprs.append(((spread - spread.rolling_mean(100)) /
                      (spread.rolling_std(100) + _EPS)).alias("spread_z_100"))
    if "tick_count" in df.columns:
        tc = pl.col("tick_count").cast(pl.Float64)
        exprs.append(((tc - tc.rolling_mean(100)) /
                      (tc.rolling_std(100) + _EPS)).alias("tick_intensity_z_100"))
    if "volume_est" in df.columns:
        exprs.append((log_ret_1.abs() / (pl.col("volume_est") + _EPS))
                     .rolling_mean(20).alias("amihud_illiq_20"))
    # quote imbalance: where the bar's close sits vs the quote midpoint, in
    # half-spread units (>0 = buying pressure). A cheap order-flow proxy.
    if "bid_close" in df.columns and "ask_close" in df.columns:
        qmid = (pl.col("bid_close") + pl.col("ask_close")) / 2
        half_spread = (pl.col("ask_close") - pl.col("bid_close")) / 2
        exprs.append(((close - qmid) / (half_spread + _EPS)).clip(-3.0, 3.0)
                     .alias("quote_imbalance"))
    # vol-normalized 5-bar momentum (raw momentum is dominated by the vol regime).
    exprs.append(((close / close.shift(5)).log() /
                  ((close / close.shift(1)).log().rolling_std(20) + _EPS))
                 .alias("mom5_z"))

    # --- session / time (causal: derived from the bar's own timestamp) ---
    dt = pl.from_epoch(pl.col("timestamp"), time_unit="ms")
    hour = dt.dt.hour()
    exprs.append(hour.alias("hour_utc"))
    exprs.append(dt.dt.weekday().alias("dow"))                 # 1=Mon .. 7=Sun
    exprs.append((hour.cast(pl.Float64) / 24.0 * 2 * math.pi).sin().alias("hour_sin"))
    exprs.append((hour.cast(pl.Float64) / 24.0 * 2 * math.pi).cos().alias("hour_cos"))
    # DST-AWARE session flags: convert the UTC bar time into each session's home
    # timezone and flag its local trading hours (auto-handles DST; Tokyo has none).
    # Requires tzdata on Windows. Falls back to fixed-UTC windows if tz is missing.
    try:
        dt_utc = dt.dt.replace_time_zone("UTC")
        tok_h = dt_utc.dt.convert_time_zone("Asia/Tokyo").dt.hour()
        ldn_h = dt_utc.dt.convert_time_zone("Europe/London").dt.hour()
        ny_h = dt_utc.dt.convert_time_zone("America/New_York").dt.hour()
        # probe once so a missing tzdata degrades gracefully rather than mid-build.
        df.select(dt_utc.dt.convert_time_zone("America/New_York").dt.hour().alias("_p")).head(1)
        exprs.append(((tok_h >= 9) & (tok_h < 18)).cast(pl.Int8).alias("is_tokyo"))
        exprs.append(((ldn_h >= 8) & (ldn_h < 17)).cast(pl.Int8).alias("is_london"))
        exprs.append(((ny_h >= 8) & (ny_h < 17)).cast(pl.Int8).alias("is_newyork"))
        exprs.append((((ldn_h >= 8) & (ldn_h < 17)) & ((ny_h >= 8) & (ny_h < 17)))
                     .cast(pl.Int8).alias("is_ldn_ny_overlap"))
    except Exception:
        for name, (h0, h1) in _SESSIONS.items():
            exprs.append(((hour >= h0) & (hour < h1)).cast(pl.Int8).alias(f"is_{name}"))
        exprs.append(((hour >= 12) & (hour < 16)).cast(pl.Int8).alias("is_ldn_ny_overlap"))
    # weekend-gap: log return across a >20h inter-bar gap (Fri->Sun reopen).
    gap_ms = pl.col("timestamp").diff()
    exprs.append(pl.when(gap_ms > 20 * 3_600_000)
                 .then(log_ret_1 / pip).otherwise(0.0).alias("weekend_gap_pips"))

    # --- trend / regime ---
    exprs.append((close / close.rolling_mean(20) - 1.0).alias("sma20_ratio"))
    exprs.append((close / close.rolling_mean(50) - 1.0).alias("sma50_ratio"))
    for w in (20, 50):
        direction = (close - close.shift(w)).abs()
        volatility_path = log_ret_1.abs().rolling_sum(w)   # sum of |moves|
        # Kaufman efficiency ratio uses price path; approximate with |ret| path.
        exprs.append((direction / (close.shift(w) * volatility_path + _EPS))
                     .alias(f"efficiency_ratio_{w}"))
    exprs.append(((log_ret_1.pow(2).rolling_sum(20).sqrt() -
                   log_ret_1.pow(2).rolling_sum(20).sqrt().rolling_mean(100)) /
                  (log_ret_1.pow(2).rolling_sum(20).sqrt().rolling_std(100) + _EPS))
                 .alias("vol_regime_z"))

    feat = df.select([pl.col("timestamp"), pl.col("close")] + exprs)
    return feat


def build_pair(pair: str, cadence: str, force: bool) -> dict:
    bar_path = layout.bars_latest(pair, cadence)
    if bar_path is None or not bar_path.exists():
        return {"status": "no_bars", "pair": pair, "cadence": cadence}
    df = pl.read_parquet(bar_path)
    if df.is_empty():
        return {"status": "empty_bars", "pair": pair, "cadence": cadence}
    U = UniverseLoader.load()
    feat = compute_features(df, U.pip(pair))
    from datetime import timezone, datetime
    latest = datetime.fromtimestamp(df["timestamp"].max() / 1000, tz=timezone.utc).date()
    out = layout.features_path(pair, cadence, latest_date=latest)
    atomic_write_parquet(feat, out, required_cols={"timestamp", "log_ret_1"})
    return {"status": "ok", "pair": pair, "cadence": cadence,
            "rows": len(feat), "n_features": len(feat.columns) - 2}


def _worker(pair, cadence, force):
    return build_pair(pair, cadence, force)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build causal FX features from bars")
    cli.add_standard_args(ap, default_workers=4, date_window=False)
    ap.add_argument("--cadences", nargs="+", default=["1h", "1d"],
                    help="Cadences to build features for (default 1h 1d).")
    args = ap.parse_args()
    pairs = cli.resolve_pairs(args, stage_name="build_features")
    tasks = [(p, cad, args.force) for p in pairs for cad in args.cadences]
    if args.dry_run:
        for t in tasks:
            print(f"  would build features {t[0]} {t[1]}")
        return
    run_per_task(tasks, _worker, workers=args.workers,
                 mode="thread" if args.workers > 1 else "serial",
                 stage_name="build_features",
                 progress_summary_keys=["cadence", "rows", "n_features"])


if __name__ == "__main__":
    main()
