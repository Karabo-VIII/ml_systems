"""Bipower Variation + Jump Activity panel ? SOTA microstructure addition.

Decomposes daily realized variance into continuous-diffusion (BPV) and
jump (JV) components, plus Lee-Mykland nonparametric jump detection.

Why this matters:
  Existing v50 vol features (yz_volatility, hl_spread, vol_cluster) all
  conflate continuous diffusion with jumps. The jump fraction JV/(BPV+JV)
  is known to predict reversals (post-jump mean-reversion) and continuations
  (no-jump trend). Industry-standard since Barndorff-Nielsen & Shephard
  (2004), Lee-Mykland (2008).

Method (per-asset, per-day):
  1. Resample raw aggTrades to 5-minute log returns (288 returns/day).
  2. Realized Variance:    RV  = ? r_i?
  3. Bipower Variation:    BPV = (pi/2) ? |r_i| * |r_{i-1}|
                                  (jump-robust because consecutive
                                   |r_i|*|r_{i-1}| dampens isolated jumps)
  4. Jump Variation:       JV  = max(RV - BPV, 0)
  5. Lee-Mykland test:     T_i = r_i / sqrt(BPV_local_window)
                                  ? flag i as jump if |T_i| exceeds
                                    Gumbel threshold (FWER ?=0.05).
  6. Aggregate daily:
        - jump_count      : number of significant jumps in the day
        - jump_signed_var : ? r_i? * sign(r_i) over jump i's
        - jump_intensity_30d : rolling 30-day EMA of jump_count

Output: data/processed/panels/daily/rv_jump_panel_<DATE>.parquet

Schema:
  date         : pl.Date
  asset        : str (e.g. "BTCUSDT")
  rv_5m        : float (realized variance, daily aggregate of 5m returns)
  bpv_5m       : float (bipower variation)
  jv_5m        : float (jump variation = max(RV-BPV, 0))
  jump_frac    : float (JV / RV ? bounded [0, 1])
  jump_count   : int (Lee-Mykland 5%-FWER significant jumps)
  jump_signed_var : float (signed sum of r_i? over jumps; positive = up jumps dominate)
  jump_intensity_30d : float (30-day EMA of jump_count, smoothed regime signal)

References:
  Barndorff-Nielsen & Shephard (2004) JFE: Power and Bipower Variation
  Lee & Mykland (2008) RFS: Jumps in Financial Markets ? Nonparametric Test
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
import layout as _layout  # noqa: E402
from parquet_io import (atomic_write_parquet, panel_delta_state,  # noqa: E402
                         append_panel_parquet, validate_existing)

PANEL_NAME = "rv_jump_panel"
RV_WINDOW_DAYS = 30  # EMA span for jump_intensity_30d; matches INTENSITY_EMA_SPAN
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# 5-minute resampling ? 288 buckets/day. Standard in BNS literature.
BUCKET_SECONDS = 300
BUCKETS_PER_DAY = 288

# Jump detection uses the day's own BPV-based volatility as the local-?
# baseline (intraday-only test). Each return's |r_i| is compared against
# the daily threshold ?_day * crit, where crit is the Gumbel quantile from
# Lee-Mykland (2008) for n returns. This avoids the multi-day warm-up
# requirement of the rolling-K formulation.
#
# ?=0.001 (per-test) is conservative for crypto's fat-tailed returns
# (kurtosis >> 3, even after BPV normalization). At ?=0.05 the test would
# flag too many inliers as jumps; at ?=0.001 the threshold ? 5?, which
# captures genuine jump events without flooding.
LM_ALPHA = 0.001

# EMA span for jump_intensity_30d (in days)
INTENSITY_EMA_SPAN = 30


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("rv", phase, message, **kw)


def lee_mykland_critical_value(n: int, alpha: float = LM_ALPHA) -> float:
    """Critical value (in raw t-statistic scale) for Lee-Mykland jump test.

    Per LM2008, under the null (no jumps), the maximum |T_i| over n returns
    has the asymptotic representation:

        (max|T_i| - C_n) / S_n ? Gumbel(0, 1)

    where C_n = (2 ln n)^(1/2) - [ln pi + ln ln n] / (2 (2 ln n)^(1/2))
          S_n = 1 / (2 ln n)^(1/2)

    For per-test FWER ?, we reject if a single |T_i| exceeds:

        crit = C_n + S_n * (-ln(-ln(1 - ?)))
    """
    c = (2.0 * math.log(n)) ** 0.5
    C_n = c - (math.log(math.pi) + math.log(math.log(n))) / (2.0 * c)
    S_n = 1.0 / c
    g_quantile = -math.log(-math.log(1.0 - alpha))
    return C_n + S_n * g_quantile


def _detect_ts_unit(ts_value: int) -> str:
    """Binance aggTrades switched from ms (13 digits) to us (16 digits)
    in 2024-2025. Per-row autodetect via magnitude.
    """
    return "us" if ts_value > 1e15 else "ms"


def _ts_to_ms(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize timestamp to milliseconds regardless of source unit."""
    sample = int(df["timestamp"][0])
    if _detect_ts_unit(sample) == "us":
        return df.with_columns((pl.col("timestamp") // 1000).alias("timestamp"))
    return df


def _resample_5m_returns(df: pl.DataFrame) -> pl.DataFrame:
    """aggTrades ? 5-minute log returns. Empty buckets ? 0 return (no trade)."""
    df = _ts_to_ms(df)
    df = df.with_columns(
        (pl.col("timestamp") // (BUCKET_SECONDS * 1000)).alias("_bucket")
    )
    # Per-bucket last price + first price for log-return; we use last
    # (matches industry convention for irregular-tick ? uniform-grid).
    grid = df.group_by("_bucket").agg([
        pl.col("price").last().alias("close"),
        pl.col("timestamp").first().alias("ts_start"),
    ]).sort("_bucket")
    grid = grid.with_columns([
        pl.col("close").log().alias("log_close"),
    ])
    grid = grid.with_columns([
        (pl.col("log_close") - pl.col("log_close").shift(1)).alias("ret_5m"),
    ]).drop_nulls(subset=["ret_5m"])
    return grid


def _daily_bpv_jv(returns: np.ndarray) -> tuple[float, float, float]:
    """Compute (RV, BPV, JV) from a vector of 5-minute log returns."""
    if len(returns) < 2:
        return 0.0, 0.0, 0.0
    r2 = returns ** 2
    rv = float(r2.sum())
    abs_r = np.abs(returns)
    # BPV = (pi/2) * ? |r_i| * |r_{i-1}|
    bpv = float((math.pi / 2.0) * (abs_r[1:] * abs_r[:-1]).sum())
    jv = max(rv - bpv, 0.0)
    return rv, bpv, jv


def _lm_jump_indicator(returns: np.ndarray, alpha: float = LM_ALPHA) -> np.ndarray:
    """Daily Lee-Mykland jump test using the day's BPV as the ? baseline.

    ?_per_step = sqrt(BPV / n)
    T_i = |r_i| / ?_per_step
    flag if T_i > Gumbel-? critical value

    This is the intraday-only formulation (no multi-day rolling window).
    Suitable for daily aggregation; if you need rolling-window LM, see the
    LM2008 sect4 cross-day variant.
    """
    n = len(returns)
    if n < 5:
        return np.zeros(n, dtype=np.int8)
    abs_r = np.abs(returns)
    pair_prod = abs_r[1:] * abs_r[:-1]
    if len(pair_prod) == 0:
        return np.zeros(n, dtype=np.int8)
    bpv = (math.pi / 2.0) * float(pair_prod.mean())
    sigma_per_step = math.sqrt(max(bpv, 1e-36))
    if sigma_per_step <= 1e-18:
        return np.zeros(n, dtype=np.int8)
    crit = lee_mykland_critical_value(n, alpha=alpha)
    t_stat = abs_r / sigma_per_step
    return (t_stat > crit).astype(np.int8)


def process_asset_day(asset: str, day_path: Path) -> dict | None:
    """Compute the daily BPV/JV/jump aggregates for one asset-day."""
    try:
        # 2026-05-13: prepare_aggtrades supersedes _ts_to_ms; adds sort for 2026-Q1+.
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bars"))
        from _aggtrades_utils import prepare_aggtrades  # noqa: E402
        df = pl.read_parquet(day_path)
        df = prepare_aggtrades(df, ts_col="timestamp")
    except Exception:
        return None
    if df.height == 0:
        return None
    grid = _resample_5m_returns(df)
    if grid.height < 10:
        return None
    returns = grid["ret_5m"].to_numpy().astype(np.float64)
    rv, bpv, jv = _daily_bpv_jv(returns)
    jump_flags = _lm_jump_indicator(returns)
    jump_count = int(jump_flags.sum())
    # signed jump variation: sum r_i? * sign(r_i) over jumps
    if jump_count > 0:
        jump_returns = returns[jump_flags == 1]
        jump_signed_var = float(np.sum(jump_returns ** 2 * np.sign(jump_returns)))
    else:
        jump_signed_var = 0.0
    # Day-end timestamp (use bucket-grid first ts as anchor)
    first_ts = int(grid["ts_start"][0])
    day_d = datetime.fromtimestamp(first_ts / 1000.0, tz=timezone.utc).date()
    # Strip USDT suffix to match the consolidator's asset filter
    # convention (other panels: BTC / ETH / ...). Without this, every
    # rv_* column joins as 100% NaN in v51 chimera.
    asset_root = asset.replace("USDT", "") if asset.endswith("USDT") else asset
    return {
        "date": day_d,
        "asset": asset_root,
        "rv_5m": rv,
        "bpv_5m": bpv,
        "jv_5m": jv,
        "jump_frac": jv / rv if rv > 1e-18 else 0.0,
        "jump_count": jump_count,
        "jump_signed_var": jump_signed_var,
    }


def build_asset(asset: str, max_days: int | None = None,
                 fps_subset: list | None = None,
                 ema_seed: float | None = None) -> pl.DataFrame:
    """Iterate raw aggTrades for one asset, producing a daily panel.

    Args:
        max_days: cap to most recent N days (smoke test).
        fps_subset: explicit file list (overrides glob); for delta builds.
        ema_seed: prior jump_intensity_30d to seed the EMA. If None, EMA
            starts at jc[0] (full-rebuild behaviour). For delta append:
            pass the existing panel's last jump_intensity_30d so the new
            window's EMA is consistent with prior history.
    """
    if fps_subset is not None:
        files = sorted(Path(p) for p in fps_subset)
    else:
        asset_dir = RAW_DIR / asset / "aggTrades"
        if not asset_dir.exists():
            return pl.DataFrame()
        files = sorted(asset_dir.glob("*.parquet"))
        if max_days is not None:
            files = files[-max_days:]
    rows = []
    for i, f in enumerate(files):
        rec = process_asset_day(asset, f)
        if rec is not None:
            rows.append(rec)
        if i % 100 == 99:
            _pl("BUILD", f"{asset}: {i+1}/{len(files)} days processed")
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows).sort("date")
    # 30-day EMA of jump_count for jump_intensity_30d
    span = INTENSITY_EMA_SPAN
    alpha = 2.0 / (span + 1.0)
    jc = df["jump_count"].to_numpy().astype(np.float64)
    ema = np.zeros_like(jc)
    if len(jc) > 0:
        ema[0] = (alpha * jc[0] + (1.0 - alpha) * ema_seed
                   if ema_seed is not None else jc[0])
        for i in range(1, len(jc)):
            ema[i] = alpha * jc[i] + (1.0 - alpha) * ema[i - 1]
    df = df.with_columns(pl.Series("jump_intensity_30d", ema))
    return df


# Default universe: u10 (matches V0/V1.x training universe)
DEFAULT_ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", nargs="+", default=None,
                    help="Explicit asset list (overrides --universe; default u10).")
    ap.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                    help="Resolve assets via UniverseLoader.")
    ap.add_argument("--max-days", type=int, default=None,
                    help="Limit to most recent N days (smoke test)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Per-asset ProcessPool workers (default 4). Each worker "
                         "reads ~700 aggTrades parquets and computes RV/BPV/jumps "
                         "for one asset; CPU+IO bound. Linear scaling up to 4-6 "
                         "workers; cap depends on RAM (~2-4 GB peak per worker).")
    ap.add_argument("--force", action="store_true",
                    help="Force rebuild ignoring existing panel (delta path skipped).")
    args = ap.parse_args()

    # @browser B1+B5: universe LOUD; explicit fallback announce
    if args.assets:
        resolved = [a.upper() if a.upper().endswith("USDT") else a.upper() + "USDT"
                    for a in args.assets]
        print(f"[rv_jumps] universe: --assets ({len(resolved)} explicit)")
    elif args.universe:
        try:
            import sys
            from pathlib import Path as _P
            sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "pipeline"))
            from universe_loader import UniverseLoader
            resolved = [s.upper() for s in UniverseLoader.load().list(args.universe)]
            print(f"[rv_jumps] universe: {args.universe} ({len(resolved)} assets)")
        except Exception as e:
            _pl("WARN", f"FALLBACK: universe={args.universe} load failed ({e}); using u10 default")
            resolved = list(DEFAULT_ASSETS)
    else:
        resolved = list(DEFAULT_ASSETS)
        print(f"[rv_jumps] universe: u10-default ({len(resolved)} assets) -- "
              f"pass --universe u50 to extend")
    args.assets = resolved

    print(f"\n{'='*70}")
    print(f"BUILD RV+JUMP PANEL  assets={len(args.assets)}  max_days={args.max_days}")
    print(f"{'='*70}\n")

    n_total = len(args.assets)
    n_workers = max(1, min(args.workers, n_total))
    print(f"[rv_jumps] parallelism: {n_workers} ProcessPool workers "
          f"(requested {args.workers}, capped at min(workers, |U|))", flush=True)

    REQUIRED_COLS = {"date", "asset", "rv_5m", "bpv_5m", "jv_5m",
                      "jump_frac", "jump_count", "jump_signed_var",
                      "jump_intensity_30d"}

    # ----- Delta detection (windowed by EMA span) -----
    existing_panel_path = _layout._pick_latest(_layout.panels_dir(), PANEL_NAME)
    existing_panel: pl.DataFrame | None = None
    delta_mode = "rebuild"
    per_asset_new_dates: dict[str, list] = {}
    per_asset_ema_seed: dict[str, float] = {}

    # Build per-asset candidate-date map from raw aggTrades file globs.
    per_asset_dates: dict[str, list] = {}
    per_asset_files: dict[str, dict] = {}
    # Phase 8: centralized listing_dates defensive filter
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
        from pipeline.listing_dates import is_pre_listing as _is_pre_listing
    except ImportError:
        _is_pre_listing = lambda *a, **k: False
    for asset in args.assets:
        asset_dir = RAW_DIR / asset / "aggTrades"
        if not asset_dir.exists():
            continue
        date_to_fp: dict = {}
        for fp in sorted(asset_dir.glob("*.parquet")):
            parts = fp.stem.split("-")
            if len(parts) < 5:
                continue
            try:
                d = date.fromisoformat("-".join(parts[-3:]))
                if not _is_pre_listing(asset, d):
                    date_to_fp[d] = fp
            except (ValueError, TypeError):
                continue
        if date_to_fp:
            per_asset_files[asset] = date_to_fp
            per_asset_dates[asset] = sorted(date_to_fp.keys())

    if existing_panel_path is not None and not args.force:
        ok, why = validate_existing(
            existing_panel_path,
            required_cols=REQUIRED_COLS,
            max_null_rate={"rv_5m": 0.05, "jump_intensity_30d": 0.05})
        if not ok:
            print(f"[rv_jumps] CORRUPT existing panel "
                  f"{existing_panel_path.name}: {why}; full rebuild",
                  flush=True)
        else:
            delta = panel_delta_state(
                existing_panel_path, per_asset_dates,
                window_days=RV_WINDOW_DAYS)
            delta_mode = delta["mode"]
            per_asset_new_dates = delta["per_asset_new_dates"]
            print(f"[rv_jumps] delta: mode={delta_mode}, {delta['reason']}",
                  flush=True)
            if delta_mode == "fresh":
                print(f"[rv_jumps] no new windows past existing max; exiting",
                      flush=True)
                return 0
            if delta_mode == "append":
                # Read existing panel + per-asset EMA seed at the day BEFORE
                # the rebuilt window starts, so the new EMA continues correctly.
                existing_panel = pl.read_parquet(existing_panel_path)
                for asset, new_dates in per_asset_new_dates.items():
                    if not new_dates:
                        continue
                    seed_cutoff = min(new_dates)  # rebuild starts here
                    asset_existing = existing_panel.filter(
                        (pl.col("asset") == asset)
                        & (pl.col("date") < seed_cutoff)
                    ).sort("date")
                    if asset_existing.height > 0:
                        per_asset_ema_seed[asset] = float(
                            asset_existing["jump_intensity_30d"][-1])

    frames = []
    # Determine per-asset fps subset.
    def _fps_for(asset: str) -> list:
        if delta_mode in ("fresh", "rebuild"):
            return None  # build_asset uses full glob + max_days
        new_dates = per_asset_new_dates.get(asset, [])
        if not new_dates:
            return []
        date_to_fp = per_asset_files.get(asset, {})
        return [str(date_to_fp[d]) for d in new_dates if d in date_to_fp]

    if n_workers <= 1:
        for i, asset in enumerate(args.assets, start=1):
            pct = 100.0 * (i - 1) / n_total
            fps_subset = _fps_for(asset)
            if fps_subset == []:
                print(f"[rv {i}/{n_total}] {asset}: nothing new", flush=True)
                continue
            print(f"[rv {i}/{n_total}] {asset} ({pct:.0f}%, "
                  f"mode={delta_mode}, files={'glob' if fps_subset is None else len(fps_subset)})",
                  flush=True)
            df = build_asset(asset, max_days=args.max_days,
                              fps_subset=fps_subset,
                              ema_seed=per_asset_ema_seed.get(asset))
            if df.height > 0:
                frames.append(df)
                print(f"  [{asset}] rows={df.height}, "
                      f"jc_mean={float(df['jump_count'].mean()):.2f}",
                      flush=True)
            else:
                _pl("SKIP", f"{asset}: SKIP no data")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {}
            for asset in args.assets:
                fps_subset = _fps_for(asset)
                if fps_subset == []:
                    continue
                futures[ex.submit(build_asset, asset, args.max_days,
                                   fps_subset, per_asset_ema_seed.get(asset))] = asset
            for fut in as_completed(futures):
                asset = futures[fut]
                completed += 1
                try:
                    df = fut.result()
                except Exception as e:
                    print(f"  [{asset}] ERROR: {type(e).__name__}: {e}",
                          flush=True)
                    continue
                if df.height > 0:
                    frames.append(df)
                    print(f"  [rv {completed}/{len(futures)}] {asset}: "
                          f"rows={df.height}, jc_mean={float(df['jump_count'].mean()):.2f}",
                          flush=True)
                else:
                    print(f"  [rv {completed}/{len(futures)}] {asset}: SKIP no data",
                          flush=True)

    if not frames:
        if delta_mode == "append":
            print(f"[rv_jumps] no new rows produced; existing panel unchanged",
                  flush=True)
            return 0
        print("[ERROR] No frames built. Exiting.")
        return 1

    new_panel = pl.concat(frames, how="vertical_relaxed").sort(["asset", "date"])

    if delta_mode == "append" and existing_panel is not None:
        # Merge: drop existing (asset, date) rows that are in new_panel,
        # concat with new rows, sort, atomic-write to new dated path.
        new_keys = new_panel.select(["asset", "date"]).unique()
        ex_keep = existing_panel.join(new_keys, on=["asset", "date"],
                                        how="anti")
        panel = pl.concat([ex_keep, new_panel],
                           how="vertical_relaxed").sort(["asset", "date"])
        print(f"[rv_jumps] delta merge: {ex_keep.height} kept + "
              f"{new_panel.height} new = {panel.height} total", flush=True)
    else:
        panel = new_panel

    d_max = panel["date"].max()
    if not isinstance(d_max, date):
        d_max = datetime.now(timezone.utc).date()
    out_path = _layout.panels_dir() / f"{PANEL_NAME}_{d_max.strftime('%Y%m%d')}.parquet"
    atomic_write_parquet(panel, out_path, required_cols=REQUIRED_COLS)
    _layout.gc_older_dated(_layout.panels_dir(), PANEL_NAME)

    print(f"\n[OK] Wrote {out_path.name}: {panel.height} rows, "
          f"{panel.select('asset').n_unique()} assets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
