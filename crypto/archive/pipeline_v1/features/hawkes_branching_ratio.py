"""G5: Tick-level Hawkes branching ratio feature (Rambaldi 2024 method).

Closes G5 from gap audit 2026-04-25.

References:
  Hardiman, S. J., Bercot, N., & Bouchaud, J.-P. (2013). Critical reflexivity
    in financial markets: a Hawkes process analysis. EPJ B 86, 442.
  Filimonov, V., & Sornette, D. (2012). Quantifying reflexivity in financial
    markets. Phys. Rev. E 85, 056108.
  Rambaldi et al. (2024). The role of tick-level branching ratio in cascade
    prediction. (Cited by ml_upgrades_research_2026_04_22.md)

What this does:
  For each asset x day:
    - Loads aggTrades parquet for that asset/day.
    - Coarse-grains into 60-second bins.
    - Computes branching ratio eta from over-dispersion: eta = 1 - sqrt(<N>/Var(N))
    - Also computes buy-side and sell-side eta separately.
    - Outputs: data/frontier/hawkes_enh/hawkes_branching_daily.parquet
  Validated downstream by IC test against ret_1d (handled by user/separate test).

Output columns:
  date, asset, eta_total, eta_buy, eta_sell, eta_imbalance, n_trades

Run:
  python src/frontier/features/hawkes_branching_ratio.py
"""
from __future__ import annotations

# CDAP contract — declared after __future__ per PEP-236.
__contract__ = {
    "kind": "pipeline_stage",
    "stage": "hawkes_branching",
    "inputs": {
        "args": ["--max-days", "--workers", "--universe {u10|u50|u100}", "--assets"],
        "upstream": "data/raw/<SYM>USDT/aggTrades/*.parquet",
    },
    "outputs": {
        "files": "data/processed/hawkes/daily/hawkes_branching_daily_*.parquet",
        "columns": ["date", "asset", "eta_total", "eta_buy", "eta_sell",
                    "eta_imbalance", "n_trades"],
        "value_ranges": {"eta_total": [0.0, 0.99]},
    },
    "invariants": {
        "atomic_write": True,
        "column_name_verify": True,
        "coverage_report_at_end": True,
        "ts_unit_per_row_autodetect": True,
        "asset_set_eq": "downstream:frontier_consolidate",
    },
}

import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
# Write to the canonical silver layer (post-2026-04-26 v3 layout). The output
# filename is dated (`hawkes_branching_daily_<YYYYMMDD>.parquet`) per the
# pipeline-wide convention; resume reads via _layout.hawkes_panel_latest().
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(ROOT / "src" / "pipeline"))
import layout as _layout  # noqa: E402
from parquet_io import atomic_write_parquet, validate_existing  # noqa: E402

OUT_DIR = _layout.hawkes_dir()
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_NAME = "hawkes_branching_daily"


def _resolve_existing_path() -> "Path|None":
    """Latest dated panel for resume; None if none exists."""
    return _layout.hawkes_panel_latest(PANEL_NAME)


# Default 10-asset universe; overridable via --universe / --assets / discovery from raw/.
DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
ASSETS = list(DEFAULT_ASSETS)


def _resolve_assets(universe: str | None, assets_arg: list | None) -> list[str]:
    """Pick the asset list. Order of precedence: --assets > --universe > discover-from-raw > default-u10.

    bar_fabric runs over 49+ assets but hawkes was historically locked to u10. When
    bar_fabric covers a wider universe, downstream consumers (frontier_consolidate)
    silently get NaN hawkes columns for non-u10 assets. Fix: accept --universe or
    discover from data/raw/ to keep coverage aligned across stages.
    """
    if assets_arg:
        return [a.upper().replace("USDT", "") for a in assets_arg]
    if universe:
        try:
            _sys.path.insert(0, str(ROOT / "src" / "pipeline"))
            from universe_loader import UniverseLoader  # noqa: E402
            syms = UniverseLoader.load().list(universe)
            return [s.replace("USDT", "").upper() for s in syms]
        except Exception as e:
            from progress import phase_log as _pl
            _pl("hawkes", "WARN", f"universe={universe} failed: {e}; using default-u10")
            return list(DEFAULT_ASSETS)
    return list(DEFAULT_ASSETS)

# Bin width for coarse-graining (seconds). 60s gives 1440 bins/day.
BIN_SEC = 60


def detect_ts_unit(ts_array: np.ndarray) -> str:
    """Per-row unit detection. Binance switched ms->us mid-2024.

    See memory/frontier_final_state_2026_04_23.md invariant #1.
    """
    if len(ts_array) == 0:
        return "ms"
    # >1e15 indicates microseconds; 1e12 for milliseconds; 1e9 for seconds
    sample = float(ts_array[len(ts_array) // 2])
    if sample > 1e15:
        return "us"
    elif sample > 1e12:
        return "ms"
    else:
        return "s"


def per_row_to_seconds(ts: np.ndarray) -> np.ndarray:
    """Convert mixed ms/us aggTrades timestamps to seconds (vectorized per-row autodetect).

    2026-05-16 perf fix: was a Python for-loop iterating every row (~2.4 sec/day
    on BTC's 1M+ rows). At workers=8 across 77 assets * 2327 days each, this
    dominated hawkes wall time (3-4 hours). np.where vectorization drops per-day
    cost ~50-100x, projected full-universe wall time 3-4hr -> 5-10 min.

    Per-row autodetect preserved: same parquet can have mixed ms+us after the
    2024 Binance scale transition, so we can't shortcut with a single-unit
    detection on a sample. np.where applies conditions per-element in C.

    Conditions identical to prior loop:
      t > 1e15  -> microseconds -> /1e6
      t > 1e12  -> milliseconds -> /1e3
      else      -> already seconds (or smaller; pass through)
    """
    ts = np.asarray(ts, dtype=np.float64)
    return np.where(ts > 1e15, ts / 1e6,
                    np.where(ts > 1e12, ts / 1e3, ts))


def branching_ratio_from_counts(N: np.ndarray) -> float:
    """Hardiman & Bouchaud-style eta from Var/Mean overdispersion.

    For an exponential Hawkes process with kernel alpha*exp(-beta*t) and branching
    ratio eta = alpha/beta < 1, coarse-grained counts satisfy:
      Var(N) / Mean(N) ~ 1 / (1 - eta)^2  (in the stationary high-rate limit)
    => eta = 1 - sqrt(Mean(N) / Var(N))

    Returns NaN if insufficient data or pathological dispersion.
    """
    if len(N) < 30:
        return float("nan")
    mu = float(np.mean(N))
    var = float(np.var(N, ddof=1))
    if mu <= 0 or var <= 0 or var < mu:
        # Sub-Poisson: not a Hawkes regime; eta -> 0
        return 0.0
    eta = 1.0 - np.sqrt(mu / var)
    return float(np.clip(eta, 0.0, 0.99))


def process_one_day(asset: str, fp: Path) -> dict | None:
    """Compute branching-ratio features for one (asset, day)."""
    try:
        # 2026-05-13: prepare_aggtrades for ts-scale + sort (handles 2026-Q1+ regressions).
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bars"))
        from _aggtrades_utils import prepare_aggtrades  # noqa: E402
        df_pl = pl.read_parquet(fp)
        # Detect the canonical ts column name first; prepare_aggtrades expects "timestamp"
        cols = df_pl.columns
        ts_canonical = "timestamp" if "timestamp" in cols else (
            "transact_time" if "transact_time" in cols else None)
        if ts_canonical and ts_canonical != "timestamp":
            df_pl = df_pl.rename({ts_canonical: "timestamp"})
        df_pl = prepare_aggtrades(df_pl, ts_col="timestamp")
        df = df_pl.to_pandas()
    except Exception as e:
        # Loud log so corrupt/partial aggTrades parquets surface in worker logs
        # (process_asset prints these via worker error handler).
        from progress import phase_log as _pl
        _pl("hawkes", "WARN", f"{asset}: hawkes_read_err {fp.name}: {type(e).__name__}: {str(e)[:80]}")
        return None
    if "transact_time" in df.columns:
        ts_col = "transact_time"
    elif "T" in df.columns:
        ts_col = "T"
    elif "time" in df.columns:
        ts_col = "time"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        return None
    if "is_buyer_maker" in df.columns:
        side_col = "is_buyer_maker"
    elif "m" in df.columns:
        side_col = "m"
    else:
        side_col = None

    ts = df[ts_col].to_numpy()
    if len(ts) < 100:
        return None
    secs = per_row_to_seconds(ts)
    t0 = secs.min()
    rel = secs - t0
    duration = rel.max()
    if duration < 60:
        return None
    n_bins = int(duration // BIN_SEC) + 1
    if n_bins < 30:
        return None

    bin_idx = (rel / BIN_SEC).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    counts_total = np.bincount(bin_idx, minlength=n_bins)
    eta_total = branching_ratio_from_counts(counts_total)

    eta_buy = eta_sell = float("nan")
    if side_col is not None:
        is_maker = df[side_col].to_numpy().astype(bool)
        # is_buyer_maker=True => seller is taker => sell-side aggressor
        sell_mask = is_maker
        buy_mask = ~is_maker
        counts_buy = np.bincount(bin_idx[buy_mask], minlength=n_bins) if buy_mask.any() else np.zeros(n_bins)
        counts_sell = np.bincount(bin_idx[sell_mask], minlength=n_bins) if sell_mask.any() else np.zeros(n_bins)
        eta_buy = branching_ratio_from_counts(counts_buy)
        eta_sell = branching_ratio_from_counts(counts_sell)

    eta_imb = (eta_buy - eta_sell) if (np.isfinite(eta_buy) and np.isfinite(eta_sell)) else float("nan")

    # date inference from filename (BTCUSDT-aggTrades-2024-01-01.parquet)
    name = fp.stem
    parts = name.split("-")
    if len(parts) >= 5:
        try:
            date_str = "-".join(parts[-3:])
            date = pd.Timestamp(date_str).date()
        except Exception:
            return None
    else:
        return None

    return {
        "date": date,
        "asset": asset,
        "eta_total": eta_total,
        "eta_buy": eta_buy,
        "eta_sell": eta_sell,
        "eta_imbalance": eta_imb,
        "n_trades": int(len(ts)),
    }


def _date_from_filename(fp: Path) -> "pd.Timestamp.date|None":
    """Infer trading date from filename like BTCUSDT-aggTrades-2024-01-01.parquet."""
    parts = fp.stem.split("-")
    if len(parts) < 5:
        return None
    try:
        return pd.Timestamp("-".join(parts[-3:])).date()
    except Exception:
        return None


def _load_existing_keys() -> set[tuple[str, "pd.Timestamp.date"]]:
    """Read latest dated panel (if any) and return the set of (asset, date) already done."""
    p = _resolve_existing_path()
    if p is None or not p.exists():
        return set()
    try:
        df = pl.read_parquet(p)
    except Exception:
        return set()
    if "asset" not in df.columns or "date" not in df.columns:
        return set()
    return set(zip(df["asset"].to_list(), df["date"].to_list()))


def process_asset(
    asset: str,
    max_days: int | None = None,
    done_keys: set | None = None,
) -> list[dict]:
    sym = f"{asset}USDT"
    asset_dir = RAW_DIR / sym / "aggTrades"
    if not asset_dir.exists():
        from progress import phase_log as _pl
        _pl("hawkes", "SKIP", f"{asset}: no aggTrades dir")
        return []
    fps = sorted(asset_dir.glob(f"{sym}-aggTrades-*.parquet"))
    if max_days is not None:
        fps = fps[-max_days:]

    if done_keys is None:
        done_keys = set()
    # Skip dates already in the existing parquet
    fps_to_run = []
    for fp in fps:
        d = _date_from_filename(fp)
        if d is None:
            continue
        if (asset, d) in done_keys:
            continue
        fps_to_run.append(fp)
    n_skip = len(fps) - len(fps_to_run)
    if n_skip:
        from progress import phase_log as _pl
        _pl("hawkes", "SCAN", f"{asset}: skipping {n_skip} already-done days; processing {len(fps_to_run)}")

    rows = []
    for i, fp in enumerate(fps_to_run):
        rec = process_one_day(asset, fp)
        if rec is not None:
            rows.append(rec)
        if (i + 1) % 100 == 0:
            from progress import phase_log as _pl
            _pl("hawkes", "BUILD", f"{asset}: processed", counters={"i": i+1, "N": len(fps_to_run)})
    from progress import phase_log as _pl
    _pl("hawkes", "OK", f"{asset}: done: {len(rows)} new valid days")
    return rows


def _process_asset_worker(args_tuple) -> tuple[str, list[dict], str | None]:
    """Top-level worker for ProcessPoolExecutor.

    Returns (asset, rows, error_msg_or_None). Memory/OS-level errors are
    re-raised so the orchestrator sees them and exits non-zero; per-asset
    compute errors are returned as the 3rd tuple element so the coverage
    report can honestly mark them ERR (not MISSING).
    """
    asset, max_days, done_keys = args_tuple
    try:
        rows = process_asset(asset, max_days=max_days, done_keys=done_keys)
        return (asset, rows, None)
    except (MemoryError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        from progress import phase_log as _pl
        _pl("hawkes", "FAIL", f"{asset}: worker error: {type(e).__name__}: {e}")
        return (asset, [], f"{type(e).__name__}: {e}")


def main(max_days_per_asset: int | None = None, force: bool = False,
         workers: int = 4, assets: list | None = None,
         universe: str | None = None) -> None:
    """Build the Hawkes branching panel. `assets` defaults to DEFAULT_ASSETS (u10).

    `universe` is informational only — used in the coverage report at the end.
    """
    if assets is None:
        assets = list(DEFAULT_ASSETS)
    # Make the resolved asset list available to downstream summary/log lines.
    global ASSETS
    ASSETS = list(assets)

    new_rows: list[dict] = []
    existing_path = _resolve_existing_path()

    # Corruption guard: before trusting existing panel for delta-resume,
    # validate schema + null rates on key features. If existing is corrupt
    # (the prior failure mode: feature columns silently null), force a
    # full rebuild with done_keys=set().
    forced_rebuild_due_to_corruption = False
    if not force and existing_path is not None and existing_path.exists():
        ok, why = validate_existing(
            existing_path,
            required_cols={"date", "asset", "eta_total", "eta_buy",
                            "eta_sell", "eta_imbalance"},
            max_null_rate={"eta_total": 0.05, "eta_imbalance": 0.05})
        if not ok:
            from progress import phase_log as _pl
            _pl("hawkes", "WARN", f"CORRUPT existing panel: {why}; "
                  f"forcing full rebuild", flush=True)
            forced_rebuild_due_to_corruption = True

    effective_force = force or forced_rebuild_due_to_corruption
    done_keys = set() if effective_force else _load_existing_keys()
    if done_keys and existing_path is not None:
        from progress import phase_log as _pl
        _pl("hawkes", "SCAN", f"resume: {len(done_keys)} (asset,date) pairs in {existing_path.name}")
    from progress import phase_log as _pl
    _pl("hawkes", "START", f"processing {len(ASSETS)} assets ({','.join(ASSETS[:5])}"
          f"{'...' if len(ASSETS) > 5 else ''}), max_days={max_days_per_asset}, "
          f"force={force}, workers={workers}")

    err_assets: dict[str, str] = {}
    if workers <= 1:
        for asset in ASSETS:
            try:
                rows = process_asset(asset, max_days=max_days_per_asset, done_keys=done_keys)
                new_rows.extend(rows)
            except (MemoryError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                _pl("hawkes", "FAIL", f"{asset}: worker error: {type(e).__name__}: {e}")
                err_assets[asset] = f"{type(e).__name__}: {e}"
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        worker_args = [(a, max_days_per_asset, done_keys) for a in ASSETS]
        with ProcessPoolExecutor(max_workers=min(workers, len(ASSETS))) as ex:
            futures = {ex.submit(_process_asset_worker, wa): wa[0] for wa in worker_args}
            for fut in as_completed(futures):
                asset_name = futures[fut]
                try:
                    _asset, rows, err = fut.result()
                except MemoryError as e:
                    err_assets[asset_name] = f"MemoryError: {e}"
                    _pl("hawkes", "FAIL", f"{asset_name}: MemoryError surfaced from worker; will exit non-zero")
                    continue
                new_rows.extend(rows)
                if err is not None:
                    err_assets[_asset] = err

    if not new_rows and not done_keys:
        _pl("hawkes", "FAIL", "HARD FAIL: no data processed and no resume "
              "state; check raw aggTrades availability", flush=True)
        _sys.exit(2)

    # Combine new with existing (if any). Effective_force = user --force OR
    # corruption-detected; either way means we discard existing rows.
    if effective_force or existing_path is None:
        df = pl.DataFrame(new_rows).sort(["asset", "date"]) if new_rows else None
    else:
        existing = pl.read_parquet(existing_path)
        if new_rows:
            new_df = pl.DataFrame(new_rows)
            df = pl.concat([existing, new_df], how="vertical_relaxed").unique(
                subset=["asset", "date"], keep="last"
            ).sort(["asset", "date"])
        else:
            df = existing
            _pl("hawkes", "SKIP", "no new days to process; output unchanged")

    if df is None or len(df) == 0:
        _pl("hawkes", "FAIL", "no data; aborting")
        return

    # Write to dated path (using max date in df) and gc older.
    if "date" in df.columns and len(df) > 0:
        d_max = df["date"].max()
        out_path = _layout.hawkes_panel_path(PANEL_NAME, d_max)
    else:
        out_path = _layout.hawkes_panel_path(PANEL_NAME)
    atomic_write_parquet(
        df, out_path,
        required_cols={"date", "asset", "eta_total", "eta_buy",
                        "eta_sell", "eta_imbalance"})
    # GC older snapshots only AFTER the new file passes column-name validation.
    _layout.gc_older_dated(_layout.hawkes_dir(), PANEL_NAME)
    _pl("hawkes", "WRITE", f"saved: {out_path} ({len(df)} rows; {len(new_rows)} new)")
    # Quick sanity: print mean eta per asset
    summary = df.group_by("asset").agg([
        pl.col("eta_total").mean().alias("mean_eta_total"),
        pl.col("eta_buy").mean().alias("mean_eta_buy"),
        pl.col("eta_sell").mean().alias("mean_eta_sell"),
        pl.col("eta_imbalance").mean().alias("mean_eta_imb"),
        pl.col("n_trades").median().alias("median_trades"),
        pl.len().alias("n_days"),
    ]).sort("asset")
    print("\n[hawkes_branching] per-asset summary:")
    print(summary.to_pandas().to_string(index=False))  # summary table — leave raw for readability

    # Coverage report (uniform across pipeline stages)
    try:
        _sys.path.insert(0, str(ROOT / "src" / "pipeline"))
        from coverage_report import print_coverage_report
        produced = set(str(a).upper() for a in df["asset"].unique().to_list())
        ok_set = produced & set(a.upper() for a in ASSETS)
        print_coverage_report(
            stage_name="hawkes_branching",
            universe=universe,
            expected_assets=ASSETS,
            ok_assets=ok_set,
            err_assets=set(err_assets.keys()),
            extra_lines=[f"Days written: {len(df)}; new this run: {len(new_rows)}",
                         f"Output: {out_path.name}"],
        )
    except Exception as e:
        _pl("hawkes", "WARN", f"coverage: {type(e).__name__}: {e}")

    # Honest-failure: any worker error => non-zero exit so refresh.py / CI
    # surface the data hole instead of treating partial output as success.
    if err_assets:
        _pl("hawkes", "FAIL",
            f"{len(err_assets)} asset(s) errored: {sorted(err_assets.keys())}; "
            f"first error: {next(iter(err_assets.values()))}")
        _sys.exit(2)


if __name__ == "__main__":
    import argparse
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-days", type=int, default=None,
                        help="Limit per asset (default: all available days)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute from scratch (default: resume from existing OUT_PATH)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Per-asset parallel workers (default 4). Each asset is "
                             "independent (per-day binning + over-dispersion calc).")
    parser.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                        help="Universe to build over (default: 10-asset hardcoded list). "
                             "Use u50 to align coverage with bar_fabric.")
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Explicit asset list (overrides --universe).")
    args = parser.parse_args()
    resolved_assets = _resolve_assets(args.universe, args.assets)
    main(max_days_per_asset=args.max_days, force=args.force, workers=args.workers,
         assets=resolved_assets, universe=args.universe)
