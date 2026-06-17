"""Frontier feature consolidator.

Reads all frontier sources from the registry, joins them per-asset to a single
daily table, and writes data/processed/frontier/<sym>usdt_frontier_daily_<DATE>.parquet.

This is the SILVER layer between raw frontier dumps and chimera v51 (post-2026-04-26 layout):
  Bronze: data/raw_external/<source>/* + data/processed/{hawkes,panels}/* (frontier panels)
  Silver: data/processed/frontier/<sym>usdt_frontier_daily_<DATE>.parquet  (per-asset, daily, unified)
  Gold:   data/processed/chimera/<sym>usdt_v51_chimera_<DATE>.parquet      (per-asset, dollar-bar, full join)

Why a separate silver layer (vs joining straight into chimera v51):
  1. Faster iteration -- updating a frontier source = rebuild silver, not chimera.
  2. Easier validation -- silver is one row per (asset, date), readable.
  3. Reusable -- daily-cadence strategies can use silver directly.

Run:
  python src/pipeline/frontier_consolidator.py                # all assets
  python src/pipeline/frontier_consolidator.py --asset BTC    # one asset
  python src/pipeline/frontier_consolidator.py --validate     # dry-run, report only
"""
from __future__ import annotations

# CDAP contract — declared after __future__ import per PEP-236.
__contract__ = {
    "kind": "pipeline_stage",
    "stage": "frontier_consolidate",
    "inputs": {
        "args": ["--asset", "--assets", "--universe {u10|u50|u100}", "--workers", "--force"],
        "upstream": [
            "data/processed/hawkes/daily/hawkes_branching_daily_*.parquet",
            "data/processed/panels/daily/*.parquet",
        ],
    },
    "outputs": {
        "files": "data/processed/frontier/daily/<sym>usdt_frontier_daily_*.parquet",
        "row_unit": "one row per (asset, date) for ~80 frontier features",
    },
    "invariants": {
        "atomic_write": True,
        "column_name_verify": True,
        "coverage_report_at_end": True,
        "asset_set_eq": "downstream:chimera_v51",
    },
    "rationale": "Silver layer; per-asset daily join of all panel features.",
}

import argparse
import re
from pathlib import Path

import polars as pl

from feature_registry import FeatureRegistry, SourceSpec
import sys as _sys
_pipe_dir = Path(__file__).resolve().parent
# G-AUDIT-023: insert(0,...) not append — late-bound modules can lose to
# same-named modules already on path. Insert ensures pipeline-local layout/
# feature_registry win.
if str(_pipe_dir) not in _sys.path:
    _sys.path.insert(0, str(_pipe_dir))
import layout as _layout
from parquet_io import atomic_write_parquet

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def normalize_date(df: pl.DataFrame, src: SourceSpec) -> pl.DataFrame:
    """Convert source date column to a uniform Date column named 'date'."""
    if src.date_col != "date":
        df = df.rename({src.date_col: "date"})
    if src.date_unit == "datetime":
        df = df.with_columns(pl.col("date").cast(pl.Date))
    return df


def load_per_asset(src: SourceSpec, asset: str) -> pl.DataFrame:
    """Load a per_asset source, filter to the requested asset, return Date + features (with prefix)."""
    df = pl.read_parquet(src.absolute_path())
    df = normalize_date(df, src)
    if src.asset_col is None:
        raise ValueError(f"per_asset source {src.name} missing asset_col")
    df = df.filter(pl.col(src.asset_col).str.to_uppercase() == asset.upper())
    cols = ["date"] + [c for c in src.features if c in df.columns]
    df = df.select(cols)
    rename = {c: f"{src.prefix}{c}" for c in src.features if c in df.columns}
    df = df.rename(rename)
    df = df.unique(subset=["date"], keep="last").sort("date")
    return df


def load_global(src: SourceSpec) -> pl.DataFrame:
    """Load a global source, return Date + features (with prefix)."""
    df = pl.read_parquet(src.absolute_path())
    df = normalize_date(df, src)
    cols = ["date"] + [c for c in src.features if c in df.columns]
    df = df.select(cols)
    rename = {c: f"{src.prefix}{c}" for c in src.features if c in df.columns}
    df = df.rename(rename)
    df = df.unique(subset=["date"], keep="last").sort("date")
    return df


def load_wide_per_asset(src: SourceSpec, asset: str) -> pl.DataFrame:
    """Extract a per-asset value from a wide schema (e.g., 'btc_fund' col -> BTC's value)."""
    df = pl.read_parquet(src.absolute_path())
    df = normalize_date(df, src)
    pat = re.compile(src.wide_pattern or "")
    asset_l = asset.lower()
    matched_col = None
    for c in df.columns:
        m = pat.match(c)
        if m and m.group(1).lower() == asset_l:
            matched_col = c
            break
    feature_name = f"{src.prefix}{src.feature_alias or 'value'}"
    if matched_col is None:
        # asset not in wide schema; return empty df with right columns
        return pl.DataFrame({"date": [], feature_name: []}, schema={"date": pl.Date, feature_name: pl.Float64})
    df = df.select(["date", matched_col]).rename({matched_col: feature_name})
    df = df.unique(subset=["date"], keep="last").sort("date")
    return df


def consolidate_one_asset(
    asset: str,
    registry: FeatureRegistry,
    out_path: Path | None = None,
    forward_fill_max_days: int = 7,
    skip_sources: set[str] | None = None,
) -> pl.DataFrame:
    """Build a per-asset daily silver table joining all frontier sources.

    `skip_sources` (optional) is a set of source names known to be absent on
    disk; pass it to avoid emitting per-asset WARN/stack-trace noise when the
    caller has already catalogued missing sources at startup.
    """
    asset_u = asset.upper()
    base_date_range = None
    parts: list[pl.DataFrame] = []

    skip_sources = skip_sources or set()
    for src_name in registry.get_chimera_join_order():
        if src_name in skip_sources:
            continue
        src = registry.get_source(src_name)
        # Pre-existence check: avoids polars emitting a multi-line "context
        # stack" error for missing parquets. If the file was newly removed
        # since the startup pre-check, fall through to the except below.
        if not src.absolute_path().exists():
            continue
        # 2026-05-24 fix: narrowed catch from bare Exception. The prior
        # `except Exception` silently swallowed schema bugs (KeyError,
        # AttributeError) as well as legitimate missing-file races. Now
        # only file-IO / parquet-shape errors are tolerated; structural
        # bugs propagate so they get fixed instead of vanishing into a
        # WARN line that nobody reads.
        try:
            if src.layout == "per_asset":
                part = load_per_asset(src, asset_u)
            elif src.layout == "global":
                part = load_global(src)
            elif src.layout == "wide_per_asset":
                part = load_wide_per_asset(src, asset_u)
            else:
                continue
        except (FileNotFoundError, IsADirectoryError, PermissionError, OSError) as e:
            from progress import phase_log as _pl
            _pl("frontier", "WARN", f"{asset_u}: load failed for {src_name} (IO): {e}")
            continue
        except pl.exceptions.ComputeError as e:
            # Polars schema/compute errors at parquet read -- tolerable
            # if the file is mid-write or column-shape changed upstream.
            from progress import phase_log as _pl
            _pl("frontier", "WARN", f"{asset_u}: load failed for {src_name} (compute): {e}")
            continue
        # NOTE: KeyError, AttributeError, ValueError, TypeError NOT caught --
        # those indicate code bugs in the loader and MUST surface.
        parts.append(part)

    if not parts:
        return pl.DataFrame()

    # Establish date range = union of all source dates
    all_dates = pl.concat([p.select("date") for p in parts]).unique().sort("date")
    if len(all_dates) == 0:
        return pl.DataFrame()

    # Build a contiguous daily index from min to max
    min_d = all_dates["date"].min()
    max_d = all_dates["date"].max()
    contiguous = pl.DataFrame({
        "date": pl.date_range(min_d, max_d, interval="1d", eager=True),
    })

    # Left-join each part onto contiguous date index
    out = contiguous
    for part in parts:
        out = out.join(part, on="date", how="left")

    # Forward-fill with a BOUNDED gap. The date index is contiguous daily (one
    # row per calendar day), so forward_fill(limit=N) caps the fill to N calendar
    # days. Beyond that the value stays null -- a dead/stale source surfaces as
    # NaN instead of silently propagating a weeks-old value into the gold chimera.
    # (Previously this did an UNBOUNDED forward_fill, ignoring forward_fill_max_days
    # entirely -- the registry's max-staleness contract was a no-op.)
    if forward_fill_max_days > 0:
        out = out.with_columns([
            pl.col(c).forward_fill(limit=forward_fill_max_days).alias(c)
            for c in out.columns if c != "date"
        ])

    # Add asset col
    out = out.with_columns(pl.lit(asset_u).alias("asset"))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic (was a bare write_parquet -> a crash mid-write left a corrupt
        # silver/.tmp file). atomic_write_parquet handles tmp+os.replace.
        atomic_write_parquet(out, out_path)
        n_features = len(out.columns) - 2  # excl date + asset
        from progress import phase_log as _pl
        _pl("frontier", "WRITE", f"{asset_u}: wrote {out_path.name}: {len(out)} rows, {n_features} features")
    return out


def _consolidate_one(asset: str) -> dict:
    """Top-level worker: consolidate a single asset. Used by ProcessPoolExecutor."""
    try:
        reg = FeatureRegistry.load()
        from datetime import datetime as _dt, timezone as _tz
        # Pass out_path=None so consolidate_one_asset returns the df without
        # writing; we use atomic_write_parquet for the write+verify+rename.
        df_out = consolidate_one_asset(asset, reg, out_path=None,
                                       forward_fill_max_days=reg.chimera.forward_fill_max_days)
        if df_out is None or len(df_out) == 0:
            return {"asset": asset, "status": "no_silver"}
        if "date" in df_out.columns:
            d_max = df_out["date"].max()
        else:
            d_max = _dt.now(_tz.utc).date()
        dst = _layout.frontier_daily_path(asset, d_max)
        atomic_write_parquet(df_out, dst, required_cols={"date", "asset"})
        # GC older snapshots only AFTER the new file passes column-name validation.
        _layout.gc_older_dated(_layout.frontier_dir(), f"{asset.lower()}usdt_frontier_daily")
        return {"asset": asset, "status": "ok", "rows": len(df_out),
                "n_features": len(df_out.columns) - 2}
    except Exception as e:
        return {"asset": asset, "status": "error", "err": f"{type(e).__name__}: {e}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None, help="Single asset (e.g. BTC). Deprecated alias for --assets [SYM].")
    # 2026-05-21 contract retrofit: --assets plural added.
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Asset list (BTC or BTCUSDT format). Overrides --universe.")
    parser.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                        help="Restrict consolidation to a universe (overrides raw discovery).")
    parser.add_argument("--validate", action="store_true", help="Just validate registry, no writes.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Per-asset parallel workers (default 4). Each worker is "
                             "lightweight (just panel joins; no polars heavy ops), so "
                             "this stage parallelizes cleanly. Pass 1 to disable.")
    parser.add_argument("--force", action="store_true",
                        help="Force fresh rebuild: delete prior dated frontier silver "
                             "snapshots in frontier/daily/ for the resolved universe "
                             "before rebuild. (frontier_consolidator already overwrites "
                             "by default; this adds explicit prior-snapshot deletion.)")
    # Phase 7 bidirectional pattern
    parser.add_argument("-r", "--reverse", action="store_true",
                        help="Reverse asset iteration (Z->A) for meet-in-middle "
                             "2x speedup. Run two terminals: one without -r, one with.")
    args = parser.parse_args()

    reg = FeatureRegistry.load()
    from progress import phase_log as _pl
    _pl("frontier", "SCAN", f"FeatureRegistry v{reg.version}: {len(reg.sources)} sources, {len(reg.list_features())} features")

    # One-shot missing-source check: catalogue declared-but-absent sources at
    # the top so we don't spam per-asset WARN with polars stack traces from
    # the inner load_*. Pass the missing set down to consolidate_one_asset
    # which will silently skip them.
    _missing_sources: set[str] = set()
    for _src_name in reg.get_chimera_join_order():
        _src = reg.get_source(_src_name)
        if not _src.absolute_path().exists():
            _missing_sources.add(_src_name)
    if _missing_sources:
        _pl("frontier", "WARN", f"{len(_missing_sources)} declared sources missing on disk "
              f"(silently skipped per-asset):")
        for _n in sorted(_missing_sources):
            _src = reg.get_source(_n)
            print(f"  - {_n}: {_src.absolute_path().relative_to(PROJECT_ROOT)} "
                  f"({len(_src.features)} features will be NaN in chimera)")

    if args.validate:
        msgs = reg.validate_against_disk()
        if msgs:
            print("Validation issues:")
            for m in msgs:
                print(f"  {m}")
        else:
            print("Registry validates clean against disk.")
        return

    # 2026-05-21 contract retrofit: resolve via --assets > --asset > --universe > raw-dir
    if args.assets:
        assets = [a.upper().replace("USDT", "") for a in args.assets]
        _pl("frontier", "SCAN", f"universe: --assets ({len(assets)} explicit)")
    elif args.asset:
        assets = [args.asset.upper().replace("USDT", "")]
        _pl("frontier", "SCAN", f"universe: --asset ({assets[0]})")
    elif args.universe:
        try:
            from universe_loader import UniverseLoader as _UL
            syms = _UL.load().list(args.universe)
            assets = [s.replace("USDT", "").upper() for s in syms]
            _pl("frontier", "SCAN", f"Universe {args.universe}: {len(assets)} assets")
        except Exception as e:
            _pl("frontier", "WARN", f"universe={args.universe} load failed ({e}); falling back to raw discovery")
            args.universe = None
    if not args.asset and not args.universe and not args.assets:
        # Discover from raw data (works pre-chimera-build).
        # Falls back to v50-chimera list if raw is unavailable.
        raw_dir = PROJECT_ROOT / "data" / "raw"
        if raw_dir.exists():
            assets = sorted({d.name.replace("USDT", "").upper()
                              for d in raw_dir.iterdir()
                              if d.is_dir() and d.name.upper().endswith("USDT")})
        else:
            assets = [s.replace("USDT", "") for s in _layout.list_v50_assets()]

    # @browser B1: --force LOUD; delete prior dated snapshots
    if args.force:
        n_deleted = 0
        d = _layout.frontier_dir()
        if d.exists():
            for asset in assets:
                sym_l = asset.lower().replace("USDT", "")
                for old in d.glob(f"{sym_l}usdt_frontier_daily*.parquet"):
                    try:
                        old.unlink()
                        n_deleted += 1
                    except Exception:
                        pass
        _pl("frontier", "SKIP", f"FORCE deleted {n_deleted} prior frontier silver snapshots before rebuild")

    # 2026-05-21 contract retrofit: pre-flight skip-existing.
    # Skip assets whose latest frontier_daily snapshot is dated >= today (UTC).
    # @browser B1: skip is LOUD; --force overrides.
    if not args.force and assets:
        from datetime import datetime as _dt2, timezone as _tz2
        _today = _dt2.now(_tz2.utc).date()
        _fdir = _layout.frontier_dir()
        _skipped = []
        _keep = []
        for asset in assets:
            sym_l = asset.lower().replace("USDT", "")
            existing_stamps = []
            if _fdir.exists():
                for f in _fdir.glob(f"{sym_l}usdt_frontier_daily*.parquet"):
                    stem_parts = f.stem.split("_")
                    if stem_parts and len(stem_parts[-1]) == 8:
                        try:
                            existing_stamps.append(_dt2.strptime(stem_parts[-1], "%Y%m%d").date())
                        except ValueError:
                            pass
            if existing_stamps and max(existing_stamps) >= _today:
                _skipped.append(asset)
            else:
                _keep.append(asset)
        if _skipped:
            _pl("frontier", "SKIP", f"skip-existing: {len(_skipped)} assets fresh (stamp >= {_today}); "
                  f"--force to rebuild", flush=True)
        assets = _keep
        if not assets:
            _pl("frontier", "SKIP", "all assets already fresh; nothing to do.")
            return

    # Phase 7 bidirectional: reverse asset list if requested
    if args.reverse and assets:
        assets = list(reversed(assets))
        _pl("frontier", "SCAN", f"REVERSE mode: iterating {len(assets)} assets Z->A "
              f"(meet-in-middle pattern)", flush=True)

    # Phase 8: centralized listing_dates marker. frontier_silver consumes
    # per-asset chimera_legacy + panel-source outputs that already
    # self-filter to post-listing dates. Marker for consumer crawler.
    try:
        import sys as _ld_sys
        from pathlib import Path as _ld_Path
        _ld_sys.path.insert(0, str(_ld_Path(__file__).resolve().parents[1]))
        from pipeline.listing_dates import is_pre_listing as _is_pre_listing  # noqa: F401
    except ImportError:
        pass

    _pl("frontier", "START", f"Consolidating frontier features for {len(assets)} assets (workers={args.workers})")
    _layout.frontier_dir().mkdir(parents=True, exist_ok=True)

    import time as _time
    t0 = _time.time()
    ok_set: set = set()
    err_set: set = set()
    skip_set: set = set()
    if args.workers <= 1:
        for asset in assets:
            r = _consolidate_one(asset)
            if r["status"] == "ok":
                ok_set.add(r["asset"].upper())
                _pl("frontier", "OK", f"{r['asset']}: {r['rows']}r {r['n_features']} features")
            elif r["status"] == "no_silver":
                skip_set.add(r["asset"].upper())
                _pl("frontier", "FAIL", f"{r['asset']}: {r['status']} {r.get('err','')}")
            else:
                err_set.add(r["asset"].upper())
                _pl("frontier", "FAIL", f"{r['asset']}: {r['status']} {r.get('err','')}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_consolidate_one, a): a for a in assets}
            for fut in as_completed(futures):
                a = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"asset": a, "status": "error", "err": str(e)}
                if r["status"] == "ok":
                    ok_set.add(r["asset"].upper())
                    _pl("frontier", "OK", f"{r['asset']}: {r['rows']}r {r['n_features']} features")
                elif r["status"] == "no_silver":
                    skip_set.add(r["asset"].upper())
                    _pl("frontier", "FAIL", f"{r['asset']}: {r['status']} {r.get('err','')}")
                else:
                    err_set.add(r["asset"].upper())
                    _pl("frontier", "FAIL", f"{r['asset']}: {r['status']} {r.get('err','')}")

    _pl("frontier", "DONE", f"{len(assets)} silver files at {_layout.frontier_dir()}/", counters={"elapsed": _time.time()-t0})

    # Coverage report (uniform across pipeline stages)
    try:
        from coverage_report import print_coverage_report
        print_coverage_report(
            stage_name="frontier_consolidate",
            universe=args.universe,
            expected_assets=assets,
            ok_assets=ok_set,
            err_assets=err_set,
            skipped_assets=skip_set,
        )
    except Exception as e:
        _pl("frontier", "WARN", f"coverage: {type(e).__name__}: {e}")

    # B1 no-silent-failure: if every asset failed, refresh.py / pre_train_gate
    # should see this as a real failure and not propagate to downstream stages.
    if assets and not ok_set:
        _pl("frontier", "FAIL", f"frontier_consolidate: 0/{len(assets)} assets succeeded; aborting non-zero")
        import sys
        sys.exit(2)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
