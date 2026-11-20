"""V51 v2 -- SOTA chimera builder. Fixes all V50 issues + new SOTA layout.

Per docs/V50_TO_V51_FIXES.md:
  - tick_seq column (within-ms tiebreaker for zero-diff timestamps)
  - target_return_<h>_raw (uncapped) alongside _clipped (BC w/ v50)
  - returns_clean recomputed without fill_null(0) silent fill
  - inline universe membership flags (is_u10/u50/u100)
  - inline asset_dna column
  - per-asset manifest with checksums + lineage
  - multi-cadence: 1d/4h/1h/15m + main dollar bars
  - reads from new SOTA layout (data/raw_external/, data/features/_global/)
  - writes to new SOTA layout (data/processed/<SYMBOL>/v51.parquet,
    data/_manifests/v51_<ASSET>.json)

Usage:
  python src/pipeline/make_dataset.py                # all assets in registry
  python src/pipeline/make_dataset.py --asset BTC    # one asset
  python src/pipeline/make_dataset.py --universe u10 # only u10 assets
  python src/pipeline/make_dataset.py --no-cadence   # skip 1d/4h/1h/15m
  python src/pipeline/make_dataset.py --skip-silver  # use cached silver

Idempotent: re-running on already-built assets is fast (skip-silver --> no-op).
"""
from __future__ import annotations

# CDAP contract
__contract__ = {
    "kind": "pipeline_stage",
    "stage": "chimera_v51",
    "inputs": {
        "args": ["--asset", "--assets", "--universe {u10|u50|u100}", "--workers",
                 "--skip-silver", "--no-cadence", "--force"],
        "upstream": [
            "data/processed/chimera_legacy/dollar/<sym>usdt_v50_chimera_*.parquet",
            "data/processed/frontier/daily/<sym>usdt_frontier_daily_*.parquet",
        ],
    },
    "outputs": {
        "files": "data/processed/chimera/{dollar,1d,4h,1h,30m,15m}/<sym>usdt_v51_chimera*_*.parquet",
        "manifests": "data/manifests/v51_<SYM>.json",
        "expected_columns": 154,
    },
    "invariants": {
        "atomic_write": True,
        "column_name_verify": True,
        "coverage_report_at_end": True,
    },
    "rationale": "Gold layer for V0/V1.x training and ChimeraLoader.",
}

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from feature_registry import FeatureRegistry  # noqa: E402
from frontier_consolidator import consolidate_one_asset  # noqa: E402
from universe_loader import UniverseLoader  # noqa: E402
import layout as _layout  # noqa: E402
from parquet_io import atomic_write_parquet  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CHIMERA_VERSION = "v51_v2"


def _git_sha() -> str | None:
    """Current git HEAD short SHA for build reproducibility (None if unavailable)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(PROJECT_ROOT), capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def file_sha256(path: Path, chunk: int = 1024 * 1024) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def add_tick_seq(df: pl.DataFrame) -> pl.DataFrame:
    """V50 fix #1: assign within-ms sequence index for zero-diff timestamps."""
    if "timestamp" not in df.columns:
        return df
    return df.with_columns(
        pl.col("timestamp").cum_count().over("timestamp").sub(1).cast(pl.Int32).alias("tick_seq")
    )


def add_clean_returns(df: pl.DataFrame) -> pl.DataFrame:
    """V50 fix #3: returns_clean = pct_change WITHOUT fill_null(0). NaN preserved."""
    if "close" not in df.columns:
        return df
    return df.with_columns(
        pl.col("close").pct_change().alias("returns_clean")
    )


def add_raw_targets(df: pl.DataFrame) -> pl.DataFrame:
    """V50 fix #4: add target_return_<h>_raw (uncapped) alongside the v50 clipped versions."""
    if "close" not in df.columns:
        return df
    cols = []
    for h in (1, 4, 16, 64):
        cols.append(
            ((pl.col("close").shift(-h) - pl.col("close")) / (pl.col("close") + 1e-9))
            .alias(f"target_return_{h}_raw")
        )
    return df.with_columns(cols)


def add_metadata_cols(df: pl.DataFrame, asset: str, loader: UniverseLoader) -> pl.DataFrame:
    """NEW: inline universe membership + DNA bucket."""
    asset_u = asset.upper()
    is_u10 = loader.is_in(asset_u, "u10")
    is_u50 = loader.is_in(asset_u, "u50")
    is_u100 = loader.is_in(asset_u, "u100")
    dna = loader.dna_for(asset_u)
    return df.with_columns([
        pl.lit(is_u10).alias("is_u10"),
        pl.lit(is_u50).alias("is_u50"),
        pl.lit(is_u100).alias("is_u100"),
        pl.lit(dna).alias("asset_dna"),
    ])


def attach_frontier(chim: pl.DataFrame, silver: pl.DataFrame) -> pl.DataFrame:
    if "date" not in chim.columns:
        chim = chim.with_columns([
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date().alias("date"),
        ])
    silver_no_asset = silver.drop("asset") if "asset" in silver.columns else silver
    # LAG-1 join (2026-05-24 MAXX audit finding): silver row stamped date=D
    # is an aggregation over the FULL day D (00:00-23:59 UTC) for every
    # daily panel — s3_metrics, liq_features, etf, multi_venue, lob_proxy,
    # hawkes, rv_jump, basis, te_panel, whale_activity, AND cross-sectional
    # z-scores computed within those panels at day D. Joining same-date to
    # a sub-daily bar at D 09:00 makes 09:01-23:59 information visible at
    # 09:00 — intra-day publication race. Validator MI flag fires on SOL's
    # s3_oi_usd, liq_long_usd, etc. (2026-05-23 run). Shift +1 day so day-D
    # silver becomes visible only at bars on D+1 (as-of yesterday EOD).
    # Earliest day per asset -> NULL silver cols (downstream tolerant).
    # Note: add_xrel_features.py (commit 908a778) separately shifts xrel_*
    # cols +1 day. Net effect: xrel cols are joined at +2 (chimera at D
    # has silver from D-1, xrel computed from those, joined at +1 -> D-2).
    # That's acceptable extra staleness on xrel; the lookahead-free property
    # is what matters.
    silver_no_asset = silver_no_asset.with_columns(
        (pl.col("date") + pl.duration(days=1)).cast(chim.schema["date"])
    )
    return chim.join(silver_no_asset, on="date", how="left")


def attach_bargrain(chim: pl.DataFrame, sym: str) -> pl.DataFrame:
    """T2-A surgical Phase 2 (+lob expansion): as-of-backward bar-level join.

    Iterates over registered bar-grain panel families. Each chimera dollar
    bar is paired with the LATEST raw row strictly BEFORE its timestamp;
    tolerance 30 min = if no row within 30min, leave NaN.

    Registered families (2026-05-24):
      - bd  : bd_bgf_imbalance_l1     (Phase 1 lift 3.74x  GREEN)
      - lob : lob_bgf_* (5 features)  (Phase 1 mean lift 10.83x  GREEN,
                                         kyle_lambda alone 28.9x)

    Per Phase 1 result files:
      runs/oracle/bd_bar_grain_phase1_btc_multiperm_RESULT.txt
      runs/oracle/lob_bar_grain_phase1_btc_multiperm_RESULT.txt

    No-op (return chim unchanged for that family) if the panel file is
    missing for this asset -- validator's SPARSE_BY_DESIGN exemption on
    {bd_bgf_, lob_bgf_} covers the resulting all-NaN cols on assets that
    pre-date the panel's raw data window.
    """
    out = chim
    for family in ("bd", "lob"):
        bgf_path = (PROJECT_ROOT / "data" / "processed" / "bar_grain" / family
                    / f"{sym.upper()}_bgf.parquet")
        if not bgf_path.exists():
            continue
        bgf = pl.read_parquet(bgf_path).sort("timestamp")
        # as-of-backward: each bar sees the latest snap BEFORE its ts;
        # tolerance caps staleness at 30 minutes.
        out = out.sort("timestamp").join_asof(
            bgf, on="timestamp", strategy="backward",
            tolerance=30 * 60 * 1000,
        )
    return out


def materialize_cadence(df: pl.DataFrame, cadence: str) -> pl.DataFrame:
    df = df.with_columns([
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt"),
    ])
    if cadence == "1d":
        df = df.with_columns(pl.col("dt").dt.date().alias("ck"))
    elif cadence == "4h":
        df = df.with_columns(pl.col("dt").dt.truncate("4h").alias("ck"))
    elif cadence == "1h":
        df = df.with_columns(pl.col("dt").dt.truncate("1h").alias("ck"))
    elif cadence == "30m":
        df = df.with_columns(pl.col("dt").dt.truncate("30m").alias("ck"))
    elif cadence == "15m":
        df = df.with_columns(pl.col("dt").dt.truncate("15m").alias("ck"))
    elif cadence == "5m":
        df = df.with_columns(pl.col("dt").dt.truncate("5m").alias("ck"))
    elif cadence == "1m":
        df = df.with_columns(pl.col("dt").dt.truncate("1m").alias("ck"))
    else:
        raise ValueError(f"unsupported cadence: {cadence}")
    df = df.sort(["ck", "timestamp", "tick_seq"]).group_by("ck", maintain_order=True).last()
    return df.drop(["ck", "dt"])


def recompute_cadence_targets(df: pl.DataFrame) -> pl.DataFrame:
    """Replace dollar-grain target_return_<h>* with cadence-grain forward returns.

    2026-05-24 fix (CRITICAL): After materialize_cadence() picks the LAST
    dollar bar of each cadence bucket, the carried-over target_return_<h>
    columns still mean "next h DOLLAR bars" (~30-40s/bar), NOT "next h
    cadence bars". A 4h sleeve ranking on target_return_1 was using a
    ~30-second forward return as its label -- silent semantic mismatch
    that masquerades as legitimate IC.

    This fn recomputes BOTH target_return_<h> (V50 clipped) and
    target_return_<h>_raw (uncapped) at cadence grain. Same clip
    thresholds as sota_shared_logic_v50 ({h=1: ±0.15, h>=4: ±0.50}); the
    raw column has no clip so 4h+ horizons keep legitimate tail returns
    that would otherwise be truncated.

    Applied in build_one after materialize_cadence. Dollar-grain output
    is NOT affected -- its targets were already correct.
    """
    if "close" not in df.columns:
        return df
    cols = []
    for h in (1, 4, 16, 64):
        fwd = ((pl.col("close").shift(-h) - pl.col("close"))
               / (pl.col("close") + 1e-9))
        clip_hi = 0.15 if h == 1 else 0.50
        cols.append(fwd.clip(-clip_hi, clip_hi).alias(f"target_return_{h}"))
        cols.append(fwd.alias(f"target_return_{h}_raw"))
    return df.with_columns(cols)


def build_one(
    asset: str,
    registry: FeatureRegistry,
    loader: UniverseLoader,
    skip_silver: bool = False,
    cadences: list[str] | None = None,
    write: bool = True,
) -> dict:
    asset_u = asset.upper()
    if asset_u.endswith("USDT"):
        asset_u = asset_u[:-4]                 # strip USDT suffix; "BTCUSDT" -> "BTC"
    asset_l = asset_u.lower()                  # canonical lowercase root: "btc"
    sym = f"{asset_u}USDT"                     # canonical sym: "BTCUSDT"
    asset_for_silver = asset_u                 # frontier_consolidator expects 'BTC' not 'BTCUSDT'

    # Sources (post-2026-04-26 layout v3 = cadence subfolders)
    v50_path = _layout.chimera_v50_latest(sym)   # may be None if missing
    silver_path = _layout.frontier_daily_path(sym)  # undated canonical path for write target
    _layout.frontier_dir().mkdir(parents=True, exist_ok=True)
    for _cad in _layout.V51_CADENCES:
        _layout.chimera_dir(_cad).mkdir(parents=True, exist_ok=True)
    _layout.DIR_MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifest_path = _layout.manifest_path(sym)

    if v50_path is None or not v50_path.exists():
        return {"asset": asset_u, "status": "skip_no_v50"}

    t0 = time.time()

    # Step 1: silver (frontier_daily.parquet) — read latest dated, write new dated
    silver_latest = _layout.frontier_daily_latest(sym)
    # 2026-05-20 RED-team HIGH 2: defensive guard. If --skip-silver was passed
    # (DAG-injected) but silver is missing, fail loud instead of falling through
    # to silver-rebuild. The DAG declared frontier_silver as a hard dep of
    # chimera_v51; missing silver here = upstream dep failure that the runner
    # missed. Fail-fast prevents chimera_v51 from masking the issue.
    if skip_silver and silver_latest is None:
        return {"asset": asset_u, "status": "missing_silver_under_skip_flag",
                "msg": f"--skip-silver set but no frontier_daily silver found for {sym}; "
                       f"upstream frontier_silver stage must have failed"}
    if skip_silver and silver_latest is not None:
        silver = pl.read_parquet(silver_latest)
    else:
        # Build to undated tmp path then rename to dated based on silver's last date
        tmp_silver = _layout.frontier_dir() / f"{asset_l}usdt_frontier_daily.tmp.parquet"
        silver = consolidate_one_asset(
            asset_for_silver, registry, out_path=tmp_silver,
            forward_fill_max_days=registry.chimera.forward_fill_max_days,
        )
        if silver is None or len(silver) == 0:
            tmp_silver.unlink(missing_ok=True)
            return {"asset": asset_u, "status": "no_silver"}
        # Compute date suffix from silver's max date
        if "date" in silver.columns:
            d_max = silver["date"].max()
            silver_date = d_max if hasattr(d_max, "year") else None
        else:
            silver_date = None
        if silver_date is None:
            silver_date = datetime.now(timezone.utc).date()
        silver_dst = _layout.frontier_daily_path(sym, silver_date)
        if silver_dst.exists():
            silver_dst.unlink()
        tmp_silver.rename(silver_dst)
        # GC deferred to end of function (after main chimera writes).
        # Premature GC here would delete the older healthy silver if a
        # downstream step fails, leaving no fallback for the runtime loader.
        silver_latest = silver_dst

    # Step 2: v50 base
    chim = pl.read_parquet(v50_path)
    n_v50 = len(chim.columns)

    # Step 3: V50 fixes
    chim = add_tick_seq(chim)              # tick_seq for ts dedup
    chim = add_clean_returns(chim)         # returns_clean (no silent fill_null)
    chim = add_raw_targets(chim)           # target_return_<h>_raw (uncapped)
    chim = add_metadata_cols(chim, sym, loader)  # is_u10/u50/u100, asset_dna -- use FULL symbol

    # Step 4: frontier join (daily silver +1d lag)
    chim = attach_frontier(chim, silver)
    # Step 4b: bar-grain BGF join (T2-A surgical Phase 2, 2026-05-24)
    # No-op for assets without a bd_bar_grain panel; adds column only when
    # the source panel exists.
    chim = attach_bargrain(chim, sym)
    n_chim = len(chim.columns)

    # Compute v51 date suffix from main chimera's max timestamp
    if "timestamp" in chim.columns:
        ts_max = chim["timestamp"].max()
        v51_date = datetime.fromtimestamp(ts_max / 1000.0, tz=timezone.utc).date()
    else:
        v51_date = datetime.now(timezone.utc).date()

    # NaN guard on join-critical cols (post-join, pre-write).
    # If silver join misaligned dates, xd_*/feature cols can be 100% NaN
    # while still PRESENT in the schema -- presence-only checks would pass.
    # Hard threshold: >25% null on any chimera-feature col = silent join failure.
    _nan_check_cols = [c for c in (
        "norm_flow_imbalance", "norm_hawkes_imbalance",
        "xd_btc_return", "xd_funding_spread", "xd_momentum_rank",
        "target_return_1", "target_return_4",
    ) if c in chim.columns]
    if _nan_check_cols:
        n_rows = max(1, len(chim))
        null_rates = chim.select([
            (pl.col(c).is_null().sum() / n_rows).alias(c) for c in _nan_check_cols
        ]).row(0, named=True)
        bad = {c: round(r, 4) for c, r in null_rates.items() if r > 0.25}
        if bad:
            raise RuntimeError(
                f"{asset_u}: v51 join produced >25% NaN on critical cols {bad} "
                f"(silver/xd misalignment). Refusing to write stale chimera."
            )

    # Step 5: cadence views
    cadence_outputs = {}
    cadence_paths = []  # collect for end-of-function GC
    if cadences:
        for cad in cadences:
            cad_df = materialize_cadence(chim, cad)
            # 2026-05-24 fix: recompute target_return_<h>* at cadence grain.
            # Pre-fix: cadence files carried dollar-bar targets (~30s ahead),
            # so a 4h sleeve scoring on target_return_1 used a ~30s label.
            cad_df = recompute_cadence_targets(cad_df)
            cad_path = _layout.chimera_v51_path(sym, cad, v51_date)
            if write:
                # Atomic write + col-presence verify in one call (G-AUDIT-020).
                atomic_write_parquet(
                    cad_df, cad_path,
                    required_cols={"timestamp", "target_return_1", "target_return_4"},
                )
                cadence_paths.append((cad, cad_path))
            cadence_outputs[cad] = {"rows": len(cad_df), "path": cad_path.name}

    # Step 6: write main (atomic + col-verify)
    chim_path = _layout.chimera_v51_path(sym, "dollar", v51_date)
    if write:
        atomic_write_parquet(
            chim, chim_path,
            required_cols={
                "timestamp", "bar_id", "close",
                "target_return_1", "target_return_4", "target_return_16", "target_return_64",
                # legacy 41 invariants (must survive the v50->v51 carry)
                "norm_flow_imbalance", "norm_hawkes_imbalance",
                "xd_btc_return", "xd_funding_spread", "xd_momentum_rank",
            },
        )

    # Step 7: manifest
    manifest = {
        "asset": asset_u,
        "symbol": sym,
        "chimera_version": CHIMERA_VERSION,
        "git_sha": _git_sha(),
        "build_time_utc": datetime.now(timezone.utc).isoformat(),
        "v51_path": str(chim_path.relative_to(PROJECT_ROOT)),
        "row_count": len(chim),
        "column_count": n_chim,
        "v50_input_sha256": file_sha256(v50_path),
        "v50_input_path": str(v50_path.relative_to(PROJECT_ROOT)),
        "silver_input_sha256": file_sha256(silver_latest) if silver_latest else None,
        "silver_input_path": str(silver_latest.relative_to(PROJECT_ROOT)) if silver_latest else None,
        "feature_registry_version": registry.version,
        "universe_membership": {
            "u10": bool(loader.is_in(sym, "u10")),
            "u50": bool(loader.is_in(sym, "u50")),
            "u100": bool(loader.is_in(sym, "u100")),
        },
        "asset_dna": loader.dna_for(sym),
        "cadence_views": cadence_outputs,
        "fixes_applied": [
            "tick_seq_for_ts_dedup",
            "returns_clean_no_fill_null",
            "target_return_h_raw_uncapped",
            "metadata_cols_inline",
        ],
    }
    if write:
        # Atomic manifest write: a crash between the chimera write and a bare
        # manifest write left a valid chimera with a stale/absent manifest, which
        # the freshness logic then misread. tmp + os.replace closes that window.
        _mtmp = manifest_path.with_suffix(".json.tmp")
        _mtmp.write_text(json.dumps(manifest, indent=2))
        os.replace(str(_mtmp), str(manifest_path))

        # ALL writes succeeded → deferred GC of older snapshots is now safe.
        # If any earlier step had raised, this code is unreachable and older
        # snapshots remain on disk as a fallback for the runtime loader.
        if silver_latest is not None:
            _layout.gc_older_dated(_layout.frontier_dir(),
                                    f"{asset_l}usdt_frontier_daily")
        for cad, _cp in cadence_paths:
            _layout.gc_older_dated(_layout.chimera_dir(cad),
                                    f"{asset_l}usdt_v51_chimera_{cad}")
        _layout.gc_older_dated(_layout.chimera_dir("dollar"),
                                f"{asset_l}usdt_v51_chimera")

    return {
        "asset": asset_u,
        "status": "ok",
        "v50_cols": n_v50,
        "v51_cols": n_chim,
        "added": n_chim - n_v50,
        "v51_rows": len(chim),
        "elapsed_s": round(time.time() - t0, 1),
        "cadence": cadence_outputs,
        "chim_path": str(chim_path.relative_to(PROJECT_ROOT)),
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
    }


def _build_one_worker(args_tuple) -> dict:
    """Top-level worker for ProcessPoolExecutor (must be picklable).

    Re-loads registry + universe in each worker since multiprocessing on Windows
    spawns fresh interpreters. Returns the same result dict as build_one().

    Sets POLARS_MAX_THREADS to (cpu_count // n_workers) so total in-flight
    threads = cpu_count (no oversubscription) but each worker still gets
    real intra-asset parallelism. Avoids both thread thrash AND
    overthrottling.
    """
    sym, skip_silver, cadences, polars_threads = args_tuple
    import os
    os.environ.setdefault("POLARS_MAX_THREADS", str(polars_threads))
    os.environ.setdefault("RAYON_NUM_THREADS", str(polars_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(polars_threads))
    try:
        # Re-import in worker (needed under spawn semantics on Windows)
        import sys as _sys
        from pathlib import Path as _Path
        _here = _Path(__file__).resolve().parent
        if str(_here) not in _sys.path:
            _sys.path.insert(0, str(_here))
        from feature_registry import FeatureRegistry as _Reg  # noqa: E402
        from universe_loader import UniverseLoader as _UL  # noqa: E402
        reg = _Reg.load()
        loader = _UL.load()
        return build_one(sym, reg, loader, skip_silver=skip_silver, cadences=cadences)
    except Exception as e:
        import traceback
        return {"asset": sym, "status": "error",
                "err": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[:1000]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None, help="Single asset. Deprecated alias for --assets [SYM].")
    # 2026-05-21 contract retrofit: --assets plural added.
    ap.add_argument("--assets", nargs="+", default=None,
                    help="Asset list (BTC or BTCUSDT format). Overrides --universe.")
    ap.add_argument("--universe", choices=["u10", "u50", "u100"], default=None,
                    help="Build only assets in this universe.")
    ap.add_argument("--skip-silver", action="store_true")
    ap.add_argument("--no-cadence", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Force fresh rebuild: delete prior dated v51 chimera snapshots "
                         "and frontier silver before rebuild. Overrides --skip-silver.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Per-asset parallel workers (default 1). Polars is "
                         "already multi-threaded internally and saturates CPU "
                         "with workers=1, so outer parallelism gives little "
                         "gain on most machines. On Linux with 32+ GB RAM, "
                         "workers=2-4 can give 1.5-2x speedup. WINDOWS NOTE: "
                         "Polars 1.x can segfault (exit 0xC0000005) under "
                         "concurrent processes; if you see this, drop to 1.")
    # Phase 7 bidirectional pattern
    ap.add_argument("-r", "--reverse", action="store_true",
                    help="Reverse asset iteration (Z->A) for meet-in-middle "
                         "2x speedup. Run two terminals: one without -r, one with. "
                         "Per-asset chimera files are independent; safe with "
                         "--workers.")
    args = ap.parse_args()

    # 2026-05-21 contract retrofit: --assets > --asset > --universe > raw discovery
    if args.assets:
        symbols = [a.upper() if a.upper().endswith("USDT") else a.upper() + "USDT"
                    for a in args.assets]
        # Migrated to pipeline.progress 2026-05-22 (top-5 producer ladder).
        from progress import phase_log as _pl
        _pl("chimera_v51", "SCAN", f"universe: --assets ({len(symbols)} explicit)")
    elif args.asset:
        symbols = [args.asset.upper() if not args.asset.endswith("USDT") else args.asset]
    elif args.universe:
        from universe_loader import UniverseLoader as _UL
        symbols = _UL.load().list(args.universe)
    else:
        # All assets that have a v50 chimera in the new layout
        symbols = _layout.list_v50_assets()

    # Phase 7 bidirectional: reverse iteration if requested
    if args.reverse and symbols:
        symbols = list(reversed(symbols))
        from progress import phase_log as _pl
        _pl("chimera_v51", "SCAN",
            f"REVERSE mode: iterating {len(symbols)} symbols Z->A (meet-in-middle pattern)")

    # Phase 8: centralized listing_dates marker. chimera_v51 consumes
    # chimera_legacy + frontier_silver outputs which already self-filter
    # to post-listing dates. Import documents the centralized-helper
    # contract (consumer crawler validation).
    try:
        import sys as _ld_sys
        _ld_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from pipeline.listing_dates import is_pre_listing as _is_pre_listing  # noqa: F401
    except ImportError:
        pass

    cadences = None if args.no_cadence else ["1d", "4h", "1h", "30m", "15m"]

    # @browser B1: --force LOUD + delete prior dated snapshots
    if args.force:
        from progress import phase_log as _pl
        if args.skip_silver:
            _pl("chimera_v51", "WARN",
                "FORCE overriding --skip-silver (rebuilding silver from scratch)")
            args.skip_silver = False
        n_deleted = 0
        # Delete v51 chimera dated snapshots for the resolved universe across all cadences.
        # Bug fix 2026-05-24: sym.lower().replace("USDT","") was a no-op because lower()
        # ran first making the suffix lowercase before replace looked for uppercase. The
        # resulting glob was f"{sym_l}usdt_v51_chimera*.parquet" with sym_l still
        # carrying "usdt" -> doubled suffix "btcusdtusdt_v51_chimera*.parquet" matched
        # zero files. Now: strip the USDT suffix BEFORE lowercasing so glob matches.
        cadence_dirs = ["dollar"] + (cadences or [])
        for sym in symbols:
            sym_l = sym.replace("USDT", "").lower()  # "BTCUSDT" -> "btc"
            for cad in cadence_dirs:
                try:
                    cad_dir = _layout.chimera_dir(cad)
                    if cad_dir.exists():
                        for old in cad_dir.glob(f"{sym_l}usdt_v51_chimera*.parquet"):
                            try:
                                old.unlink()
                                n_deleted += 1
                            except Exception:
                                pass
                except Exception:
                    pass
        from progress import phase_log as _pl
        _pl("chimera_v51", "SKIP",
            f"FORCE deleted {n_deleted} prior chimera_v51 snapshots across "
            f"{len(cadence_dirs)} cadences")

    from progress import phase_log as _pl
    _pl("chimera_v51", "START",
        f"V51 v2 build: {len(symbols)} symbols, cadences={cadences}, "
        f"skip_silver={args.skip_silver}, workers={args.workers}, force={args.force}")

    # Per-asset skip-existing: skip if v51 chimera 'dollar' output:
    #   (a) exists,
    #   (b) is fresher than BOTH the v50 chimera_legacy AND the frontier silver,
    #   (c) carries the schema invariants required for downstream training
    #       (xd_* cross-asset cols + frontier features).
    # Mirrors run_pipeline staleness semantics. Bypassed by --force.
    # Provenance: 2026-04-29 user run hit corrupt v51 files (131 cols, zero
    # xd_*) built from a broken v50 chimera_legacy (missing Phase 2). Pure
    # mtime check would silently keep these poisoned outputs.
    REQUIRED_V51_COLS = {
        "timestamp", "bar_id", "close",
        "target_return_1", "target_return_4", "target_return_16", "target_return_64",
        "norm_flow_imbalance", "norm_hawkes_imbalance",
        "xd_btc_return", "xd_funding_spread", "xd_momentum_rank",
    }
    pre_skipped: list = []
    pre_corrupt: list = []
    if not args.force:
        keep_symbols = []
        for sym in symbols:
            sym_u = sym.upper() if sym.upper().endswith("USDT") else sym.upper() + "USDT"
            v51_latest = _layout.chimera_v51_latest(sym_u, "dollar")
            if v51_latest is None or not v51_latest.exists():
                keep_symbols.append(sym)
                continue
            v51_m = v51_latest.stat().st_mtime
            v50_p = _layout.chimera_v50_latest(sym_u)
            silver_p = _layout.frontier_daily_latest(sym_u)
            v50_m = v50_p.stat().st_mtime if (v50_p and v50_p.exists()) else 0.0
            silver_m = silver_p.stat().st_mtime if (silver_p and silver_p.exists()) else 0.0
            mtime_fresh = v51_m >= max(v50_m, silver_m)
            if not mtime_fresh:
                keep_symbols.append(sym)
                continue
            # Schema invariant check: existing file must carry the required
            # cols. If missing, treat as stale and rebuild.
            try:
                cols = set(pl.read_parquet_schema(v51_latest).keys())
                missing = REQUIRED_V51_COLS - cols
            except Exception:
                missing = REQUIRED_V51_COLS  # unreadable -> rebuild
            if missing:
                pre_corrupt.append((sym_u, sorted(missing)[:3]))
                keep_symbols.append(sym)
                continue
            pre_skipped.append({
                "asset": sym_u, "status": "skip_fresh",
                "v51_path": v51_latest.name,
            })
        from progress import phase_log as _pl
        if pre_skipped:
            _pl("chimera_v51", "SKIP",
                f"{len(pre_skipped)} assets fresh "
                f"(v51 newer than v50 + silver, schema OK); use --force to rebuild")
        if pre_corrupt:
            _pl("chimera_v51", "WARN",
                f"{len(pre_corrupt)} assets have stale v51 "
                f"(missing required cols -- will rebuild)")
            for sym_u, missing in pre_corrupt[:5]:
                _pl("chimera_v51", "WARN", f"  {sym_u}: missing {missing}")
            if len(pre_corrupt) > 5:
                _pl("chimera_v51", "WARN", f"  ... and {len(pre_corrupt) - 5} more")
        symbols = keep_symbols

    results = list(pre_skipped)
    t_start = time.time()
    if not symbols:
        from progress import phase_log as _pl
        _pl("chimera_v51", "SKIP", "all assets fresh; nothing to build")

    if args.workers <= 1:
        # Sequential path (original behavior)
        reg = FeatureRegistry.load()
        loader = UniverseLoader.load()
        for i, sym in enumerate(symbols, 1):
            try:
                r = build_one(sym, reg, loader, skip_silver=args.skip_silver,
                              cadences=cadences)
            except Exception as e:
                r = {"asset": sym, "status": "error", "err": str(e)}
            results.append(r)
            _print_result(i, len(symbols), sym, r, cadences)
    else:
        # Parallel path: spawn fully-isolated child processes (subprocess) per
        # asset. Each child re-invokes make_dataset.py --asset <SYM>. This
        # avoids polars/rayon state issues we hit with ProcessPoolExecutor on
        # Windows + provides clean OS-level memory cleanup between assets.
        import os
        import subprocess as _sp
        cpu = os.cpu_count() or 8
        polars_threads = max(2, cpu // args.workers)
        from progress import phase_log as _pl
        _pl("chimera_v51", "BUILD",
            f"Worker thread budget: cpu_count={cpu} workers={args.workers} "
            f"-> polars_threads_per_worker={polars_threads} "
            f"(total={args.workers * polars_threads})")

        log_dir = PROJECT_ROOT / "logs" / "make_dataset"
        log_dir.mkdir(parents=True, exist_ok=True)

        def _spawn(sym: str) -> "_sp.Popen":
            # CRITICAL: child MUST run with --workers 1 (sequential path).
            # Previous bug: hardcoded "12" caused each child to re-enter the
            # parallel path and spawn its own children -> infinite recursion
            # -> Windows segfaults (exit=0xC0000005) and exit=0 noops.
            # The child's job is to process ONE asset; sequential is correct.
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--asset", sym, "--workers", "1"]
            if args.skip_silver:
                cmd.append("--skip-silver")
            if args.no_cadence:
                cmd.append("--no-cadence")
            env = os.environ.copy()
            env["POLARS_MAX_THREADS"] = str(polars_threads)
            env["RAYON_NUM_THREADS"] = str(polars_threads)
            env["OMP_NUM_THREADS"] = str(polars_threads)
            log_path = log_dir / f"build_{sym}.log"
            log_f = open(log_path, "w", encoding="utf-8")
            return _sp.Popen(cmd, stdout=log_f, stderr=_sp.STDOUT,
                             env=env, cwd=str(PROJECT_ROOT)), log_f, log_path

        # Pool of in-flight children
        running: list[tuple[str, _sp.Popen, "object", Path]] = []  # (sym, proc, log_f, log_path)
        pending = list(symbols)
        completed = 0
        # Capture run-start time so stale manifests from PRIOR runs aren't
        # mis-classified as success. A child is OK only if its manifest
        # mtime > _run_started (i.e. was written/updated by this run).
        _run_started_epoch = time.time()

        def _drain_one():
            nonlocal completed
            # Wait for any one child to finish (poll loop)
            import time as _t
            while True:
                for idx, (sym, proc, log_f, log_path) in enumerate(running):
                    rc = proc.poll()
                    if rc is not None:
                        log_f.close()
                        completed += 1
                        running.pop(idx)
                        # Read manifest to derive result info, fallback to status.
                        # FRESH check: manifest must be newer than run-start;
                        # otherwise it's a stale leftover from a prior build.
                        from layout import manifest_path as _mp
                        mp = _mp(sym)
                        manifest_fresh = (mp.exists()
                                          and mp.stat().st_mtime >= _run_started_epoch - 1)
                        if rc == 0 and manifest_fresh:
                            try:
                                m = json.loads(mp.read_text())
                                r = {
                                    "asset": sym, "status": "ok",
                                    "v50_cols": 0, "v51_cols": m.get("column_count", 0),
                                    "added": 0, "v51_rows": m.get("row_count", 0),
                                    "elapsed_s": 0.0,
                                    "cadence": m.get("cadence_views", {}),
                                }
                            except Exception as e:
                                r = {"asset": sym, "status": "ok_no_manifest",
                                     "err": str(e)}
                        else:
                            # Surface the real error from the child log
                            # instead of "stale_no_fresh_manifest" (which only
                            # describes the symptom, not the cause). Tail the
                            # log for the most informative line.
                            err_excerpt = ""
                            try:
                                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                                # Prefer lines mentioning error/missing/Traceback;
                                # else last non-empty line.
                                for L in reversed(tail[-50:]):
                                    Ll = L.lower()
                                    if any(k in Ll for k in ("error", "missing", "traceback", "no_silver")):
                                        err_excerpt = L.strip()[:200]
                                        break
                                if not err_excerpt:
                                    for L in reversed(tail[-10:]):
                                        if L.strip():
                                            err_excerpt = L.strip()[:200]
                                            break
                            except Exception:
                                pass
                            err_kind = ("stale_no_fresh_manifest" if rc == 0 and mp.exists()
                                        else f"exit={rc}")
                            r = {"asset": sym, "status": "error",
                                 "err": (f"{err_kind}: {err_excerpt}"
                                         if err_excerpt else f"{err_kind} log={log_path}")}
                        results.append(r)
                        _print_result(completed, len(symbols), sym, r, cadences)
                        return
                _t.sleep(0.2)

        spawned = 0
        for sym in pending:
            while len(running) >= args.workers:
                _drain_one()
            proc, log_f, log_path = _spawn(sym)
            running.append((sym, proc, log_f, log_path))
            spawned += 1
            # Migrated to pipeline.progress (2026-05-22) for homogeneous interface.
            # bootstrap (line 62-63) appends src/pipeline/ to sys.path; import
            # the bare module name (NOT pipeline.progress). Per RED-team flag
            # from auditor sweep ad217ce9239ceb16b.
            from progress import phase_log
            phase_log("chimera_v51", "BUILD",
                      f"spawned {sym} (running={len(running)}, completed={completed})",
                      counters={"i": spawned, "N": len(pending)})
        while running:
            _drain_one()

    elapsed = time.time() - t_start
    n_ok = sum(1 for r in results if r["status"] == "ok")
    from pipeline.progress import phase_log
    phase_log("chimera_v51", "DONE",
              f"{n_ok}/{len(symbols)} OK; outputs under {_layout.DIR_CHIMERA}; "
              f"manifests under {_layout.DIR_MANIFESTS}",
              counters={"elapsed": elapsed})

    # Coverage report (uniform across pipeline stages)
    try:
        from coverage_report import print_coverage_report
        ok_set = set(r["asset"].upper() for r in results if r.get("status") == "ok")
        ok_no_manifest = set(r["asset"].upper() for r in results
                             if r.get("status") == "ok_no_manifest")
        skip_fresh_set = set(r["asset"].upper() for r in results
                             if r.get("status") == "skip_fresh")
        err_set = set(r["asset"].upper() for r in results
                      if r.get("status") not in ("ok", "ok_no_manifest", "skip_fresh"))
        # Treat skip_fresh as success for coverage purposes (asset has output).
        # Use original requested universe (pre-skip filter) so coverage shows
        # the full picture.
        original_symbols = list(symbols) + [r["asset"] for r in pre_skipped]
        print_coverage_report(
            stage_name="chimera_v51",
            universe=args.universe,
            expected_assets=original_symbols,
            ok_assets=ok_set | ok_no_manifest | skip_fresh_set,
            err_assets=err_set,
            extra_lines=[f"Cadences: {','.join(cadences) if cadences else '(none)'}",
                         f"Workers: {args.workers}",
                         f"Skipped (fresh): {len(skip_fresh_set)}"],
        )
    except Exception as e:
        print(f"[coverage] WARN: {type(e).__name__}: {e}", flush=True)

    # Propagate non-zero exit on errors so parent orchestrators (run_pipeline,
    # make_dataset --workers>1 spawner) see real failures instead of silent
    # rc=0. Skip-fresh is success.
    n_err = sum(1 for r in results if r.get("status") not in ("ok", "ok_no_manifest", "skip_fresh"))
    if n_err > 0:
        sys.exit(2)


def _print_result(i: int, total: int, sym: str, r: dict, cadences) -> None:
    if r["status"] == "ok":
        cad_str = ", ".join(f"{c}:{r['cadence'][c]['rows']}r" for c in (cadences or []))
        print(f"[{i:>3}/{total}] {sym:>12} OK  "
              f"{r['v50_cols']:>3}->{r['v51_cols']:>3} +{r['added']:>3}  "
              f"{r['v51_rows']:>8}r  {cad_str}  ({r['elapsed_s']}s)")
    else:
        print(f"[{i:>3}/{total}] {sym:>12} {r['status']}  err={r.get('err', '')}")


if __name__ == "__main__":
    # Required on Windows for ProcessPoolExecutor with spawn semantics
    import multiprocessing
    multiprocessing.freeze_support()
    main()
