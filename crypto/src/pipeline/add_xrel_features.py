"""Cross-asset relative feature enrichment for v51 chimera (XREL stage).

Adds magnitude-preserving cross-asset relative features to existing chimera parquets.
Addresses root cause identified in dossiers INDEX.md Insights 35-37:
  norm_* features are per-asset rolling z-scores -- they destroy absolute magnitude.
  rv_bpv_5m (RAW) showed KS=0.253 within-day signal; norm_* showed KS=0.013-0.023.

Design:
  - All dollar-bar chimera features are DAILY-CONSTANT (constant within a date, as
    verified 2026-05-18: rv_bpv_5m, liq_long_xsec_z all have 1 unique value per date).
  - Cross-asset metrics are computed at DATE level across all 87 u100 assets.
  - Three metrics per feature:
      xrel_<f>_xrank   : fractional rank 0-1 within universe on that date
      xrel_<f>_xpct10  : binary 1 if asset in top-10% of universe (top-8 of 87)
      xrel_<f>_xratio  : asset value / universe median (NaN if median=0)
  - Output is written back to the SAME parquet files (atomic tmp+rename).
  - NO LOOKAHEAD: cross-section uses only same-date values (no future dates).

Features targeted (RAW, high within-day KS from signature mining):
  rv_bpv_5m    : bipower variation -- KS=0.253, 9/9 quarters stable
  rv_rv_5m     : realized variance  -- KS=0.251, 9/9 quarters stable
  hbr_eta_total: Hawkes branching   -- KS=0.064, useful secondary
  hbr_n_trades : trade count        -- raw activity proxy
  liq_long_usd : liquidation volume -- raw magnitude
  wh_whale_net_usd: whale net flow  -- signed, cross-rank meaningful
  lob_kyle_lambda_mean: Kyle lambda from LOB -- raw, high KS

Usage:
  python src/pipeline/add_xrel_features.py              # all 87 assets
  python src/pipeline/add_xrel_features.py --dry-run    # print plan, no writes
  python src/pipeline/add_xrel_features.py --asset BTC  # single asset
  python src/pipeline/add_xrel_features.py --force      # overwrite existing xrel_ cols

Smoke test (10 assets, 30 days):
  python src/pipeline/add_xrel_features.py --smoke

RWYB compliance: run script before committing. Check output with
  python src/pipeline/add_xrel_features.py --verify
"""
from __future__ import annotations

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "xrel_features",
    "inputs": {
        "upstream": ["data/processed/chimera/dollar/*_v51_chimera_*.parquet"],
        "args": ["--asset", "--assets", "--universe", "--dry-run", "--force", "--smoke", "--verify"],
    },
    "outputs": {
        "files": "data/processed/chimera/dollar/*_v51_chimera_*.parquet (xrel_ cols added in-place)",
        "new_columns": [
            "xrel_rv_bpv_5m_xrank", "xrel_rv_bpv_5m_xpct10", "xrel_rv_bpv_5m_xratio",
            "xrel_rv_rv_5m_xrank", "xrel_rv_rv_5m_xpct10", "xrel_rv_rv_5m_xratio",
            "xrel_hbr_eta_total_xrank", "xrel_hbr_eta_total_xpct10", "xrel_hbr_eta_total_xratio",
            "xrel_hbr_n_trades_xrank", "xrel_hbr_n_trades_xpct10", "xrel_hbr_n_trades_xratio",
            "xrel_liq_long_usd_xrank", "xrel_liq_long_usd_xpct10", "xrel_liq_long_usd_xratio",
            "xrel_wh_whale_net_usd_xrank", "xrel_wh_whale_net_usd_xpct10", "xrel_wh_whale_net_usd_xratio",
            "xrel_lob_kyle_lambda_mean_xrank", "xrel_lob_kyle_lambda_mean_xpct10", "xrel_lob_kyle_lambda_mean_xratio",
        ],
    },
    "invariants": {
        "atomic_write": True,
        "no_lookahead": True,
        "additive_only": True,  # NEVER removes existing columns
        "daily_constant": True,  # xrel_ cols are constant within each date (same as source)
    },
    "rationale": (
        "norm_* features destroy magnitude signal (rv_bpv_5m RAW KS=0.253 vs norm_ KS~0.02). "
        "Cross-asset rank/ratio features preserve magnitude without look-ahead."
    ),
}

import argparse
import os
import sys
import time
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHIMERA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"
# 2026-05-21: also propagate xrel_* to cadence files (1d/4h/1h/15m)
# so pre_train_gate's cadence_*_schema validators pass. xrel_* are daily-constant
# (same value for every bar within a date), so broadcasting by date works.
CADENCE_DIRS = [PROJECT_ROOT / "data" / "processed" / "chimera" / c
                 for c in ("1d", "4h", "1h", "15m")]

# Features to enrich with cross-asset relatives.
# Each must be present in chimera (or silently skipped if absent).
XREL_FEATURES = [
    "rv_bpv_5m",
    "rv_rv_5m",
    "hbr_eta_total",
    "hbr_n_trades",
    "liq_long_usd",
    "wh_whale_net_usd",
    "lob_kyle_lambda_mean",
]

# Top-N% threshold for xpct10 (top 10% of 87 = top 8 assets)
TOP_N_FRAC = 0.10


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("xrel", phase, message, **kw)


def _asset_from_file(path: Path) -> str:
    """Extract asset symbol from chimera filename."""
    name = path.name  # e.g. btcusdt_v51_chimera_20260515.parquet
    return name.split("usdt")[0].upper()


def load_daily_panel(files: list[Path], features: list[str]) -> pl.DataFrame:
    """Load one daily row per (asset, date) for all requested features.

    Returns a DataFrame with columns: [date, asset] + features (one row per date).
    NO_LOOKAHEAD: no future data used; values come from the asset's own daily snapshot.
    """
    cols_to_load = ["date"] + [f for f in features]
    dfs = []
    for path in files:
        asset = _asset_from_file(path)
        avail_cols = [c for c in pl.scan_parquet(path).columns if c in cols_to_load]
        df = pl.read_parquet(path, columns=avail_cols)
        # All chimera features are daily-constant -- take one row per date
        daily = df.unique(subset=["date"], keep="first")
        # Add missing feature columns as null so panel is uniform
        for feat in features:
            if feat not in daily.columns:
                daily = daily.with_columns(pl.lit(None, dtype=pl.Float64).alias(feat))
        daily = daily.select(["date"] + features).with_columns(
            pl.lit(asset).alias("asset")
        )
        dfs.append(daily)

    panel = pl.concat(dfs, how="diagonal").sort(["date", "asset"])
    return panel


def compute_xrel(panel: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """Compute cross-asset xrank/xpct10/xratio for each feature on each date.

    xrank   : fractional rank within universe [0, 1], NaN if all null
    xpct10  : 1 if xrank > (1 - TOP_N_FRAC), else 0  (top 10%)
    xratio  : value / median(universe), NaN if median == 0 or value is null

    Returns panel with 3 new columns per feature (prefixed xrel_<feat>_).
    """
    new_cols = []
    for feat in features:
        rank_col = f"xrel_{feat}_xrank"
        top_col = f"xrel_{feat}_xpct10"
        ratio_col = f"xrel_{feat}_xratio"

        # Fractional rank within each date group (handles NaN gracefully via sort)
        panel = panel.with_columns(
            pl.col(feat)
            .rank(method="average")
            .over("date")
            .alias("_raw_rank")
        ).with_columns(
            pl.col("_raw_rank").count().over("date").alias("_n")
        ).with_columns(
            (pl.col("_raw_rank") / pl.col("_n")).alias(rank_col)
        ).drop(["_raw_rank", "_n"])

        # Binary top-10% flag
        panel = panel.with_columns(
            (pl.col(rank_col) > (1.0 - TOP_N_FRAC)).cast(pl.Int8).alias(top_col)
        )

        # Ratio to daily mean-of-absolutes (heavy-tail robust scale).
        # 2026-05-21 second-pass fix: median(|values|) was still collapsing for
        # bimodal-distributed signed features like lob_kyle_lambda_mean and
        # wh_whale_net_usd, where ~half the universe has near-zero values
        # (low-activity assets) and ~half has real values. Median(|values|)
        # picks up the low-activity half (≈0), forcing divisor → 0 → clip.
        #
        # mean(|values|) is robust to bimodal: the active-asset values dominate
        # the mean even when half the universe is zero. Empirical check (kyle
        # lambda, BTC date 2026-05-19): median(|values|)=0.068 → BTC ratio=17
        # (recent); but tail-100k spans ~10 days with many zero-divisor days.
        # mean(|values|)=1.88 → BTC ratio=0.62 (well within clamp).
        # Trade-off: smaller dynamic range, but no clamping. Magnitude semantics
        # preserved (sign of value, magnitude relative to universe-average).
        panel = panel.with_columns(
            pl.col(feat).abs().mean().over("date").alias("_abs_mean")
        ).with_columns(
            pl.when(pl.col("_abs_mean") > 1e-12)
            .then((pl.col(feat) / pl.col("_abs_mean")).clip(-100.0, 100.0))
            .otherwise(None)
            .alias(ratio_col)
        ).drop("_abs_mean")

    return panel


def write_enriched(path: Path, xrel_daily: pl.DataFrame, force: bool = False) -> bool:
    """Join daily xrel_ values back to the full dollar-bar parquet and write atomically.

    Returns True if write happened, False if skipped (already has xrel_ and not force).
    """
    asset = _asset_from_file(path)
    df = pl.read_parquet(path)

    existing_xrel = [c for c in df.columns if c.startswith("xrel_")]
    if existing_xrel and not force:
        return False  # Already enriched; skip

    # Drop any existing xrel_ columns (clean re-enrichment when --force)
    if existing_xrel:
        df = df.drop(existing_xrel)

    # Extract xrel rows for this asset
    asset_xrel = xrel_daily.filter(pl.col("asset") == asset).drop("asset")

    # Determine xrel columns (only xrel_ prefixed ones -- exclude source feature cols)
    xrel_cols = [c for c in asset_xrel.columns if c.startswith("xrel_")]

    # Join on date (all dollar bars on the same date get the same daily xrel values)
    if "date" not in df.columns:
        _pl("WARN", f"{asset}: no date column -- skipping enrichment")
        return False

    df_date_type = df.schema["date"]
    # LAG-1 join: xrel computed from day D's daily aggregations (n_trades,
    # liq_total, hbr_eta, etc.) reflects the FULL day's activity -- including
    # trades AFTER any given dollar bar within day D. Joining same-date is a
    # same-day publication race (validator MI=0.0502 FAIL on hbr_n_trades_xratio,
    # 2026-05-23). Shift xrel date +1 so day-D xrel becomes the value visible
    # at bars on day D+1 (equivalent to as-of yesterday EOD). Earliest day
    # gets NULL xrel; downstream null-handling already in place.
    asset_xrel = asset_xrel.with_columns(
        (pl.col("date") + pl.duration(days=1)).cast(df_date_type)
    )

    enriched = df.join(asset_xrel.select(["date"] + xrel_cols), on="date", how="left")

    # Atomic write: tmp file + rename
    tmp = path.with_suffix(".xrel_tmp.parquet")
    enriched.write_parquet(str(tmp), compression="zstd", compression_level=3)
    tmp.replace(path)
    return True


def write_enriched_cadence(cadence_path: Path, xrel_daily: pl.DataFrame,
                             force: bool = False) -> bool:
    """Broadcast daily xrel_* values onto a CADENCE chimera file (1d/4h/1h/15m).

    Same logic as write_enriched (dollar) but for cadence files. xrel_* values
    are daily-constant so joining on date broadcasts correctly to all sub-day bars.
    Returns True if write happened, False if skipped.
    """
    asset = _asset_from_file(cadence_path)
    df = pl.read_parquet(cadence_path)
    existing_xrel = [c for c in df.columns if c.startswith("xrel_")]
    if existing_xrel and not force:
        return False
    if existing_xrel:
        df = df.drop(existing_xrel)
    asset_xrel = xrel_daily.filter(pl.col("asset") == asset).drop("asset")
    xrel_cols = [c for c in asset_xrel.columns if c.startswith("xrel_")]
    if not xrel_cols:
        return False
    if "date" not in df.columns:
        # Derive from timestamp
        if "timestamp" in df.columns:
            df = df.with_columns(
                pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date().alias("date")
            )
        else:
            return False
    df_date_type = df.schema["date"]
    # LAG-1 join (see write_enriched docstring): no-lookahead semantics for
    # daily aggregations joined onto sub-day bars.
    asset_xrel = asset_xrel.with_columns(
        (pl.col("date") + pl.duration(days=1)).cast(df_date_type)
    )
    enriched = df.join(asset_xrel.select(["date"] + xrel_cols), on="date", how="left")
    tmp = cadence_path.with_suffix(".xrel_tmp.parquet")
    enriched.write_parquet(str(tmp), compression="zstd", compression_level=3)
    tmp.replace(cadence_path)
    return True


def run(
    files: list[Path],
    features: list[str],
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Main pipeline: build panel, compute xrel, write back per asset."""
    t0 = time.time()
    n_assets = len(files)
    _pl("OK", f"XREL enrichment: {n_assets} assets, {len(features)} features")
    print(f"     Features: {features}")

    if dry_run:
        print("[DRY-RUN] Would process:")
        for f in files:
            print(f"  {_asset_from_file(f)}")
        return {"dry_run": True, "n_assets": n_assets}

    # Step 1: Load daily panel
    print("[...] Loading daily panel across all assets...")
    panel = load_daily_panel(files, features)
    n_dates = panel["date"].n_unique()
    n_rows = len(panel)
    _pl("OK", f"Daily panel: {n_rows} rows ({n_assets} assets x {n_dates} dates)")

    # Step 2: Compute cross-asset xrel features
    print("[...] Computing cross-asset xrel features...")
    panel_xrel = compute_xrel(panel, features)
    new_xrel_cols = [c for c in panel_xrel.columns if c.startswith("xrel_")]
    _pl("OK", f"Computed {len(new_xrel_cols)} new xrel columns")

    # Validation: check xrank range and top10 counts
    for feat in features:
        rank_col = f"xrel_{feat}_xrank"
        if rank_col in panel_xrel.columns:
            rmin = panel_xrel[rank_col].min()
            rmax = panel_xrel[rank_col].max()
            if rmin is not None and (rmin < 0 or rmax > 1):
                _pl("WARN", f"{rank_col}: rank out of [0,1] range: min={rmin:.3f} max={rmax:.3f}")

    # Step 3: Write back per asset
    print("[...] Writing enriched parquets...")
    n_written = 0
    n_skipped = 0
    for path in files:
        asset = _asset_from_file(path)
        written = write_enriched(path, panel_xrel, force=force)
        if written:
            n_written += 1
            if verbose:
                _pl("OK", f"{asset} -- enriched")
        else:
            n_skipped += 1
            if verbose:
                print(f"  [--] {asset} -- skipped (already has xrel_, use --force)")

    # Step 4: ALSO broadcast xrel_* to cadence files (1d/4h/1h/15m) — 2026-05-21 fix.
    # xrel_* are daily-constant so join-by-date works cleanly.
    print("[...] Propagating xrel_* to cadence chimera files...")
    n_cadence_written = 0
    n_cadence_skipped = 0
    # Build asset filter from the dollar files we just processed (so cadence
    # filter matches dollar filter when --assets / --universe is used).
    asset_filter = {_asset_from_file(p).upper() for p in files}
    for cadence_dir in CADENCE_DIRS:
        if not cadence_dir.exists():
            continue
        cadence_files = sorted(cadence_dir.glob("*_v51_chimera_*.parquet"))
        for cpath in cadence_files:
            asset_c = _asset_from_file(cpath).upper()
            if asset_c not in asset_filter:
                continue
            try:
                if write_enriched_cadence(cpath, panel_xrel, force=force):
                    n_cadence_written += 1
                else:
                    n_cadence_skipped += 1
            except Exception as e:
                _pl("WARN", f"{asset_c} {cadence_dir.name}: {type(e).__name__}: {e}")
    print(f"[OK] Cadence xrel propagation: {n_cadence_written} written, "
          f"{n_cadence_skipped} skipped across {len(CADENCE_DIRS)} cadences")

    elapsed = time.time() - t0
    print(f"[OK] XREL enrichment COMPLETE: {n_written} dollar + "
          f"{n_cadence_written} cadence written, "
          f"{n_skipped} dollar / {n_cadence_skipped} cadence skipped in {elapsed:.1f}s")
    return {
        "n_written": n_written,
        "n_skipped": n_skipped,
        "n_xrel_cols": len(new_xrel_cols),
        "elapsed_s": elapsed,
    }


def verify(files: list[Path], features: list[str]) -> None:
    """Spot-check enriched parquets: verify xrel_ cols present and sane."""
    n_ok = 0
    n_fail = 0
    for path in files[:5]:  # Sample first 5
        asset = _asset_from_file(path)
        df = pl.read_parquet(path)
        xrel = [c for c in df.columns if c.startswith("xrel_")]
        expected = len(features) * 3
        if len(xrel) < expected:
            _pl("FAIL", f"{asset}: expected {expected} xrel cols, got {len(xrel)}")
            n_fail += 1
            continue
        # Check xrank is in [0,1]
        for feat in features:
            rank_col = f"xrel_{feat}_xrank"
            if rank_col not in df.columns:
                continue
            daily = df.unique(subset=["date"], keep="first")
            # daily cols are constant within date so just check unique values
            rmin = daily[rank_col].min()
            rmax = daily[rank_col].max()
            if rmin is not None and (rmin < -0.01 or rmax > 1.01):
                _pl("FAIL", f"{asset} {rank_col}: range [{rmin:.3f},{rmax:.3f}]")
                n_fail += 1
                continue
        _pl("OK", f"{asset}: {len(xrel)} xrel cols, xrank range [0,1] check pass")
        n_ok += 1

    print(f"\n[VERIFY] {n_ok} OK, {n_fail} FAIL (sampled {min(5,len(files))} assets)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Add cross-asset xrel_ features to v51 chimera.")
    # 2026-05-21 contract retrofit: --assets plural added alongside --asset singular.
    ap.add_argument("--asset", default=None, help="Single asset (e.g. BTC). Deprecated alias for --assets [SYM].")
    ap.add_argument("--assets", nargs="+", default=None,
                    help="Asset list (BTCUSDT or BTC format). Overrides --universe.")
    ap.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                    help="Resolve assets via UniverseLoader. Default: all chimera files in input-dir.")
    ap.add_argument("--workers", type=int, default=1, help="Not used by xrel (cross-section)).")
    ap.add_argument("--dry-run", action="store_true", help="Print plan, no writes")
    ap.add_argument("--force", action="store_true", help="Overwrite existing xrel_ columns")
    ap.add_argument("--smoke", action="store_true", help="Run on first 10 assets only")
    ap.add_argument("--verify", action="store_true", help="Spot-check existing xrel_ cols")
    ap.add_argument(
        "--input-dir",
        default=None,
        help="Override input directory (default: data/processed/chimera/dollar)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve input directory
    input_dir = Path(args.input_dir).resolve() if args.input_dir else CHIMERA_DIR
    if not input_dir.exists():
        _pl("FAIL", f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Discover chimera files
    all_files = sorted(input_dir.glob("*_v51_chimera_*.parquet"))
    if not all_files:
        _pl("FAIL", f"No chimera files found at {CHIMERA_DIR}")
        sys.exit(1)

    # 2026-05-21 contract retrofit: build asset filter from --assets / --asset / --universe
    asset_filter: list[str] | None = None
    if args.assets:
        asset_filter = [a.replace("USDT", "").lower() for a in args.assets]
    elif args.asset:
        asset_filter = [args.asset.replace("USDT", "").lower()]
    elif args.universe:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from universe_loader import UniverseLoader
            raw = UniverseLoader.load().list(args.universe)
            asset_filter = [s.replace("USDT", "").lower() for s in raw]
            _pl("BUILD", f"universe: {args.universe} ({len(asset_filter)} assets)")
        except Exception as e:
            _pl("FAIL", f"FALLBACK: --universe {args.universe} load failed ({e}); processing all")
            asset_filter = None
    if asset_filter:
        filtered = [f for f in all_files if any(f.name.startswith(sym) for sym in asset_filter)]
        if not filtered:
            _pl("FAIL", f"No chimera files matching asset filter: {asset_filter}")
            sys.exit(1)
        all_files = filtered
        _pl("BUILD", f"filtered to {len(all_files)} assets")

    if args.smoke:
        all_files = all_files[:10]
        print(f"[SMOKE] Using first 10 assets only")

    if args.verify:
        verify(all_files, XREL_FEATURES)
        return

    result = run(
        files=all_files,
        features=XREL_FEATURES,
        dry_run=args.dry_run,
        force=args.force,
    )
    if not args.dry_run and result.get("n_written", 0) == 0 and not args.force:
        print("[INFO] Nothing written. Use --force to re-enrich already-processed assets.")


if __name__ == "__main__":
    main()
