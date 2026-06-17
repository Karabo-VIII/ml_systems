"""Spot-perp basis features.

basis_pct = (perp_close - spot_close) / spot_close * 100
Positive = contango (perp premium, bullish positioning)
Negative = backwardation (perp discount, stress/panic)

Inputs:
    data/processed/panels/daily/spot_klines_daily.parquet
        (built by src/pipeline/ingest/binance_spot_klines.py)
    data/processed/chimera_legacy/dollar/<sym>usdt_v50_chimera_<DATE>.parquet
        (built by make_dataset_legacy.py; we extract daily perp close from bars)

Output:
    data/processed/panels/daily/basis_features_long.parquet

Features per asset, per date (registry liq_features spec compatible):
    basis_pct           (perp - spot) / spot * 100
    basis_z30           per-asset rolling 30d z-score (shifted-1, no leak)
    basis_delta_1d      d/dt change (1d)
    basis_delta_3d      d/dt change (3d)
    basis_xsec_z        cross-sectional z per day (across all assets)
    basis_bull_shock    binary: basis_z30 > +2 (extreme contango)
    basis_bear_shock    binary: basis_z30 < -2 (extreme backwardation)
    basis_panic         binary: basis_pct < -0.5 (absolute stress)
    basis_frenzy        binary: basis_pct > +1.0 (absolute overextension)
"""
from __future__ import annotations
import os

# CDAP contract -- declared after __future__ per PEP-236.
__contract__ = {
    "kind": "panel_builder",
    "stage": "basis_signals",
    "inputs": {
        "args": ["--force"],
        "upstream": [
            "data/processed/panels/daily/spot_klines_daily.parquet",
            "data/processed/chimera_legacy/dollar/*_v50_chimera_*.parquet",
        ],
    },
    "outputs": {
        "files": "data/processed/panels/daily/basis_features_long.parquet",
        "columns": ["date", "asset", "perp_close", "spot_close", "basis_pct",
                    "basis_z30", "basis_delta_1d", "basis_delta_3d",
                    "basis_xsec_z", "basis_bull_shock", "basis_bear_shock",
                    "basis_panic", "basis_frenzy"],
    },
    "invariants": {
        "atomic_write": True,
        "column_name_verify": True,
    },
}

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
import layout as _layout  # noqa: E402

SPOT_IN = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "spot_klines_daily.parquet"
OUT = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "basis_features_long.parquet"


def _resolve_chimera_root_from_stem(stem: str) -> str | None:
    """Extract asset root (e.g. 'BTC') from filename stem like 'btcusdt_v50_chimera_20260427'."""
    stem = stem.lower()
    if len(stem) > 9 and stem[-9] == "_" and stem[-8:].isdigit():
        stem = stem[:-9]
    sym = stem.replace("usdt_v50_chimera", "").upper()
    return sym or None


def _scan_one_chimera_legacy(fp: Path) -> pd.DataFrame | None:
    """Read one chimera_legacy parquet, return per-day last-close.

    Returns None on read failure / too-few rows / unresolvable asset.
    """
    try:
        df = pl.read_parquet(fp, columns=["timestamp", "close"]).to_pandas()
    except Exception:
        return None
    if len(df) < 500:
        return None
    sym = _resolve_chimera_root_from_stem(fp.stem)
    if not sym:
        return None
    df["date"] = pd.to_datetime(
        df["timestamp"].apply(lambda _t: _t // 1000 if _t >= 1e15 else _t),
        unit="ms").dt.normalize()
    d = df.groupby("date").agg({"close": "last"}).reset_index()
    d["asset"] = sym
    d = d.rename(columns={"close": "perp_close"})
    return d


def load_perp_daily(asset_filter: list[str] | None = None,
                    workers: int = 1) -> pd.DataFrame:
    """Read every chimera_legacy parquet, extract daily last-close per asset.

    asset_filter: optional list of asset roots (e.g. ['BTC','ETH']) — restricts the
        scan to filenames matching those roots. Cross-section z is then computed
        over the restricted set.
    workers: thread workers for parallel parquet reads (per-file IO is independent).
    """
    legacy_dir = _layout.chimera_legacy_dir()
    fps = sorted(legacy_dir.glob("*_v50_chimera*.parquet"))

    # Migrated to pipeline.progress (2026-05-22) for homogeneous interface.
    # Import inline because basis_signals can be run as a script. The file's
    # bootstrap (line 62) inserts src/pipeline/ on sys.path — NOT src/ — so
    # the bare module name `progress` is the correct import (RED-team flag
    # from auditor sweep ad217ce9239ceb16b).
    from progress import phase_log, ProgressTask

    if asset_filter:
        filter_set = {a.upper().replace("USDT", "") for a in asset_filter}
        before = len(fps)
        fps = [fp for fp in fps if (_resolve_chimera_root_from_stem(fp.stem) or "") in filter_set]
        phase_log("basis_feat", "SCAN",
                  f"--assets filter: scanning {len(fps)}/{before} files "
                  f"({len(filter_set)} asset roots requested)")
    else:
        phase_log("basis_feat", "SCAN", f"full cross-section scan: {len(fps)} files")

    rows: list[pd.DataFrame] = []
    if workers > 1 and len(fps) > 4:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ProgressTask("basis_feat", total=len(fps),
                           label=f"scan ({workers} threads)") as bar:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_fp = {ex.submit(_scan_one_chimera_legacy, fp): fp for fp in fps}
                for fut in as_completed(future_to_fp):
                    d = fut.result()
                    if d is not None:
                        rows.append(d)
                    bar.update(1, msg=f"kept={len(rows)}")
    else:
        with ProgressTask("basis_feat", total=len(fps), label="scan (serial)") as bar:
            for fp in fps:
                d = _scan_one_chimera_legacy(fp)
                if d is not None:
                    rows.append(d)
                bar.update(1, msg=f"kept={len(rows)}")

    if not rows:
        return pd.DataFrame(columns=["date", "asset", "perp_close"])
    return pd.concat(rows, ignore_index=True)


def build(asset_filter: list[str] | None = None, workers: int = 1) -> pd.DataFrame:
    """Build basis features panel.

    asset_filter: optional list of asset roots restricting both spot + perp inputs;
        cross-section z is then computed over the restricted set.
    workers: thread workers for parallel chimera_legacy parquet reads.
    """
    if not SPOT_IN.exists():
        raise FileNotFoundError(
            f"basis_signals depends on {SPOT_IN.name} (built by "
            f"binance_spot_klines.py). Missing at {SPOT_IN}.")

    spot = pd.read_parquet(SPOT_IN)
    spot["date"] = pd.to_datetime(spot["date"])
    if asset_filter:
        filter_set = {a.upper().replace("USDT", "") for a in asset_filter}
        spot_before = len(spot)
        spot = spot[spot["asset"].str.upper().isin(filter_set)].reset_index(drop=True)
        from progress import phase_log as _pl
        _pl("basis_feat", "SCAN",
            f"--assets filter spot: kept {len(spot)}/{spot_before} rows")
    perp = load_perp_daily(asset_filter=asset_filter, workers=workers)
    if perp.empty:
        raise RuntimeError("basis_signals: no perp daily data found in chimera_legacy/. "
                           "Run make_dataset_legacy.py first.")
    perp["date"] = pd.to_datetime(perp["date"])

    # Phase 6 audit fix (pipeline_audit_crawler dead-feature finding):
    # bs_basis_delta_1d was 100% zero across BTC/ETH/SOL because spot OR
    # perp had duplicate (date, asset) rows, the merge cartesianed them,
    # diff() across identical-value sibling rows returned 0, then chimera
    # dedup kept the zero row. Dedupe BOTH sides before merge.
    n_spot_before = len(spot)
    n_perp_before = len(perp)
    spot = spot.drop_duplicates(subset=["date", "asset"], keep="last")
    perp = perp.drop_duplicates(subset=["date", "asset"], keep="last")
    if len(spot) < n_spot_before:
        from progress import phase_log as _pl
        _pl("basis_feat", "WARN",
            f"dropped {n_spot_before - len(spot)} duplicate (date, asset) rows from spot input")
    if len(perp) < n_perp_before:
        from progress import phase_log as _pl
        _pl("basis_feat", "WARN",
            f"dropped {n_perp_before - len(perp)} duplicate (date, asset) rows from perp input")

    df = spot.merge(perp, on=["date", "asset"], how="inner")
    if df.empty:
        raise RuntimeError("basis_signals: no overlap between spot + perp panels. "
                           "Check date ranges and asset names.")
    df["basis_pct"] = (df["perp_close"] - df["spot_close"]) / df["spot_close"] * 100
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)

    # Per-asset rolling z (shifted-1 = no leak) + delta
    g = df.groupby("asset")["basis_pct"]
    rm = g.transform(lambda s: s.shift(1).rolling(30, min_periods=10).mean())
    rs = g.transform(lambda s: s.shift(1).rolling(30, min_periods=10).std())
    df["basis_z30"] = ((df["basis_pct"] - rm) / rs.replace(0, np.nan)).clip(-5.0, 5.0)
    df["basis_delta_1d"] = g.diff()
    df["basis_delta_3d"] = g.diff(3)

    # Cross-sectional z per day (across all assets in the panel)
    df["basis_xsec_z"] = df.groupby("date")["basis_pct"].transform(
        lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
    ).clip(-5.0, 5.0)

    # Event flags (binary)
    df["basis_bull_shock"] = (df["basis_z30"] > 2.0).fillna(False).astype(int)
    df["basis_bear_shock"] = (df["basis_z30"] < -2.0).fillna(False).astype(int)
    df["basis_panic"] = (df["basis_pct"] < -0.5).fillna(False).astype(int)
    df["basis_frenzy"] = (df["basis_pct"] > 1.0).fillna(False).astype(int)

    keep = ["date", "asset", "perp_close", "spot_close", "basis_pct",
            "basis_z30", "basis_delta_1d", "basis_delta_3d",
            "basis_xsec_z", "basis_bull_shock", "basis_bear_shock",
            "basis_panic", "basis_frenzy"]
    return df[keep]


def main() -> int:
    # Migrated to pipeline.progress 2026-05-22 — homogeneous CLI per
    # docs/PIPELINE_PROGRESS_CONVENTION_2026_05_22.md
    from progress import phase_log, stage_run
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="Force rebuild even if OUT panel is fresher than chimera inputs.")
    # 2026-05-21 (revised): --assets / --universe / --workers wired through.
    # NB cross-section semantics: passing --assets RESTRICTS the panel's universe,
    # so basis_xsec_z is computed over the restricted set. For pipeline-wide
    # consistency with the T2 orchestration convention, this is acceptable as
    # long as downstream consumers either read all assets or filter to the same
    # subset. The full-universe build is the default (no --assets flag).
    ap.add_argument("--assets", nargs="+", default=None,
                    help="Restrict spot + perp scan to these asset roots (e.g. BTC ETH SOL). "
                         "Cross-section z is then computed over the restricted set. "
                         "Default: scan all chimera_legacy assets.")
    ap.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                    help="Restrict scan via universe spec (resolved through universe_loader). "
                         "Mutually exclusive with --assets in spirit; --assets wins if both passed.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Thread workers for parallel chimera_legacy parquet reads.")
    ap.add_argument("--dry-run", action="store_true", help="Print plan, no writes.")
    args = ap.parse_args()

    # Resolve --universe to an asset list when --assets not provided.
    asset_filter: list[str] | None = args.assets
    if asset_filter is None and args.universe:
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
            from universe_loader import UniverseLoader  # noqa: E402
            asset_filter = [s.upper().replace("USDT", "")
                            for s in UniverseLoader.load().list(args.universe)]
            phase_log("basis_feat", "SCAN",
                      f"--universe {args.universe} resolved to {len(asset_filter)} assets")
        except Exception as e:
            phase_log("basis_feat", "WARN",
                      f"--universe {args.universe} resolve failed "
                      f"({type(e).__name__}: {e}); falling back to full cross-section")
            asset_filter = None

    # 2026-05-21 contract retrofit: skip-existing.
    # Skip rebuild if OUT panel mtime >= max(chimera_legacy mtime). --force overrides.
    if OUT.exists() and not args.force:
        out_mtime = OUT.stat().st_mtime
        # Cheapest staleness check: compare against any chimera_legacy file mtime.
        # Anchor to project root (NOT cwd): a relative path silently found nothing
        # when run from another directory, skewing the freshness decision.
        from pathlib import Path as _P
        leg_dir = _P(__file__).resolve().parents[3] / "data/processed/chimera_legacy/dollar"
        if leg_dir.exists():
            max_leg = max((f.stat().st_mtime for f in leg_dir.glob("*_v50_chimera*.parquet")),
                           default=0.0)
            if out_mtime >= max_leg:
                phase_log("basis_feat", "SKIP",
                          "OUT panel fresher than chimera_legacy inputs; --force to rebuild")
                return 0

    if args.dry_run:
        phase_log("basis_feat", "SKIP", f"DRY-RUN: would rebuild {OUT}")
        return 0

    workers = max(1, int(args.workers))
    phase_log("basis_feat", "START",
              f"reading {SPOT_IN.name} + scanning chimera_legacy "
              f"({workers} thread{'s' if workers > 1 else ''})")

    df = build(asset_filter=asset_filter, workers=workers)
    phase_log("basis_feat", "BUILD",
              f"built (scanned subset): {len(df)} rows, "
              f"{df['asset'].nunique()} assets, {df['date'].nunique()} dates")

    # Cross-section incremental merge: when --assets / --universe restricts the
    # scan to a subset, MERGE with the existing panel so other assets' rows are
    # preserved. basis_xsec_z is then RE-COMPUTED on the merged universe so the
    # cross-sectional reference set stays correct.
    if asset_filter and OUT.exists():
        rebuilt_assets = set(df["asset"].unique())
        existing = pd.read_parquet(OUT)
        existing["date"] = pd.to_datetime(existing["date"])
        existing_kept = existing[~existing["asset"].isin(rebuilt_assets)].copy()
        phase_log("basis_feat", "BUILD",
                  f"merge: keeping {existing_kept['asset'].nunique()} non-rebuilt "
                  f"assets ({len(existing_kept)} rows); replacing {len(rebuilt_assets)} rebuilt")
        # Drop xsec_z from existing (will be re-computed) — keep all other cols.
        if "basis_xsec_z" in existing_kept.columns:
            existing_kept = existing_kept.drop(columns=["basis_xsec_z"])
        if "basis_xsec_z" in df.columns:
            df = df.drop(columns=["basis_xsec_z"])
        # Align column sets before concat.
        cols = [c for c in df.columns if c in existing_kept.columns]
        df = pd.concat([existing_kept[cols], df[cols]], ignore_index=True)
        df = df.sort_values(["asset", "date"]).reset_index(drop=True)
        # Re-compute cross-sectional z on the merged universe.
        df["basis_xsec_z"] = df.groupby("date")["basis_pct"].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
        ).clip(-5.0, 5.0)
        phase_log("basis_feat", "BUILD",
                  f"merged: {len(df)} rows, {df['asset'].nunique()} assets, "
                  f"{df['date'].nunique()} dates")

    # Atomic-tmp-rename + column-name verify (G-AUDIT-020).
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    import pyarrow.parquet as _pq
    written = set(_pq.read_schema(tmp).names)
    required = {"date", "asset", "basis_pct", "basis_z30",
                "basis_xsec_z", "basis_bull_shock", "basis_panic"}
    missing = required - written
    if missing:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"basis panel missing required cols: {sorted(missing)}")
    if OUT.exists():
        OUT.unlink()
    os.replace(str(tmp), str(OUT))  # atomic overwrite (Windows-safe)
    phase_log("basis_feat", "WRITE", f"saved: {OUT}")

    # Coverage summary — emit as single OK line for parseability.
    n_assets = df["asset"].nunique()
    coverage_msg = (f"date={df['date'].min().date()}->{df['date'].max().date()} "
                    f"assets={n_assets} rows={len(df)} "
                    f"basis_pct mean={df['basis_pct'].mean():+.3f}% "
                    f"std={df['basis_pct'].std():.3f}%")
    phase_log("basis_feat", "OK", coverage_msg)
    for s in ["basis_bull_shock", "basis_bear_shock", "basis_panic", "basis_frenzy"]:
        if s in df.columns:
            n = int(df[s].fillna(0).sum())
            phase_log("basis_feat", "OK",
                      f"signal_incidence {s}: {n} asset-days ({100*n/len(df):.2f}%)")
    phase_log("basis_feat", "DONE", "basis_features_long")
    return 0


if __name__ == "__main__":
    sys.exit(main())
