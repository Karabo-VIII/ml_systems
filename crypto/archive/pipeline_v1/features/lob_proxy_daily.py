"""LOB proxy daily aggregator.

Input panels (bar-level, one file per (symbol, day)):
    data/processed/panels/daily/lob_proxy_<SYM>USDT_<YYYYMMDD>.parquet
    (built by src/pipeline/features/lob_proxy_panel.py)

Output panel (long format, daily):
    data/processed/panels/daily/lob_proxy_daily.parquet

Schema per (date, asset):
    date, asset
    lob_l1_imb_mean, lob_l1_imb_std
    lob_l5_imb_mean, lob_l5_imb_std
    lob_spread_bps_mean, lob_spread_bps_p90
    lob_top_pressure_mean
    lob_count_imb_mean
    lob_run_length_p50
    lob_kyle_lambda_mean, lob_kyle_lambda_abs_max
    lob_n_bars                   (raw bar count for that day)

Why a separate panel: lob_proxy_panel.py emits BAR-level data (one row per
DIB bar, ~5-50 bars per day per asset). Chimera silver layer is daily-only;
joining bar-level data would explode rows or require per-bar timestamp
matching that's expensive. Daily aggregation is the right granularity.

The bar-level files remain on disk for strategies that need bar resolution
(read directly via panel parquet); this panel is the chimera-feed.
"""
from __future__ import annotations
import os

# CDAP contract
__contract__ = {
    "kind": "panel_builder",
    "stage": "lob_proxy_daily",
    "inputs": {
        "args": ["--force", "--assets", "--universe", "--dry-run"],
        "upstream": "data/processed/panels/daily/lob_proxy_<SYM>USDT_<YYYYMMDD>.parquet",
    },
    "outputs": {
        "files": "data/processed/panels/daily/lob_proxy_daily.parquet",
        "columns": ["date", "asset",
                    "lob_l1_imb_mean", "lob_l1_imb_std",
                    "lob_l5_imb_mean", "lob_l5_imb_std",
                    "lob_spread_bps_mean", "lob_spread_bps_p90",
                    "lob_top_pressure_mean",
                    "lob_count_imb_mean",
                    "lob_run_length_p50",
                    "lob_kyle_lambda_mean", "lob_kyle_lambda_abs_max",
                    "lob_n_bars"],
    },
    "invariants": {
        "atomic_write": True,
        "column_name_verify": True,
    },
}

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
PANELS_DIR = PROJECT_ROOT / "data" / "processed" / "panels" / "daily"
OUT = PANELS_DIR / "lob_proxy_daily.parquet"

BAR_LEVEL_GLOB = "lob_proxy_*USDT_*.parquet"

# Match `lob_proxy_<ASSET>USDT_<YYYYMMDD>.parquet` -> (asset, date_str).
FILE_RE = re.compile(r"^lob_proxy_([A-Za-z0-9]+?)USDT_(\d{8})$")


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("lob_daily", phase, message, **kw)


def _parse_filename(stem: str) -> tuple[str, str] | None:
    m = FILE_RE.match(stem)
    if not m:
        return None
    return m.group(1).upper(), m.group(2)


def build() -> pl.DataFrame:
    fps = sorted(PANELS_DIR.glob(BAR_LEVEL_GLOB))
    if not fps:
        raise FileNotFoundError(
            f"lob_proxy_daily: no bar-level files matching {BAR_LEVEL_GLOB} in "
            f"{PANELS_DIR}. Run lob_proxy_panel.py first.")

    rows = []
    n_files_ok = 0
    n_files_skip = 0
    for fp in fps:
        # Don't recurse on our own output.
        if fp.name == OUT.name:
            continue
        parsed = _parse_filename(fp.stem)
        if parsed is None:
            n_files_skip += 1
            continue
        asset, date_str = parsed
        try:
            df = pl.read_parquet(fp)
        except Exception:
            n_files_skip += 1
            continue
        if df.is_empty():
            n_files_skip += 1
            continue
        # Aggregate the bar-level features to one row per (date, asset).
        agg = {
            "date": datetime.strptime(date_str, "%Y%m%d").date(),
            "asset": asset,
            "lob_n_bars": df.height,
        }
        for col, op in [
            ("l1_imbalance_avg", "mean"),
            ("l5_imbalance_avg", "mean"),
            ("spread_bps_avg", "mean"),
            ("top_pressure_avg", "mean"),
            ("proxy_count_imb", "mean"),
            ("proxy_kyle_lambda", "mean"),
        ]:
            if col in df.columns:
                agg[f"lob_{col.replace('_avg','').replace('proxy_','')}_mean"] = (
                    float(df[col].mean()) if df[col].mean() is not None else None)
        # std on imbalances
        for col, name in [("l1_imbalance_avg", "lob_l1_imb_std"),
                            ("l5_imbalance_avg", "lob_l5_imb_std")]:
            if col in df.columns:
                agg[name] = float(df[col].std()) if df[col].std() is not None else None
        # spread p90
        if "spread_bps_avg" in df.columns:
            agg["lob_spread_bps_p90"] = float(df["spread_bps_avg"].quantile(0.9))
        # run length p50
        if "proxy_run_length" in df.columns:
            agg["lob_run_length_p50"] = float(df["proxy_run_length"].quantile(0.5))
        # kyle lambda abs max
        if "proxy_kyle_lambda" in df.columns:
            agg["lob_kyle_lambda_abs_max"] = float(df["proxy_kyle_lambda"].abs().max())
        rows.append(agg)
        n_files_ok += 1

    if not rows:
        raise RuntimeError(f"lob_proxy_daily: aggregated 0 rows from {len(fps)} files "
                           f"({n_files_skip} skipped)")

    print(f"[lob_daily] aggregated {n_files_ok} bar-level files -> {len(rows)} (date, asset) rows "
          f"({n_files_skip} skipped)", flush=True)

    # Normalize column names per contract (lob_l1_imbalance -> lob_l1_imb).
    # Build polars DataFrame with explicit schema so missing keys -> null.
    cols = ["date", "asset",
            "lob_l1_imb_mean", "lob_l1_imb_std",
            "lob_l5_imb_mean", "lob_l5_imb_std",
            "lob_spread_bps_mean", "lob_spread_bps_p90",
            "lob_top_pressure_mean",
            "lob_count_imb_mean",
            "lob_run_length_p50",
            "lob_kyle_lambda_mean", "lob_kyle_lambda_abs_max",
            "lob_n_bars"]
    # The aggregator keys above use names like "lob_l1_imbalance_mean" --
    # remap to canonical "lob_l1_imb_mean" etc.
    rename = {
        "lob_l1_imbalance_mean": "lob_l1_imb_mean",
        "lob_l5_imbalance_mean": "lob_l5_imb_mean",
        "lob_spread_bps_mean": "lob_spread_bps_mean",
        "lob_top_pressure_mean": "lob_top_pressure_mean",
        "lob_count_imb_mean": "lob_count_imb_mean",
        "lob_kyle_lambda_mean": "lob_kyle_lambda_mean",
    }
    out_rows = []
    for r in rows:
        new = {}
        for k, v in r.items():
            new[rename.get(k, k)] = v
        # Fill missing canonical keys with None.
        for c in cols:
            new.setdefault(c, None)
        out_rows.append({c: new.get(c) for c in cols})

    df = pl.DataFrame(out_rows).sort(["asset", "date"])
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="Force rebuild even if OUT panel is fresher than inputs.")
    # 2026-05-21 contract retrofit
    ap.add_argument("--assets", nargs="+", default=None,
                    help="No-op for cross-section panel. Accepted for pipeline uniformity.")
    ap.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                    help="No-op for cross-section panel. Accepted for pipeline uniformity.")
    ap.add_argument("--workers", type=int, default=1, help="Not used.")
    ap.add_argument("--dry-run", action="store_true", help="Print plan, no writes.")
    args = ap.parse_args()
    if args.assets or args.universe:
        print(f"[lob_daily] note: --assets/--universe accepted but no-op for cross-section panel",
              flush=True)

    # Skip-existing: OUT fresher than max(bar-level lob_proxy_panel mtimes)
    if OUT.exists() and not args.force:
        out_mtime = OUT.stat().st_mtime
        max_in = max((f.stat().st_mtime for f in PANELS_DIR.glob("lob_proxy_*.parquet")
                       if f != OUT), default=0.0)
        if out_mtime >= max_in:
            _pl("SKIP", f"skip: OUT panel fresher than bar-level lob inputs; --force to rebuild")
            return 0

    if args.dry_run:
        _pl("BUILD", f"DRY-RUN: would rebuild {OUT}")
        return 0

    _pl("SCAN", f"scanning {PANELS_DIR} for bar-level files...")
    df = build()
    print(f"[lob_daily] built: {df.height} rows, {len(df.columns)} cols, "
          f"{df.select('asset').n_unique()} assets, "
          f"{df.select('date').n_unique()} dates")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    written = set(pl.read_parquet_schema(tmp).keys())
    required = {"date", "asset", "lob_l1_imb_mean", "lob_spread_bps_mean",
                "lob_n_bars"}
    missing = required - written
    if missing:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"lob_proxy_daily missing required cols: {sorted(missing)}")
    if OUT.exists():
        OUT.unlink()
    os.replace(str(tmp), str(OUT))  # atomic overwrite (Windows-safe)
    _pl("OK", f"saved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
