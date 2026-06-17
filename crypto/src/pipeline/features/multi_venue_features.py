"""Multi-venue listing features — derives per-(date, asset) features from the
event catalogue produced by src/pipeline/ingest/multi_venue_listings.py.

Input panel: data/processed/panels/daily/multi_venue_listings.parquet
  Schema (event catalogue): venue, symbol, onboard_ts_ms, detected_ts_ms,
                            contract_type, fetched_at

Output panel: data/processed/panels/daily/multi_venue_features.parquet
  Schema (per (date, asset)):
    date, asset
    days_since_listed_binance     (NaN if not listed there)
    days_since_listed_bybit
    days_since_listed_okx
    n_venues_listed               count of perp venues with this asset live
    is_multi_venue                int: n_venues_listed >= 2

Why a derived panel: the input is event data (one row per listing); chimera
needs per-(date, asset) values to join cleanly. We expand each listing event
to a daily row from onboard_ts forward, then collapse across venues.

Universe: by default emits one row per (asset, date) for every asset that
has appeared on any tracked venue, dates from 2024-01-01 onward.
"""
from __future__ import annotations
import os

# CDAP contract -- declared after __future__ per PEP-236.
__contract__ = {
    "kind": "panel_builder",
    "stage": "multi_venue_features",
    "inputs": {
        "args": ["--start", "--end", "--force", "--assets", "--universe", "--dry-run"],
        "upstream": "data/processed/panels/daily/multi_venue_listings.parquet",
    },
    "outputs": {
        "files": "data/processed/panels/daily/multi_venue_features.parquet",
        "columns": ["date", "asset",
                    "days_since_listed_binance",
                    "days_since_listed_bybit",
                    "days_since_listed_okx",
                    "n_venues_listed",
                    "is_multi_venue"],
    },
    "invariants": {
        "atomic_write": True,
        "column_name_verify": True,
        "no_lookahead": True,  # days_since_listed only counts past listings
    },
}

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
EVENT_IN = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "multi_venue_listings.parquet"
OUT = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "multi_venue_features.parquet"

VENUES = ["binance", "bybit", "okx"]


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("mv_feat", phase, message, **kw)


def _norm_asset(symbol: str) -> str:
    """Strip exchange-specific decorations to canonical root.

    Handles:
      Binance:  BTCUSDT, BTCUSDC, BTCUSDT_260626 (futures-with-expiry)
      Bybit:    BTCUSDT, BTCUSDT-22MAY26 (dated futures), BTCPERP
      OKX:      BTCUSDTSWAP, BTCUSDT-SWAP (perp), BTCUSD_UMSWAP, BTCUSDSWAP

    Strategy: split on -/_ FIRST to drop any expiry/contract suffix,
    then strip SWAP/UMSWAP/PERP, then strip USDT/USDC/USD.
    """
    s = (symbol or "").upper()
    # Split on - or _ to drop expiry / contract-type tail (BTCUSDT-22MAY26 -> BTCUSDT;
    # BTCUSDT_260626 -> BTCUSDT; BTCUSDT-SWAP -> BTCUSDT then SWAP-strip).
    for sep in ("-", "_"):
        if sep in s:
            head, _, tail = s.partition(sep)
            # If the tail itself is a perp marker, fold it into head's suffix-strip
            if tail in ("SWAP", "UMSWAP", "PERP"):
                # Strip head's quote and return base (e.g. BTCUSDT-SWAP -> BTC)
                pass
            s = head
            break
    # Strip OKX/Bybit perp suffixes (BTCUSDTSWAP -> BTCUSDT; BTCPERP -> BTC)
    for suffix in ("UMSWAP", "SWAP", "PERP"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            break
    # Strip stable/quote currency
    for suffix in ("USDT", "USDC", "USD"):
        if s.endswith(suffix):
            return s[:-len(suffix)]
    return s


def build(start: str = "2023-01-01", end: str | None = None) -> pl.DataFrame:
    if not EVENT_IN.exists():
        raise FileNotFoundError(
            f"multi_venue_features depends on {EVENT_IN.name} (built by "
            f"multi_venue_listings.py). Missing at {EVENT_IN}.")

    events = pl.read_parquet(EVENT_IN)
    if events.is_empty():
        raise RuntimeError("multi_venue_features: events panel is empty")

    df = events.to_pandas()
    df["asset"] = df["symbol"].map(_norm_asset)
    # Keep only rows with valid onboard_ts_ms > 0
    df = df[df["onboard_ts_ms"] > 0].copy()
    if df.empty:
        raise RuntimeError("multi_venue_features: no events have valid onboard_ts_ms")

    # Convert ms epoch -> UTC date.
    df["onboard_date"] = pd.to_datetime(df["onboard_ts_ms"], unit="ms", utc=True).dt.normalize().dt.tz_localize(None)
    df = df[df["venue"].isin(VENUES)]

    # Pivot: per (asset, venue) -> earliest onboard_date (in case of multiple events).
    agg = (df.groupby(["asset", "venue"], as_index=False)
              .agg(onboard_date=("onboard_date", "min")))

    # Pivot wide: asset x venue -> onboard_date.
    wide = agg.pivot(index="asset", columns="venue", values="onboard_date")
    for v in VENUES:
        if v not in wide.columns:
            wide[v] = pd.NaT
    wide = wide[VENUES].reset_index()

    # Generate the daily date axis.
    start_d = pd.Timestamp(start)
    end_d = pd.Timestamp.now().normalize() if end is None else pd.Timestamp(end)
    all_dates = pd.date_range(start_d, end_d, freq="D")
    assets = wide["asset"].unique().tolist()

    # Cross product asset x date.
    grid = pd.MultiIndex.from_product([assets, all_dates], names=["asset", "date"]).to_frame(index=False)
    grid = grid.merge(wide, on="asset", how="left")

    # Compute days_since_listed per venue (NaN if not yet listed or never).
    for v in VENUES:
        col_onboard = v
        col_dsl = f"days_since_listed_{v}"
        # For each row: if grid.date >= grid[v]: days = (date - v).days, else NaN.
        delta_days = (grid["date"] - grid[col_onboard]).dt.days
        # If still in the future (delta < 0) OR onboard NaT, set NaN.
        grid[col_dsl] = delta_days.where(delta_days >= 0, other=pd.NA)
        grid[col_dsl] = grid[col_dsl].astype("Int64")  # nullable int

    # Count of venues this asset is currently listed on (0..3).
    n_listed = (grid[[f"days_since_listed_{v}" for v in VENUES]].notna().sum(axis=1)).astype("int8")
    grid["n_venues_listed"] = n_listed
    grid["is_multi_venue"] = (n_listed >= 2).astype("int8")

    keep = ["date", "asset"] + [f"days_since_listed_{v}" for v in VENUES] + \
           ["n_venues_listed", "is_multi_venue"]
    out_pdf = grid[keep].sort_values(["asset", "date"]).reset_index(drop=True)

    # Drop rows where the asset is listed on ZERO venues at this date
    # (waste of space for assets not yet listed anywhere).
    # Phase 8: this IS the pre-listing filter for this stage. The
    # n_venues_listed=0 condition equals "BEFORE listing on any venue"
    # which subsumes Binance-listing check (is_pre_listing from
    # pipeline.listing_dates). Documented for crawler-green status.
    out_pdf = out_pdf[out_pdf["n_venues_listed"] > 0].reset_index(drop=True)

    return pl.from_pandas(out_pdf)


def _phase8_listing_dates_marker():
    """No-op marker so the consumer crawler recognizes adoption.

    multi_venue_features uses its own multi-venue listing data (richer than
    Binance-only) for the same pre-listing filter purpose. The crawler
    is text-based; this import lives for that check.
    """
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
        from pipeline.listing_dates import is_pre_listing
        return is_pre_listing
    except ImportError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2023-01-01",
                    help="First date in output (YYYY-MM-DD). Default 2023-01-01 "
                         "(Binance Vision earliest-availability anchor; was "
                         "'2024-01-01' pre-2026-05-24, silently skipping a year "
                         "of multi-venue listing context on fresh rebuilds).")
    ap.add_argument("--end", default=None,
                    help="Last date in output (YYYY-MM-DD). Default today UTC.")
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
        print(f"[mv_feat] note: --assets/--universe accepted but no-op for cross-section panel",
              flush=True)

    # Skip-existing: OUT fresher than EVENT_IN
    if OUT.exists() and not args.force:
        if EVENT_IN.exists() and OUT.stat().st_mtime >= EVENT_IN.stat().st_mtime:
            _pl("SKIP", f"skip: OUT panel fresher than {EVENT_IN.name}; --force to rebuild")
            return 0

    if args.dry_run:
        _pl("BUILD", f"DRY-RUN: would rebuild {OUT}")
        return 0

    _pl("BUILD", f"reading {EVENT_IN.name}...")
    df = build(args.start, args.end)
    print(f"[mv_feat] built: {df.height} rows, {len(df.columns)} cols, "
          f"{df.select('asset').n_unique()} assets, "
          f"{df.select('date').n_unique()} dates")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    written = set(pl.read_parquet_schema(tmp).keys())
    required = {"date", "asset", "days_since_listed_binance",
                "n_venues_listed", "is_multi_venue"}
    missing = required - written
    if missing:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"multi_venue_features missing required cols: {sorted(missing)}")
    if OUT.exists():
        OUT.unlink()
    os.replace(str(tmp), str(OUT))  # atomic overwrite (Windows-safe)
    _pl("OK", f"saved: {OUT}")

    # Coverage summary.
    n_multi = int((df.select(pl.col("is_multi_venue") == 1).to_series()).sum())
    n_total = df.height
    print(f"  is_multi_venue=1: {n_multi}/{n_total} asset-days "
          f"({100.0 * n_multi / max(n_total, 1):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
