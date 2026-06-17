"""build_asset_own_regime_panel.py -- Step 1 of per-asset architecture.

For each asset in chimera 1d:
  Compute rolling 30d close-to-close return → bucket into regime label.
  Same cuts as btc_regime_30d: bull >= +5%, chop in (-5%, +5%), bear in
  (-15%, -5%], crash <= -15%.

Output: data/processed/asset_own_regime_panel.parquet
  Columns: asset, date, close, ret_30d, asset_own_regime

This is UNIVERSAL infrastructure — reusable across MA/EMA, RSI, Donchian, etc.

2026-05-21 contract retrofit: --assets / --universe / --force / --dry-run added.
"""
from __future__ import annotations

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "asset_own_regime_panel",
    "inputs": {
        "args": ["--assets", "--universe", "--force", "--dry-run"],
        "upstream": "data/processed/chimera/1d/*_v51_chimera_1d_*.parquet",
    },
    "outputs": {"file": "data/processed/asset_own_regime_panel.parquet"},
    "invariants": {
        "skip_if_fresher_than_inputs": True,
        "loud_force": True,
    },
}

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CHIMERA_1D = ROOT / "data" / "processed" / "chimera" / "1d"
OUT = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"


def regime_bucket(ret_30d: float) -> str:
    if pd.isna(ret_30d): return "unknown"
    if ret_30d <= -0.15: return "crash"
    if ret_30d <= -0.05: return "bear"
    if ret_30d >= 0.05:  return "bull"
    return "chop"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", nargs="+", default=None,
                    help="Restrict panel rebuild to these assets (BTC or BTCUSDT format). "
                         "Default: all chimera 1d files. NOTE: panel is full-rewrite; "
                         "--assets filters which assets get re-computed but other assets' "
                         "prior rows are PRESERVED via merge with existing panel.")
    ap.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                    help="Restrict via UniverseLoader. Default: all chimera 1d files.")
    ap.add_argument("--force", action="store_true",
                    help="Force full rebuild even if OUT is fresher than chimera inputs.")
    ap.add_argument("--workers", type=int, default=1, help="Not used (cheap cross-section).")
    ap.add_argument("--dry-run", action="store_true", help="Print plan, no writes.")
    return ap.parse_args()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parse_args()

    files = sorted(glob.glob(str(CHIMERA_1D / "*_v51_chimera_1d_*.parquet")))
    print(f"Found {len(files)} chimera 1d files")

    # Resolve --assets / --universe filter
    asset_filter: set[str] | None = None
    if args.assets:
        asset_filter = {a.upper().replace("USDT", "") for a in args.assets}
        print(f"[asset_own_regime] --assets filter: {len(asset_filter)} assets")
    elif args.universe:
        try:
            sys.path.insert(0, str(ROOT / "src" / "pipeline"))
            from universe_loader import UniverseLoader  # type: ignore
            asset_filter = {s.replace("USDT", "").upper()
                             for s in UniverseLoader.load().list(args.universe)}
            print(f"[asset_own_regime] --universe {args.universe}: {len(asset_filter)} assets")
        except Exception as e:
            print(f"[asset_own_regime] FALLBACK: --universe {args.universe} failed "
                  f"({e}); using all chimera files")
            asset_filter = None

    # Skip-existing: if OUT is fresher than max(chimera_1d mtimes) and not forced, skip.
    if OUT.exists() and not args.force and not args.assets:
        out_m = OUT.stat().st_mtime
        max_in = max((Path(f).stat().st_mtime for f in files), default=0.0)
        if out_m >= max_in:
            print(f"[asset_own_regime] skip: OUT fresher than chimera inputs; --force to rebuild")
            return 0

    if args.dry_run:
        print(f"[asset_own_regime] DRY-RUN: would rebuild {OUT}")
        if asset_filter:
            print(f"  Asset filter: {sorted(asset_filter)[:5]}{'...' if len(asset_filter)>5 else ''}")
        return 0

    all_rows = []
    n_processed = 0
    for i, f in enumerate(files):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        if asset_filter and sym not in asset_filter:
            continue
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception as e:
            print(f"  [skip] {sym}: {e}")
            continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        df["asset"] = sym
        df["ret_30d"] = df["close"].pct_change(30)
        df["asset_own_regime"] = df["ret_30d"].apply(regime_bucket)
        all_rows.append(df[["asset", "date", "close", "ret_30d", "asset_own_regime"]])
        n_processed += 1
        if n_processed % 20 == 0:
            print(f"  {n_processed} processed")

    if not all_rows:
        print("[FATAL] no asset data loaded")
        return 2

    new_panel = pd.concat(all_rows, ignore_index=True)

    # If filtering, merge with existing panel rows for unaffected assets.
    if asset_filter and OUT.exists() and not args.force:
        try:
            existing = pl.read_parquet(OUT).to_pandas()
            existing = existing[~existing["asset"].isin(asset_filter)]
            panel = pd.concat([existing, new_panel], ignore_index=True)
            print(f"[asset_own_regime] merged: kept {existing['asset'].nunique()} unchanged "
                  f"assets + {new_panel['asset'].nunique()} rebuilt")
        except Exception as e:
            print(f"[asset_own_regime] merge failed ({e}); writing partial panel")
            panel = new_panel
    else:
        panel = new_panel

    print(f"\nPanel rows: {len(panel):,}")
    print(f"Assets: {panel['asset'].nunique()}")
    print(f"Date range: {panel['date'].min()} -> {panel['date'].max()}")
    print(f"Regime distribution:")
    print(panel["asset_own_regime"].value_counts().to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)
    print(f"\n[OK] wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
