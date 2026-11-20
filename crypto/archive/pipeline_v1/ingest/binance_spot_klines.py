"""Binance spot daily klines from data.binance.vision (free, no key).

Monthly archive at:
    https://data.binance.vision/data/spot/monthly/klines/<SYMBOL>/1d/<SYMBOL>-1d-YYYY-MM.zip

CSV columns (Binance kline format):
    open_time, open, high, low, close, volume, close_time, quote_volume,
    num_trades, taker_buy_base, taker_buy_quote, ignore

We extract close prices to compute spot-perp basis downstream.

Output:
    data/processed/panels/daily/spot_klines_daily.parquet
        columns: date, asset, spot_close

Universe-aware: defaults to u10; pass --universe u50 / u100 or --assets to widen.
"""
from __future__ import annotations
import os

# CDAP contract -- declared after __future__ per PEP-236.
__contract__ = {
    "kind": "panel_builder",
    "stage": "binance_spot_klines",
    "inputs": {
        "args": ["--universe", "--assets", "--start", "--force"],
        "upstream": "https://data.binance.vision/data/spot/monthly/klines/...",
    },
    "outputs": {
        "files": "data/processed/panels/daily/spot_klines_daily.parquet",
        "columns": ["date", "asset", "spot_close"],
    },
    "invariants": {
        "atomic_write": True,
        "column_name_verify": True,
    },
}

import argparse
import concurrent.futures
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline.ingest._manifest import MissingManifest

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "spot_klines_daily.parquet"

# Manifest root: per-symbol sub-dirs under spot_klines_manifests/.
# Key = YYYY-MM (monthly granularity, matches the monthly zip fetch unit).
_MANIFEST_ROOT = OUT_DIR / "spot_klines_manifests"
_mm = MissingManifest(_MANIFEST_ROOT)

BASE = "https://data.binance.vision/data/spot/monthly/klines"
UA = "v4-pipeline/1.0"

DEFAULT_U10 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
               "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("spot", phase, message, **kw)


def fetch_month(symbol: str, ym: str, retries: int = 3,
                recheck_missing: bool = False) -> list[dict]:
    """Fetch one month of daily klines for one symbol. Returns [] on 404 or failure.

    confirmed_missing manifest: if (symbol, ym) was previously 404'd and the
    mark is still fresh, skip the network call unless recheck_missing=True.
    """
    if not recheck_missing and _mm.is_known_missing(symbol, ym):
        return []
    url = f"{BASE}/{symbol}/1d/{symbol}-1d-{ym}.zip"
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                content = r.read()
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                with z.open(z.namelist()[0]) as f:
                    text = f.read().decode()
            rows = []
            for line in text.splitlines():
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                try:
                    ts = int(parts[0])
                    close = float(parts[4])
                except ValueError:
                    continue  # skip header if present
                # Binance switched ms -> us at some point in 2025; autodetect.
                unit = "us" if ts > 1e15 else "ms"
                rows.append({
                    "date": pd.to_datetime(ts, unit=unit).normalize(),
                    "asset": symbol.replace("USDT", ""),
                    "spot_close": close,
                })
            # Successful fetch: clear any stale manifest entry.
            _mm.unmark_missing(symbol, ym)
            return rows
        except urllib.error.HTTPError as e:
            if e.code == 404:
                _mm.mark_missing(symbol, ym)
                return []
            time.sleep(1 + i)
        except Exception:
            time.sleep(1 + i)
    return []


def fetch_asset(symbol: str, start: str = "2020-01", end: str | None = None,
                workers: int = 8, recheck_missing: bool = False) -> pd.DataFrame:
    """Fetch all months of daily klines for one symbol via ThreadPool."""
    start_d = pd.Timestamp(start + "-01")
    end_d = pd.Timestamp.now() if end is None else pd.Timestamp(end + "-01")
    # Phase 8 centralized pre-listing skip: shift start_d forward to the
    # asset's Binance listing if start_d is earlier. Saves N pre-listing
    # month-fetches per asset.
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
        from pipeline.listing_dates import get_listing_date
        listing = pd.Timestamp(get_listing_date(symbol))
        if start_d < listing:
            print(f"[{symbol}] pre-listing skip: clamping start from "
                  f"{start_d.date()} to {listing.date()}", flush=True)
            start_d = listing
    except Exception:
        pass
    months = pd.date_range(start_d, end_d, freq="MS").strftime("%Y-%m").tolist()
    all_rows = []
    ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_month, symbol, ym,
                          recheck_missing=recheck_missing): ym for ym in months}
        for fut in concurrent.futures.as_completed(futs):
            rows = fut.result()
            if rows:
                all_rows.extend(rows)
                ok += 1
    if not all_rows:
        return pd.DataFrame(columns=["date", "asset", "spot_close"])
    df = (pd.DataFrame(all_rows)
            .drop_duplicates(subset="date")
            .sort_values("date")
            .reset_index(drop=True))
    _pl("OK", f"{symbol}: fetched {ok}/{len(months)} months = {len(df)} daily rows")
    return df


def _resolve_universe(args) -> list[str]:
    """Resolve symbol list from --assets / --universe / default u10."""
    if args.assets:
        return [(a.upper() if a.upper().endswith("USDT") else a.upper() + "USDT")
                for a in args.assets]
    if args.universe:
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
            from universe_loader import UniverseLoader
            syms = [s.upper() for s in UniverseLoader.load().list(args.universe)]
            _pl("BUILD", f"universe: {args.universe} ({len(syms)} assets)")
            return syms
        except Exception as e:
            print(f"[FALLBACK] universe={args.universe} load failed ({e}); using u10",
                  flush=True)
    _pl("BUILD", f"universe: u10-default ({len(DEFAULT_U10)} assets)")
    return list(DEFAULT_U10)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                    help="Resolve symbols via UniverseLoader (default u10).")
    ap.add_argument("--assets", nargs="+", default=None,
                    help="Explicit symbol list (overrides --universe).")
    ap.add_argument("--start", default="2020-01", help="Start year-month YYYY-MM.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Per-symbol month-fetch ThreadPool workers (HTTP concurrency "
                         "WITHIN each symbol's monthly fetches). Default 8.")
    ap.add_argument("--symbol-workers", type=int, default=4,
                    help="Symbol-level concurrency: how many DIFFERENT symbols to "
                         "fetch in parallel. Default 4. Total HTTP threads = "
                         "workers x symbol-workers; cap at ~16 to stay under "
                         "Binance Vision per-IP rate limit.")
    ap.add_argument("--force", action="store_true",
                    help="No-op: this script always rebuilds the spot-klines "
                         "panel via atomic-write (overwrites). Accepted for "
                         "uniform refresh.py orchestration.")
    ap.add_argument("--recheck-missing", action="store_true",
                    help="Bypass the confirmed_missing manifest and re-attempt "
                         "every previously-404'd month.")
    args = ap.parse_args()

    assets = _resolve_universe(args)

    print(f"\n{'='*70}")
    print(f"BUILD SPOT KLINES PANEL  start={args.start}  |U|={len(assets)}")
    print(f"{'='*70}\n")

    # Compound parallelism: --symbol-workers symbols in flight, each with
    # --workers months in flight. Total HTTP threads = symbol_workers x workers.
    sym_workers = max(1, min(args.symbol_workers, len(assets)))
    frames = []
    n_empty = 0
    completed = 0
    if sym_workers <= 1:
        for i, a in enumerate(assets, 1):
            _pl("BUILD", f"{i}/{len(assets)}: {a}...")
            df = fetch_asset(a, start=args.start, workers=args.workers,
                             recheck_missing=args.recheck_missing)
            if len(df) > 0:
                frames.append(df)
            else:
                n_empty += 1
                _pl("SKIP", f"{a}: no spot klines available")
    else:
        print(f"[spot] parallelism: {sym_workers} symbols x {args.workers} months = "
              f"{sym_workers * args.workers} concurrent HTTP fetches", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=sym_workers) as ex:
            futures = {ex.submit(fetch_asset, a, args.start, None, args.workers,
                                 args.recheck_missing): a
                        for a in assets}
            for fut in concurrent.futures.as_completed(futures):
                a = futures[fut]
                completed += 1
                try:
                    df = fut.result()
                except Exception as e:
                    _pl("FAIL", f"{a}: ERROR: {type(e).__name__}: {e}")
                    n_empty += 1
                    continue
                if len(df) > 0:
                    frames.append(df)
                    _pl("BUILD", f"{completed}/{len(assets)}: {a}: {len(df)} rows")
                else:
                    n_empty += 1
                    _pl("BUILD", f"{completed}/{len(assets)}: {a}: no spot klines available")

    if not frames:
        print("[spot] HARD FAIL: no data fetched for any asset", flush=True)
        sys.exit(2)

    panel = (pd.concat(frames, ignore_index=True)
               .sort_values(["asset", "date"])
               .reset_index(drop=True))

    # G-AUDIT-020: atomic-tmp-rename + column-name verify.
    tmp = OUT_PATH.with_suffix(".parquet.tmp")
    panel.to_parquet(tmp, index=False)
    import pyarrow.parquet as _pq
    written = set(_pq.read_schema(tmp).names)
    required = {"date", "asset", "spot_close"}
    missing = required - written
    if missing:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"spot_klines panel missing required cols: {sorted(missing)}")
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    os.replace(str(tmp), str(OUT_PATH))  # atomic overwrite (Windows-safe)
    print(f"\n[spot] saved: {OUT_PATH}")
    print(f"  rows: {len(panel)}, assets: {panel['asset'].nunique()}, "
          f"date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    if n_empty > 0:
        print(f"  WARN: {n_empty}/{len(assets)} assets had no spot data "
              f"(perp-only listings; basis will be NaN for these)", flush=True)
        # Partial coverage is acceptable for spot data (some perp assets have
        # no spot listing). Don't fail.
    return 0


if __name__ == "__main__":
    sys.exit(main())
