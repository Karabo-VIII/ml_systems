"""Binance historical futures metrics ingest from data.binance.vision S3 bucket.

GOLDEN DATASET: public S3 bucket, no API key, no rate limit. Data from 2020-09-01
for all actively-traded USDT perpetuals. 5-minute resolution, ~288 rows/day.

Columns available per file:
    create_time
    symbol
    sum_open_interest                    (coins)
    sum_open_interest_value              (USD)
    count_toptrader_long_short_ratio     (top-trader ACCOUNT ratio long/short)
    sum_toptrader_long_short_ratio       (top-trader POSITION ratio long/short)
    count_long_short_ratio               (global account ratio)
    sum_taker_long_short_vol_ratio       (taker buy/sell volume ratio)

Strategy-relevant fields:
    - sum_toptrader_long_short_ratio: what smart traders' POSITIONS look like
    - count_toptrader_long_short_ratio: what fraction of smart accounts are long
    - count_long_short_ratio: retail consensus (often contrarian)
    - sum_taker_long_short_vol_ratio: aggressive-order bias (momentum)

Output:
    data/frontier/metrics/s3_metrics_daily_{asset}.parquet (per asset)
    data/frontier/metrics/s3_metrics_panel.parquet (long-format all assets)

Download plan: 54 assets x ~1500 days x ~11KB = ~900MB total.
Rate limit: S3 is effectively unlimited. 16 concurrent downloads safe.
"""
from __future__ import annotations

import concurrent.futures
import io
import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from pipeline.ingest._manifest import MissingManifest

OUT_DIR = ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Manifest lives in its own subdirectory so it doesn't pollute the panel dir.
# Per-symbol manifests: data/processed/panels/daily/s3_metrics_manifests/<SYM>/_manifest.json
_MANIFEST_ROOT = OUT_DIR / "s3_metrics_manifests"
_mm = MissingManifest(_MANIFEST_ROOT)

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
UA = "v4_crypto_stystem-frontier/1.0"
START_DATE = "2022-01-01"  # reduce download; 2020-09 available but slow


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("s3_metrics", phase, message, **kw)


def _download_day(symbol: str, date: pd.Timestamp,
                  recheck_missing: bool = False) -> pd.DataFrame | None:
    ds = date.strftime("%Y-%m-%d")
    # confirmed_missing manifest: skip known-missing dates unless recheck requested.
    if not recheck_missing and _mm.is_known_missing(symbol, ds):
        return None  # caller treats None == no data for this date
    url = f"{BASE}/{symbol}/{symbol}-metrics-{ds}.zip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                df = pd.read_csv(f)
        # Successful fetch -- unmark if it was previously confirmed missing.
        _mm.unmark_missing(symbol, ds)
        return df
    except urllib.error.HTTPError as e:
        # 404 = day not published; persist to manifest to avoid re-attempts.
        if e.code == 404:
            _mm.mark_missing(symbol, ds)
        else:
            print(f"  [s3_metrics_dl] {symbol} {ds} HTTP {e.code}: {str(e)[:80]}", flush=True)
        return None
    except Exception as e:
        print(f"  [s3_metrics_dl] {symbol} {ds} {type(e).__name__}: {str(e)[:80]}", flush=True)
        return None


def fetch_asset_history(symbol: str, start_date: str = START_DATE,
                        end_date: str | None = None, max_workers: int = 16,
                        recheck_missing: bool = False) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    dates = pd.date_range(start_date, end_date, freq="D")
    # Phase 8 centralized pre-listing skip: strip dates BEFORE asset's
    # Binance listing. Saves N x M wasted S3 HEAD requests per asset.
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
        from pipeline.listing_dates import is_pre_listing as _is_pre_listing
        n_before = len(dates)
        dates = pd.DatetimeIndex([d for d in dates
                                    if not _is_pre_listing(symbol, d.to_pydatetime())])
        n_skipped = n_before - len(dates)
        if n_skipped > 0:
            print(f"[{symbol}] pre-listing skip: dropped {n_skipped} dates "
                  f"(before Binance listing)", flush=True)
    except Exception:
        pass    # fallback: process all dates (legacy behavior)
    _pl("START", f"{symbol}: fetching {len(dates)} days from {start_date}...")

    frames = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_download_day, symbol, d, recheck_missing): d for d in dates}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            df = fut.result()
            if df is not None and len(df) > 0:
                frames.append(df)
            if i % 200 == 0:
                _pl("OK", f"{symbol}: {i}/{len(dates)} complete")

    if not frames:
        _pl("BUILD", f"{symbol}: no data")
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    all_df["create_time"] = pd.to_datetime(all_df["create_time"])
    all_df["date"] = all_df["create_time"].dt.normalize()

    # Aggregate to daily means
    daily = (
        all_df.groupby("date")
        .agg({
            "sum_open_interest": "mean",
            "sum_open_interest_value": "mean",
            "count_toptrader_long_short_ratio": "mean",
            "sum_toptrader_long_short_ratio": "mean",
            "count_long_short_ratio": "mean",
            "sum_taker_long_short_vol_ratio": "mean",
        })
        .reset_index()
    )
    daily["asset"] = symbol.replace("USDT", "")
    daily = daily.rename(columns={
        "sum_open_interest": "oi",
        "sum_open_interest_value": "oi_usd",
        "count_toptrader_long_short_ratio": "top_acct_lsr",
        "sum_toptrader_long_short_ratio": "top_pos_lsr",
        "count_long_short_ratio": "global_lsr",
        "sum_taker_long_short_vol_ratio": "taker_lsr",
    })
    return daily


def main():
    # Universe resolution + framework primitives.
    import argparse
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    # NOTE: do NOT alias polars as `_pl` — that name is the module-level
    # phase_log helper; shadowing it breaks all _pl() calls in this scope
    # (commit 2026-05-23 fix; pre-fix symptom: TypeError: 'module' object
    # is not callable at line 183).
    import polars as pl
    from parquet_io import atomic_write_parquet, safe_unlink
    from cli import add_standard_args, resolve_assets

    ap = argparse.ArgumentParser()
    add_standard_args(ap, default_workers=16, date_window=False)
    ap.add_argument("--recheck-missing", action="store_true",
                    help="Bypass the confirmed_missing manifest and re-attempt "
                         "every previously-404'd date.")
    args = ap.parse_args()

    assets = resolve_assets(args, stage_name="s3_metrics")  # default = DEFAULT_U10

    panel_path = OUT_DIR / "s3_metrics_panel.parquet"
    # G-AUDIT-022: --force must invalidate the merged panel too, not just
    # per-asset caches. Otherwise a partial-failure rerun leaves the OLD
    # panel on disk while content-hash marks it "fresh".
    if args.force and safe_unlink(panel_path):
        print(f"[s3_metrics] --force: removed stale panel {panel_path.name}",
              flush=True)

    # Outer-loop parallelism: per-asset fetches run concurrently, sharing the
    # global args.workers budget (outer * inner <= args.workers). Previously
    # the outer was strictly serial, leaving 50 assets at 30-60s each =
    # 25-50min mostly-idle. Outer cap = min(4, len(assets)) keeps inner
    # threading meaningful for date-range fan-out.
    outer = max(1, min(4, len(assets)))
    inner = max(1, args.workers // outer)

    def _one(a: str):
        out_path = OUT_DIR / f"s3_metrics_daily_{a}.parquet"
        if args.force and safe_unlink(out_path):
            _pl("BUILD", f"{a} --force: removed stale cache")
        if out_path.exists():
            _pl("BUILD", f"{a} cache hit, loading...")
            return a, pd.read_parquet(out_path), None
        t0 = time.time()
        try:
            d = fetch_asset_history(a, max_workers=inner,
                                    recheck_missing=args.recheck_missing)
        except Exception as e:
            return a, None, f"fetch_asset_history raised: {e}"
        if len(d) > 0:
            atomic_write_parquet(pl.from_pandas(d), out_path)
            print(f"[s3_metrics] {a} saved {len(d)} days in {time.time()-t0:.0f}s",
                  flush=True)
            return a, d, None
        return a, None, "zero rows fetched after retries"

    all_daily = []
    if outer == 1:
        for a in assets:
            _, d, err = _one(a)
            if err:
                _pl("FAIL", f"{a} FAIL: {err}")
            elif d is not None:
                all_daily.append(d)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=outer) as ex:
            futs = {ex.submit(_one, a): a for a in assets}
            for fut in as_completed(futs):
                a = futs[fut]
                try:
                    _, d, err = fut.result()
                except Exception as e:
                    _pl("BUILD", f"{a} CRASH: {e}")
                    continue
                if err:
                    _pl("FAIL", f"{a} FAIL: {err}")
                elif d is not None:
                    all_daily.append(d)

    if not all_daily:
        print("[s3_metrics] ERROR: nothing fetched -- aborting (no panel written)",
              flush=True)
        sys.exit(2)

    panel = (pd.concat(all_daily, ignore_index=True)
             .sort_values(["asset", "date"]).reset_index(drop=True))
    atomic_write_parquet(pl.from_pandas(panel), panel_path,
                          required_cols={"date", "asset"})
    print(f"[s3_metrics] panel saved: {panel_path} ({len(panel)} rows, "
          f"{panel['asset'].nunique()} assets)", flush=True)
    print(f"  date range: {panel['date'].min().date()} -> {panel['date'].max().date()}",
          flush=True)


if __name__ == "__main__":
    main()
