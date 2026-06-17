"""Binance Vision daily bookDepth-profile backfill ingester (CORRECTED 2026-05-03).

Replaces the speculative `binance_vision_depth.py` (which assumed a
firstUpdateId/lastUpdateId raw-stream schema that does not exist on
binance.vision). VERIFIED via raw fetch 2026-05-03: actual schema is
a pre-aggregated DEPTH PROFILE at ±5% price bands.

Endpoint:
    https://data.binance.vision/data/futures/um/daily/bookDepth/<SYM>/
        <SYM>-bookDepth-<YYYY-MM-DD>.zip
    (UM perp; CM perp at .../futures/cm/daily/bookDepth/...)

VERIFIED schema (BTCUSDT-bookDepth-2026-04-30.csv inside the zip):
    timestamp,  percentage,  depth,  notional

  timestamp:  YYYY-MM-DD HH:MM:SS  (snapshot moment, ~3s cadence)
  percentage: distance from mid in %  (typically -5..-1, +1..+5  in 1% steps)
              negative = bid side, positive = ask side
  depth:      cumulative quantity of base currency within that % band
  notional:   cumulative quote-currency value within that % band

Coverage (verified 2026-05-03 via direct curl):
    First file: BTCUSDT-bookDepth-2023-01-01.zip   (3+ years history)
    Latest:     BTCUSDT-bookDepth-2026-05-01.zip   (current; daily refresh)
    File size:  ~520KB compressed -> ~1.84MB CSV -> ~28K-32K rows/day
    Cadence:    ~3 second snapshot interval per band
    Bands:      11 levels (-5%, -4%, ..., -1%, +1%, ..., +5%)

Output (atomic write per parquet_io contract):
    data/raw_external/binance_vision/depth_profile/<SYMBOL>/<DATE>.parquet
    Schema: ts_ms (i64), symbol (str), percentage (f64),
            depth (f64), notional (f64)

Per @browser B2 (no silent failures): every download attempt logs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Path bootstrap so the canonical cli helper imports when run as a direct script
# (refresh.py invokes producers directly; src/ is not on sys.path otherwise).
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src" / "pipeline"))
sys.path.insert(0, str(_ROOT / "src"))
from pipeline.cli import resolve_assets
from typing import Optional

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline.parquet_io import atomic_write_parquet, is_fresh
from pipeline.ingest._manifest import MissingManifest

OUT_ROOT = PROJECT_ROOT / "data" / "raw_external" / "binance_vision" / "depth_profile"
UA = "Mozilla/5.0 (compatible; v4_crypto_system/1.0; binance-vision-depth-profile)"

# Shared manifest helper (2026-05-24: extracted from inline to _manifest.py).
_mm = MissingManifest(OUT_ROOT)

__contract__ = {
    "kind": "ingester",
    "stage": "binance_vision_depth_profile",
    "inputs": {"upstream": "https://data.binance.vision/data/futures/um/daily/bookDepth/<SYM>/"},
    "outputs": {
        "files": "data/raw_external/binance_vision/depth_profile/<SYMBOL>/<DATE>.parquet",
        "columns": ["ts_ms", "symbol", "percentage", "depth", "notional"],
    },
    "invariants": {
        "no_lookahead": True,
        "T1_lag": False,  # Daily files are stable; no lookahead by construction
        "atomic_write": True,
        "no_silent_overwrite": True,  # Per parquet_io.is_fresh
    },
    "framework_helpers": ["atomic_write_parquet", "is_fresh"],
}


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("book_depth", phase, message, **kw)


# Sentinel: non-404 transient failure (5xx/timeout/network) distinct from a
# genuine 404 (None). Callers must NOT mark a _TRANSIENT date confirmed-missing.
_TRANSIENT = object()


def _download_zip(url: str, retries: int = 3):
    """Download zip. Returns: bytes on success; None on 404 (genuinely absent);
    _TRANSIENT sentinel on exhausted non-404 retries (do NOT persist as missing).

    Logs LOUD on every failure (B2 / no silent failures).
    """
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # 404 is a legitimate "not yet available" -- not a failure.
                return None
            last_err = e
        except Exception as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    print(f"  [DOWNLOAD ERR] {url}: {last_err}", flush=True)
    return _TRANSIENT


def _parse_csv(zip_bytes: bytes, symbol: str, date_str: str) -> Optional[pl.DataFrame]:
    """Extract the inner CSV from the zip, validate schema, return DataFrame.

    REJECTS the zip if columns don't match the verified schema; never
    silently coerces a wrong-shape file (per "trust the numbers").
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            csv_name = next((n for n in names if n.lower().endswith(".csv")), None)
            if csv_name is None:
                print(f"  [REJECT] no CSV inside zip for {symbol} {date_str}", flush=True)
                return None
            with zf.open(csv_name) as f:
                raw = f.read()
    except zipfile.BadZipFile:
        print(f"  [REJECT] bad zip for {symbol} {date_str}", flush=True)
        return None

    try:
        df = pl.read_csv(io.BytesIO(raw), infer_schema_length=1000)
    except Exception as e:
        print(f"  [REJECT] CSV parse error for {symbol} {date_str}: {e}", flush=True)
        return None

    expected = {"timestamp", "percentage", "depth", "notional"}
    actual = set(c.lower() for c in df.columns)
    if not expected.issubset(actual):
        print(f"  [REJECT] schema mismatch for {symbol} {date_str}: "
              f"expected {expected} got {actual}", flush=True)
        return None

    # Normalize column names to lowercase
    df = df.rename({c: c.lower() for c in df.columns})
    # Convert timestamp -> ts_ms (int64 epoch ms)
    df = df.with_columns([
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S",
                                          strict=False)
                            .dt.epoch(time_unit="ms")
                            .cast(pl.Int64).alias("ts_ms"),
        pl.lit(symbol).alias("symbol"),
        pl.col("percentage").cast(pl.Float64),
        pl.col("depth").cast(pl.Float64),
        pl.col("notional").cast(pl.Float64),
    ])
    return df.select(["ts_ms", "symbol", "percentage", "depth", "notional"])


def fetch_one_day(symbol: str, date: datetime.date, *,
                   futures_kind: str = "um", force: bool = False,
                   recheck_missing: bool = False) -> dict:
    """Download + parse + write one day of bookDepth for one symbol.

    Returns status dict: {"status": "OK"|"SKIP_FRESH"|"SKIP_KNOWN_MISSING"
                                  |"NOT_AVAILABLE"|"ERROR",
                           "rows": int, "out_path": str}

    confirmed_missing manifest (2026-05-24): if `date_str` is in the per-asset
    manifest's confirmed_missing AND marked <RECHECK_STALE_DAYS ago, skip the
    network call entirely. `--recheck-missing` (force flag) bypasses this.
    """
    sym = symbol.upper()
    date_str = date.strftime("%Y-%m-%d")
    out_dir = OUT_ROOT / sym
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}.parquet"

    if not force and out_path.exists() and is_fresh(out_path, force=False):
        return {"status": "SKIP_FRESH", "rows": 0, "out_path": str(out_path)}

    # Consult the manifest BEFORE the network call.
    if not force and not recheck_missing:
        if _mm.is_known_missing(sym, date_str):
            return {"status": "SKIP_KNOWN_MISSING", "rows": 0,
                    "out_path": str(out_path)}

    url = (f"https://data.binance.vision/data/futures/{futures_kind}/daily/"
           f"bookDepth/{sym}/{sym}-bookDepth-{date_str}.zip")
    raw = _download_zip(url)
    if raw is _TRANSIENT:
        # Transient (5xx/timeout): do NOT mark missing -- retry on next run.
        return {"status": "TRANSIENT_ERR", "rows": 0, "out_path": str(out_path)}
    if raw is None:
        # Genuine 404: persist so future runs skip it (until stale-recheck).
        _mm.mark_missing(sym, date_str)
        return {"status": "NOT_AVAILABLE", "rows": 0, "out_path": str(out_path)}

    df = _parse_csv(raw, sym, date_str)
    if df is None or df.is_empty():
        return {"status": "ERROR", "rows": 0, "out_path": str(out_path)}

    atomic_write_parquet(df, out_path,
                          required_cols=["ts_ms", "symbol", "percentage",
                                         "depth", "notional"])
    # If this date was previously confirmed_missing (e.g., file became
    # available after a Binance Vision republish), unmark it.
    _mm.unmark_missing(sym, date_str)
    return {"status": "OK", "rows": len(df), "out_path": str(out_path)}


def backfill(symbols: list[str], start: datetime.date, end: datetime.date,
              *, workers: int = 4, futures_kind: str = "um",
              force: bool = False, recheck_missing: bool = False) -> None:
    """Parallel daily backfill across (symbol, date) pairs."""
    days = [start + datetime.timedelta(days=i)
            for i in range((end - start).days + 1)]
    tasks = [(sym, d) for sym in symbols for d in days]
    print(f"[binance_vision_depth_profile] backfill {len(symbols)} symbols x "
          f"{len(days)} days = {len(tasks)} tasks  workers={workers}  "
          f"recheck_missing={recheck_missing}", flush=True)

    n_ok = n_skip = n_skip_missing = n_404 = n_err = 0
    rows_total = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one_day, sym, d,
                              futures_kind=futures_kind, force=force,
                              recheck_missing=recheck_missing): (sym, d)
                   for sym, d in tasks}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            sym, d = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"  [TASK ERR] {sym} {d}: {e}", flush=True)
                n_err += 1
                continue
            status = r["status"]
            if status == "OK":
                n_ok += 1
                rows_total += r["rows"]
            elif status == "SKIP_FRESH":
                n_skip += 1
            elif status == "SKIP_KNOWN_MISSING":
                n_skip_missing += 1
            elif status == "NOT_AVAILABLE":
                n_404 += 1
            else:
                n_err += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                el = time.time() - t0
                print(f"  [{i+1}/{len(tasks)}] elapsed={el/60:.1f}min  "
                      f"ok={n_ok} skip={n_skip} skip_missing={n_skip_missing} "
                      f"404={n_404} err={n_err}  "
                      f"rows={rows_total:,}", flush=True)
    print(f"\n[binance_vision_depth_profile] DONE  ok={n_ok}  skip={n_skip}  "
          f"skip_missing={n_skip_missing}  404={n_404}  err={n_err}  "
          f"rows={rows_total:,}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Binance Vision bookDepth-profile backfill")
    p.add_argument("--symbols", nargs="+", required=False,
                   default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                            "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"],
                   help="Symbols to backfill (UM-futures). Default: top-10 cohort.")
    p.add_argument("--start", default="2023-01-01",
                   help="Start date YYYY-MM-DD (default 2023-01-01 — Binance "
                        "Vision earliest availability for bookDepth. Was '2024-01-01' "
                        "pre-2026-05-24; that default silently skipped ~1 year of "
                        "available bd_* history. Per-asset listing-date 404s are "
                        "handled (skip if pre-listing).")
    p.add_argument("--end", default=None,
                   help="End date YYYY-MM-DD (default today UTC).")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel download workers (default 4; bandwidth-bounded).")
    p.add_argument("--futures-kind", default="um", choices=["um", "cm"],
                   help="UM (USDT-margined) or CM (coin-margined). Default um.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if output parquet exists.")
    p.add_argument("--recheck-missing", action="store_true",
                   help="Bypass the per-asset confirmed_missing manifest and re-attempt "
                        "every date. Use after a long gap (Binance Vision republishes do "
                        "happen) or to verify a previously-404'd date now succeeds.")
    # cli_universe_support contract: refresh.py passes --universe/--assets. Accept
    # them and resolve to <ASSET>USDT symbols (overrides --symbols). Without this the
    # canonical `refresh.py --target raw_book_depth --universe u100` invocation crashed
    # with "unrecognized arguments: --universe" (caught on the 2026-05-30 refresh run).
    p.add_argument("--universe", default=None,
                   help="Universe spec (u10/u50/u100) -> cohort <ASSET>USDT symbols.")
    p.add_argument("--assets", nargs="+", default=None,
                   help="Explicit asset list (btc eth ...) -> <ASSET>USDT; overrides --symbols.")
    args = p.parse_args()

    if args.universe or args.assets:
        args.symbols = resolve_assets(args, default=args.symbols, suffix="USDT",
                                      stage_name="book_depth")

    start = datetime.date.fromisoformat(args.start)
    end = (datetime.date.fromisoformat(args.end) if args.end
           else datetime.date.today() - datetime.timedelta(days=1))

    backfill(args.symbols, start, end,
             workers=args.workers, futures_kind=args.futures_kind,
             force=args.force, recheck_missing=args.recheck_missing)


if __name__ == "__main__":
    main()
