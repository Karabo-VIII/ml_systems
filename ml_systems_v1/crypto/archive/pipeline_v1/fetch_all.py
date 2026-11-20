# CDAP contract — declared up-front so audit can validate without imports.
__contract__ = {
    "kind": "pipeline_stage",
    "stage": "fetch_binance",
    "inputs": {
        "args": ["--workers", "--assets", "--universe {u10|u50|u100}",
                 "--start-date", "--top-n", "--from-screener"],
        "config_keys": ["data.start_date", "data.assets"],
    },
    "outputs": {
        "files": "data/raw/<SYMBOL>USDT/{aggTrades,funding,metrics}/*.parquet",
        "row_unit": "one parquet per (asset, day)",
    },
    "invariants": {
        "ts_unit_per_row_autodetect": True,    # Binance ms->us 2024-2025 switch
        "atomic_write": True,                   # zip->parse->parquet
    },
    "rationale": "Bronze layer; everything downstream depends on this output schema.",
}

import requests
import zipfile
import polars as pl
import yaml
import json
import time
import io
import sys
import os
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# PROXY SETTINGS
PROXIES = {}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

# --- SMART SKIP LIST ---
METRICS_EARLIEST_DATE = datetime(2021, 1, 1)

# Phase 8: ASSET_LAUNCH_DATES + cache + _resolve_launch_date + _resolve_fapi_symbol
# now centralized in src/pipeline/listing_dates.py. Import-only shims below so
# any external callers depending on these names continue to work.
# Path-fixup: fetch_all.py is invoked as top-level script (not via -m), so
# the parent of `pipeline/` must be on sys.path for the centralized import.
import sys as _sys
from pathlib import Path as _Path
_PIPELINE_PARENT = str(_Path(__file__).resolve().parents[1])
if _PIPELINE_PARENT not in _sys.path:
    _sys.path.insert(0, _PIPELINE_PARENT)

# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper.
# fetch_all bootstraps src/ on sys.path (line 63: _PIPELINE_PARENT), so the
# fully-qualified import works. Only out-of-tqdm prints migrated; in-tqdm
# tqdm.write() lines preserved for bar interleaving.
def _fetch_pl(phase, message, **kw):
    from pipeline.progress import phase_log
    phase_log("fetch", phase, message, **kw)

from pipeline.listing_dates import (
    ASSET_LAUNCH_DATES,
    get_listing_date as _ld_get_listing_date,
    resolve_fapi_symbol as _ld_resolve_fapi_symbol,
    LAUNCH_DATES_CACHE_PATH as _LAUNCH_DATES_CACHE_PATH,
    _load_cache as _load_launch_dates_cache,
)

# 2026-05-24: lift atomic_write_parquet to module-level so the 4 download
# call-sites (funding bulk, parallel download success path, API funding
# fallback, API metrics fallback) all use the same tmp-rename contract.
# Pre-fix, those sites used plain df.write_parquet(path) -- a kill/crash
# mid-write left a half-written parquet that downstream zip-detect would
# read as "exists, skip" and forever skip the day. With atomic_write_parquet
# the partial write lives at <path>.tmp and is cleaned up on next run.
from pipeline.parquet_io import atomic_write_parquet as _atomic_write_parquet


def _resolve_launch_date(symbol: str,
                          default: datetime = datetime(2019, 9, 25)) -> datetime:
    """Backwards-compat shim. Centralized in pipeline.listing_dates.

    Priority:
        1. config/asset_launch_dates.json (~531 symbols from Binance API)
        2. 1000-prefix variant in cache (e.g. PEPEUSDT -> 1000PEPEUSDT)
        3. ASSET_LAUNCH_DATES legacy hardcoded dict (10 majors)
        4. `default`
    """
    return _ld_get_listing_date(symbol, default=default)


def _resolve_fapi_symbol(symbol: str) -> str:
    """Backwards-compat shim. Centralized in pipeline.listing_dates.

    Returns 1000-prefix form for low-priced tokens (SHIB/PEPE/BONK/etc.).
    """
    cache = _load_launch_dates_cache()
    if symbol in cache:
        return symbol
    base = symbol.replace("USDT", "")
    prefixed = f"1000{base}USDT"
    if prefixed in cache:
        return prefixed
    return symbol


# Retry delay (seconds) for the confirmation retry attempt
CONFIRM_RETRY_DELAY = 3

API_DISABLED = False
_RECHECK_STALE_DAYS = None  # Set by process_asset from --recheck-stale

# Threshold: if more than this many dates are missing, use bulk API fetch
BULK_API_THRESHOLD = 50

# Default parallelism for per-day downloads. data.binance.vision is on S3;
# 24 workers comfortably without rate limiting. Configurable via --workers.
DEFAULT_FETCH_WORKERS = 24

# Thread-local storage for per-thread requests.Session (cheap connection reuse).
_thread_local = threading.local()


def _get_thread_session():
    """One Session per thread (kept alive across tasks for connection reuse)."""
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = get_retry_session()
        _thread_local.session = s
    return s

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_retry_session(retries=3, backoff_factor=0.5):
    session = requests.Session()
    if PROXIES: session.proxies.update(PROXIES)

    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 503, 504)
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update(HEADERS)
    return session

def get_existing_files(directory):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return set()

    existing = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.endswith('.parquet') and entry.stat().st_size > 100:
                existing.add(entry.name)
    return existing

def handle_rate_limit(response, task_url):
    if response.status_code in [418, 429]:
        wait_time = int(response.headers.get("Retry-After", 60))
        tqdm.write(f"[RATE LIMIT] HIT ({response.status_code}) on {task_url}")
        time.sleep(wait_time)
        return True
    return False


# --- MANIFEST: Track confirmed-missing dates ---
# Each asset gets a _fetch_manifest.json that records dates where Binance
# genuinely has no data. On restart, these dates are skipped instead of
# re-attempted. Use --recheck-missing to force a re-check.

MANIFEST_FILENAME = "_fetch_manifest.json"

def load_manifest(asset_dir):
    """Load the fetch manifest for an asset, or create an empty one."""
    path = asset_dir / MANIFEST_FILENAME
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            # Ensure structure
            if "confirmed_missing" not in manifest:
                manifest["confirmed_missing"] = {}
            for dtype in ("aggTrades", "funding", "metrics", "klines_1m", "bookTicker"):
                if dtype not in manifest["confirmed_missing"]:
                    manifest["confirmed_missing"][dtype] = {}
            return manifest
        except (json.JSONDecodeError, KeyError):
            pass
    return {
        "version": 1,
        "confirmed_missing": {
            "aggTrades": {},
            "funding": {},
            "metrics": {},
            "klines_1m": {},
            "bookTicker": {},
        }
    }

def save_manifest(asset_dir, manifest):
    """Persist manifest to disk (atomic: tmp + os.replace).

    Was a direct open()+json.dump: a kill mid-write truncated the manifest, and
    the next load (which swallows JSONDecodeError) silently reset ALL
    confirmed-missing entries -> a full re-fetch of everything.
    """
    asset_dir.mkdir(parents=True, exist_ok=True)
    path = asset_dir / MANIFEST_FILENAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
    os.replace(str(tmp), str(path))

def is_confirmed_missing(manifest, data_type, date_str, recheck_stale_days=None):
    """Check if a date is confirmed missing for a given data type.

    If recheck_stale_days is set, entries confirmed more than N days ago
    are treated as NOT confirmed (will be re-fetched to check for backfills).
    """
    entry = manifest["confirmed_missing"].get(data_type, {}).get(date_str)
    if entry is None:
        return False
    if recheck_stale_days is not None:
        confirmed_at = entry.get("confirmed_at", "2020-01-01")
        try:
            confirmed_date = datetime.strptime(confirmed_at, "%Y-%m-%d")
            age_days = (datetime.now() - confirmed_date).days
            if age_days >= recheck_stale_days:
                return False  # Stale -- recheck it
        except ValueError:
            return False  # Can't parse date -- recheck
    return True

def mark_missing(manifest, data_type, date_str, reason="unknown"):
    """Record a date as confirmed missing from Binance.

    Defensive: initializes the sub-dict if absent so new data_type values
    (e.g. klines_1m, bookTicker added 2026-05-14) don't KeyError on legacy
    manifests that pre-date them.
    """
    cm = manifest.setdefault("confirmed_missing", {}).setdefault(data_type, {})
    cm[date_str] = {
        "confirmed_at": datetime.now().strftime("%Y-%m-%d"),
        "reason": reason,
    }

def count_missing(manifest, data_type):
    """Count confirmed missing dates for a data type."""
    return len(manifest["confirmed_missing"].get(data_type, {}))

def clear_missing(manifest, data_type=None):
    """Clear confirmed missing entries (for --recheck-missing)."""
    manifest.setdefault("confirmed_missing", {})
    if data_type:
        manifest["confirmed_missing"][data_type] = {}
    else:
        for dtype in ("aggTrades", "funding", "metrics", "klines_1m", "bookTicker"):
            manifest["confirmed_missing"][dtype] = {}


def standardize_metrics(df):
    df.columns = [c.strip() for c in df.columns]

    time_col = None
    for t in ["timestamp", "createTime", "create_time", "time"]:
        if t in df.columns: time_col = t; break

    if not time_col:
        return None, f"Missing Timestamp. Found: {df.columns}"

    if df.schema[time_col] == pl.String:
        df = df.with_columns(
            pl.col(time_col).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            .dt.timestamp("ms").cast(pl.Int64).alias("timestamp")
        )
    else:
        df = df.with_columns(pl.col(time_col).cast(pl.Int64).alias("timestamp"))

    oi_col = None
    candidates = ["sumOpenInterestValue", "sum_open_interest_value", "openInterestValue", "sumOpenInterest", "openInterest"]

    for c in candidates:
        if c in df.columns: oi_col = c; break

    if not oi_col: return None, f"Missing OI Column."

    ls_col = None
    for c in ["sumLongShortRatio", "longShortRatio", "long_short_ratio"]:
        if c in df.columns: ls_col = c; break

    cols = [pl.col("timestamp"), pl.col(oi_col).cast(pl.Float64).alias("open_interest_val")]

    if ls_col:
        cols.append(pl.col(ls_col).cast(pl.Float64).alias("long_short_ratio"))
    else:
        cols.append(pl.lit(1.0).alias("long_short_ratio"))

    return df.select(cols), None

# --- API FALLBACK FUNCTIONS ---

def fetch_api_funding(symbol, target_date, session):
    global API_DISABLED
    if API_DISABLED: return None
    base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    start_ts = int(target_date.replace(hour=0, minute=0, second=0).timestamp() * 1000)
    end_ts = int(target_date.replace(hour=23, minute=59, second=59).timestamp() * 1000)
    # R32++ pipeline-CRIT fix: resolve to 1000-prefix for low-priced assets
    fapi_symbol = _resolve_fapi_symbol(symbol)
    try:
        r = session.get(base_url, params={"symbol": fapi_symbol, "startTime": start_ts, "endTime": end_ts, "limit": 1000}, timeout=10)
        if r.status_code == 403: API_DISABLED = True; return None
        if handle_rate_limit(r, "API_FUNDING"): return None
        if r.status_code != 200: return None
        data = r.json()
        if not data: return None
        df = pl.DataFrame(data).rename({"fundingTime": "timestamp", "fundingRate": "funding_rate"})
        return df.select([pl.col("timestamp").cast(pl.Int64), pl.col("funding_rate").cast(pl.Float64)])
    except (requests.RequestException, KeyError, ValueError, TypeError): return None

def fetch_api_metrics(symbol, target_date, session):
    global API_DISABLED
    if API_DISABLED: return None
    base_url = "https://fapi.binance.com/futures/data/openInterestHist"
    start_ts = int(target_date.replace(hour=0, minute=0, second=0).timestamp() * 1000)
    end_ts = int(target_date.replace(hour=23, minute=59, second=59).timestamp() * 1000)
    # R32++ pipeline-CRIT fix: resolve to 1000-prefix for low-priced assets
    fapi_symbol = _resolve_fapi_symbol(symbol)
    try:
        r = session.get(base_url, params={"symbol": fapi_symbol, "period": "5m", "startTime": start_ts, "endTime": end_ts, "limit": 288}, timeout=10)
        if r.status_code == 403: API_DISABLED = True; return None
        if handle_rate_limit(r, "API_METRICS"): return None
        if r.status_code != 200: return None
        df_clean, err = standardize_metrics(pl.DataFrame(r.json()))
        return df_clean
    except (requests.RequestException, KeyError, ValueError, TypeError): return None


def bulk_fetch_funding_via_api(symbol, missing_dates, output_dir, manifest):
    """Fetch all missing funding data via paginated API calls.

    The Binance /fapi/v1/fundingRate endpoint returns up to 1000 records per call.
    Funding occurs 3x/day (every 8 hours), so 1000 records covers ~333 days.
    This is vastly faster than per-day API calls for assets where ZIPs are 404.
    """
    global API_DISABLED
    if API_DISABLED:
        return 0

    if not missing_dates:
        return 0

    # Sort dates ascending
    missing_dates_sorted = sorted(missing_dates)
    start_dt = missing_dates_sorted[0]
    end_dt = missing_dates_sorted[-1]
    start_ts = int(start_dt.replace(hour=0, minute=0, second=0).timestamp() * 1000)
    end_ts = int(end_dt.replace(hour=23, minute=59, second=59).timestamp() * 1000)

    # Build set of date strings we need
    needed_dates = {d.strftime("%Y-%m-%d") for d in missing_dates_sorted}

    # R32++ pipeline-CRIT fix: resolve to 1000-prefix for low-priced assets.
    # Without this, the bulk API returned [] for SHIB/PEPE/BONK/FLOKI etc.,
    # forcing fallback to per-date sync of 1831 dates per asset -> 4h timeout.
    fapi_symbol = _resolve_fapi_symbol(symbol)
    if fapi_symbol != symbol:
        _fetch_pl("DL", f"BULK_API: {symbol} mapped to {fapi_symbol} (1000-prefix asset)")

    print(f"   [BULK API] Fetching {fapi_symbol} funding via API "
          f"({len(needed_dates)} dates, {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')})...")

    base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    session = get_retry_session()
    all_records = []
    cursor_ts = start_ts
    page = 0
    page_retries = 0
    max_retries_per_page = 5

    while cursor_ts < end_ts:
        page += 1
        try:
            r = session.get(base_url, params={
                "symbol": fapi_symbol, "startTime": cursor_ts,
                "endTime": end_ts, "limit": 1000
            }, timeout=15)

            if r.status_code == 403:
                API_DISABLED = True
                _fetch_pl("WARN", f"API returned 403, disabling API fallback")
                break
            if handle_rate_limit(r, "BULK_FUNDING"):
                page_retries += 1
                if page_retries > max_retries_per_page:
                    _fetch_pl("WARN", f"Bulk API: exceeded {max_retries_per_page} retries on page {page}")
                    break
                time.sleep(5)
                continue
            page_retries = 0  # reset on success
            if r.status_code != 200:
                _fetch_pl("WARN", f"Bulk API page {page}: HTTP {r.status_code}")
                break

            data = r.json()
            if not data:
                break

            all_records.extend(data)

            # Move cursor past last record
            last_ts = max(rec.get("fundingTime", 0) for rec in data)
            if last_ts <= cursor_ts:
                break
            cursor_ts = last_ts + 1

            # Rate limit protection
            time.sleep(0.3)

        except Exception as e:
            _fetch_pl("WARN", f"Bulk API page {page} error: {type(e).__name__}")
            break

    if not all_records:
        _fetch_pl("WARN", f"Bulk API returned 0 records for {symbol}")
        return 0

    _fetch_pl("DL", f"BULK_API: Got {len(all_records)} funding records in {page} pages")

    # Convert to DataFrame and split by date
    df_all = pl.DataFrame(all_records)
    if "fundingTime" not in df_all.columns or "fundingRate" not in df_all.columns:
        _fetch_pl("WARN", f"Unexpected API columns: {df_all.columns}")
        return 0

    df_all = df_all.with_columns([
        pl.col("fundingTime").cast(pl.Int64).alias("timestamp"),
        pl.col("fundingRate").cast(pl.Float64).alias("funding_rate"),
    ]).select(["timestamp", "funding_rate"])

    # Split by date and save daily files
    fetched_count = 0
    df_all = df_all.with_columns(
        ((pl.col("timestamp") // 86400000) * 86400000).alias("day_ts")
    )

    for day_ts in df_all["day_ts"].unique().sort().to_list():
        day_dt = datetime(1970, 1, 1) + timedelta(milliseconds=int(day_ts))
        d_str = day_dt.strftime("%Y-%m-%d")

        if d_str not in needed_dates:
            continue

        filename = f"{symbol}-funding-{d_str}.parquet"
        filepath = output_dir / filename

        if filepath.exists() and filepath.stat().st_size > 100:
            continue

        day_df = df_all.filter(pl.col("day_ts") == day_ts).select(["timestamp", "funding_rate"])
        if len(day_df) > 0:
            # Phase B7 retrofit: atomic-tmp-rename via parquet_io helper
            # (module-level import; was lazy-import-with-fallback pre-2026-05-24).
            _atomic_write_parquet(day_df, filepath,
                                   required_cols={"timestamp", "funding_rate"})
            fetched_count += 1

    _fetch_pl("DL", f"BULK_API: Saved {fetched_count} daily funding files for {symbol}")
    return fetched_count


# ─── PARALLEL DOWNLOAD HELPERS (post-2026-04-27 throughput refactor) ────────

# Per-data-type ZIP -> DataFrame parsers. Pure functions; no I/O.

_AGGTRADE_COLS = [
    "agg_trade_id", "price", "qty", "first_trade_id",
    "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match",
]


def _parse_aggtrades_zip(content: bytes) -> "pl.DataFrame|None":
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pl.read_csv(f.read(), has_header=False, new_columns=_AGGTRADE_COLS)
                return df.select([
                    pl.col("timestamp").cast(pl.Int64),
                    pl.col("price").cast(pl.Float64),
                    pl.col("qty").cast(pl.Float64),
                    pl.col("is_buyer_maker").cast(pl.Boolean),
                ]).unique(subset=["timestamp", "price", "qty"])
    except Exception as e:
        tqdm.write(f"[parse_aggtrades_err] {type(e).__name__}: {str(e)[:120]}")
        return None


def _parse_funding_zip(content: bytes) -> "pl.DataFrame|None":
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pl.read_csv(f.read())
                if "calcTime" in df.columns:
                    df = df.rename({"calcTime": "timestamp", "fundingRate": "funding_rate"})
                return df.select([
                    pl.col("timestamp").cast(pl.Int64),
                    pl.col("funding_rate").cast(pl.Float64),
                ])
    except Exception as e:
        tqdm.write(f"[parse_funding_err] {type(e).__name__}: {str(e)[:120]}")
        return None


def _parse_metrics_zip(content: bytes) -> "pl.DataFrame|None":
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            with z.open(z.namelist()[0]) as f:
                df_clean, _err = standardize_metrics(pl.read_csv(f.read()))
                return df_clean
    except Exception as e:
        tqdm.write(f"[parse_metrics_err] {type(e).__name__}: {str(e)[:120]}")
        return None


# Klines schema (Binance daily 1m klines):
# open_time, open, high, low, close, volume, close_time, quote_volume,
# n_trades, taker_buy_volume, taker_buy_quote_volume, ignore
_KLINES_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def _parse_klines_zip(content: bytes) -> "pl.DataFrame|None":
    """Parse 1-min kline ZIP. Returns timestamp + OHLCV + taker buy split.

    Keep enough fields to derive dollar bars later (volume + close → notional;
    taker_buy_volume/volume → flow imbalance proxy at bar level).
    """
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pl.read_csv(f.read(), has_header=False, new_columns=_KLINES_COLS)
                return df.select([
                    pl.col("open_time").cast(pl.Int64).alias("timestamp"),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                    pl.col("quote_volume").cast(pl.Float64),
                    pl.col("n_trades").cast(pl.Int64),
                    pl.col("taker_buy_volume").cast(pl.Float64),
                    pl.col("taker_buy_quote_volume").cast(pl.Float64),
                ])
    except Exception as e:
        tqdm.write(f"[parse_klines_err] {type(e).__name__}: {str(e)[:120]}")
        return None


def _download_one_zip(task: dict, parser_fn) -> tuple[str, str]:
    """Worker: download one ZIP, parse, save parquet. Returns (status, reason).

    status in {'ok', 'skip_existing', 'rate_limited', 'fail'}.
    `fail` reason is suitable for mark_missing(). Used by ThreadPoolExecutor —
    each thread reuses a thread-local Session for connection pooling.
    """
    path = task["path"]
    if path.exists() and path.stat().st_size > 100:
        return ("skip_existing", "")

    session = _get_thread_session()
    reasons = []

    # --- Attempt 1: download
    try:
        r = session.get(task["url"], timeout=15)
        if r.status_code == 200:
            df = parser_fn(r.content)
            if df is not None and len(df) > 0:
                _atomic_write_parquet(df, path)
                return ("ok", "")
            reasons.append("dl1_parse_empty")
        elif r.status_code in (418, 429):
            return ("rate_limited", f"http_{r.status_code}")
        else:
            reasons.append(f"dl1_{r.status_code}")
    except Exception as e:
        reasons.append(f"dl1_err_{type(e).__name__}")

    # --- Attempt 2: confirmation retry with fresh session, slight delay
    time.sleep(CONFIRM_RETRY_DELAY)
    try:
        retry_session = get_retry_session()
        r2 = retry_session.get(task["url"], timeout=15)
        if r2.status_code == 200:
            df = parser_fn(r2.content)
            if df is not None and len(df) > 0:
                _atomic_write_parquet(df, path)
                return ("ok", "")
            reasons.append("dl2_parse_empty")
        else:
            reasons.append(f"dl2_{r2.status_code}")
    except Exception as e:
        reasons.append(f"dl2_err_{type(e).__name__}")

    return ("fail", "+".join(reasons) if reasons else "unknown")


def _run_parallel_downloads(
    tasks: list[dict],
    parser_fn,
    dtype_name: str,
    manifest: dict,
    label: str,
    max_workers: int = DEFAULT_FETCH_WORKERS,
) -> tuple[int, int, list[dict]]:
    """Run the per-day download tasks in parallel.

    Returns: (n_ok, n_new_missing, failed_tasks)
        failed_tasks: list of tasks that errored (status='fail') — caller may
        do a sequential API fallback pass on these.
    """
    if not tasks:
        return (0, 0, [])

    manifest_lock = threading.Lock()
    n_ok = n_new_missing = 0
    failed: list[dict] = []
    rate_limited: list[dict] = []

    with tqdm(total=len(tasks), unit="file", leave=False, desc=f"{label} ||={max_workers}") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_download_one_zip, t, parser_fn): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    status, reason = fut.result()
                except Exception as e:
                    status, reason = "fail", f"future_{type(e).__name__}_{e}"

                if status == "ok" or status == "skip_existing":
                    n_ok += 1
                elif status == "rate_limited":
                    rate_limited.append(t)
                else:  # fail
                    failed.append({**t, "reason": reason})

                pbar.update(1)

    if rate_limited:
        # Brief backoff + retry rate-limited tasks sequentially (rare path).
        time.sleep(30)
        for t in rate_limited:
            try:
                status, reason = _download_one_zip(t, parser_fn)
            except Exception as e:
                status, reason = "fail", f"retry_{type(e).__name__}"
            if status == "ok" or status == "skip_existing":
                n_ok += 1
            else:
                failed.append({**t, "reason": reason or "rate_limited"})

    # Mark confirmed-missing for tasks that failed (with manifest lock for safety)
    with manifest_lock:
        for t in failed:
            mark_missing(manifest, dtype_name, t["d_str"], t.get("reason", "unknown"))
        n_new_missing = len(failed)

    return n_ok, n_new_missing, failed


# --- WORKERS ---

def sync_trades(symbol, date_range, output_dir, manifest, max_workers=DEFAULT_FETCH_WORKERS):
    """Sync aggTrades in parallel. No API fallback — download-only."""
    launch_date = _resolve_launch_date(symbol)
    valid_dates = [d for d in date_range if d >= launch_date]
    if not valid_dates:
        return 0

    existing = get_existing_files(output_dir)
    tasks = []
    skipped_missing = 0

    print(f"   [SCAN] Scanning Trades ({len(existing)} exists)...", end="\r")
    for date in valid_dates:
        d_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-aggTrades-{d_str}.parquet"
        if filename in existing:
            continue
        if is_confirmed_missing(manifest, "aggTrades", d_str, _RECHECK_STALE_DAYS):
            skipped_missing += 1
            continue
        url = f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{d_str}.zip"
        tasks.append({"date": date, "url": url, "path": output_dir / filename, "d_str": d_str})

    if not tasks:
        skip_note = f" ({skipped_missing} confirmed missing)" if skipped_missing else ""
        _fetch_pl("OK", f"Trades: Up to date.{skip_note}")
        return 0
    if skipped_missing:
        _fetch_pl("SKIP", f"Trades: {skipped_missing} dates confirmed missing from Binance")
    _fetch_pl("DL", f"Trades: Syncing {len(tasks)} files (parallel={max_workers})...")

    n_ok, n_new_missing, _failed = _run_parallel_downloads(
        tasks, _parse_aggtrades_zip, "aggTrades", manifest,
        label=f"TRADES {symbol}", max_workers=max_workers,
    )
    return n_new_missing


def sync_klines(symbol, date_range, output_dir, manifest, max_workers=DEFAULT_FETCH_WORKERS,
                interval: str = "1m"):
    """Sync 1-minute klines (Binance spot, daily ZIPs).

    ~60x smaller than aggTrades. Suitable for tier-C tail assets where dollar-bar
    microstructure is overkill — cross-sectional ranker / breadth signals only need
    OHLCV + volume + taker-buy split, all of which klines provide.

    Output: data/raw/<SYM>/klines_<interval>/<SYM>-<interval>-<DATE>.parquet
    """
    launch_date = _resolve_launch_date(symbol)
    valid_dates = [d for d in date_range if d >= launch_date]
    if not valid_dates:
        return 0

    existing = get_existing_files(output_dir)
    tasks = []
    skipped_missing = 0
    dtype_name = f"klines_{interval}"

    print(f"   [SCAN] Scanning Klines/{interval} ({len(existing)} exists)...", end="\r")
    for date in valid_dates:
        d_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-{interval}-{d_str}.parquet"
        if filename in existing:
            continue
        if is_confirmed_missing(manifest, dtype_name, d_str, _RECHECK_STALE_DAYS):
            skipped_missing += 1
            continue
        url = (f"https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/"
               f"{symbol}-{interval}-{d_str}.zip")
        tasks.append({"date": date, "url": url, "path": output_dir / filename, "d_str": d_str})

    if not tasks:
        skip_note = f" ({skipped_missing} confirmed missing)" if skipped_missing else ""
        _fetch_pl("OK", f"Klines/{interval}: Up to date.{skip_note}")
        return 0
    if skipped_missing:
        _fetch_pl("SKIP", f"Klines/{interval}: {skipped_missing} dates confirmed missing")
    _fetch_pl("DL", f"Klines/{interval}: Syncing {len(tasks)} files (parallel={max_workers})...")

    n_ok, _, failed = _run_parallel_downloads(
        tasks, _parse_klines_zip, dtype_name, manifest,
        label=f"KLINES/{interval} {symbol}", max_workers=max_workers,
    )
    return len(failed)


def sync_funding(symbol, date_range, output_dir, manifest, max_workers=DEFAULT_FETCH_WORKERS):
    """Sync funding rates. Parallel download + bulk API fallback for failures."""
    launch_date = _resolve_launch_date(symbol)
    valid_dates = [d for d in date_range if d >= launch_date]
    if not valid_dates:
        return 0

    # R32+++ pipeline-crawler-CRIT fix: Binance Vision ZIP archive uses the
    # 1000-prefix path for low-priced assets (1000SHIBUSDT/1000SHIBUSDT-...).
    # Prior fix patched REST API but ZIPs still 404'd. Apply same resolution
    # to the ZIP URL path AND output filename (keeping data-dir filename
    # consistent with caller expectations: <symbol>-funding-<date>.parquet).
    fapi_symbol = _resolve_fapi_symbol(symbol)

    existing = get_existing_files(output_dir)
    tasks = []
    skipped_missing = 0

    print(f"   [SCAN] Scanning Funding ({len(existing)} exists)...", end="\r")
    for date in valid_dates:
        d_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-funding-{d_str}.parquet"
        if filename in existing:
            continue
        if is_confirmed_missing(manifest, "funding", d_str, _RECHECK_STALE_DAYS):
            skipped_missing += 1
            continue
        url = f"https://data.binance.vision/data/futures/um/daily/fundingRate/{fapi_symbol}/{fapi_symbol}-fundingRate-{d_str}.zip"
        tasks.append({"date": date, "url": url, "path": output_dir / filename, "d_str": d_str})

    if not tasks:
        skip_note = f" ({skipped_missing} confirmed missing)" if skipped_missing else ""
        _fetch_pl("OK", f"Funding: Up to date.{skip_note}")
        return 0
    if skipped_missing:
        _fetch_pl("SKIP", f"Funding: {skipped_missing} dates confirmed missing from Binance")

    # --- BULK API PRE-FETCH (when many dates missing, faster than per-day) ---
    if len(tasks) > BULK_API_THRESHOLD:
        missing_dates = [t["date"] for t in tasks]
        bulk_count = bulk_fetch_funding_via_api(symbol, missing_dates, output_dir, manifest)
        if bulk_count > 0:
            existing = get_existing_files(output_dir)
            tasks = [t for t in tasks if t["path"].name not in existing]
            if not tasks:
                _fetch_pl("OK", f"Funding: All filled by bulk API ({bulk_count} files)")
                return 0
            _fetch_pl("DL", f"Funding: {len(tasks)} remaining after bulk API")

    _fetch_pl("DL", f"Funding: Syncing {len(tasks)} files (parallel={max_workers})...")

    n_ok, _, failed = _run_parallel_downloads(
        tasks, _parse_funding_zip, "funding", manifest,
        label=f"FUNDING {symbol}", max_workers=max_workers,
    )

    # API fallback for failures (rare path; kept sequential to avoid hammering REST)
    if failed:
        # Undo the optimistic mark_missing for entries we'll retry via API.
        for t in failed:
            try:
                manifest["confirmed_missing"]["funding"].pop(t["d_str"], None)
            except Exception:
                pass
        recovered = 0
        api_session = get_retry_session()
        for t in failed:
            df_api = fetch_api_funding(symbol, t["date"], api_session)
            if df_api is not None and len(df_api) > 0:
                _atomic_write_parquet(df_api, t["path"])
                recovered += 1
            else:
                mark_missing(manifest, "funding",
                             t["d_str"], (t.get("reason", "") + "+api_empty").strip("+"))
        if recovered:
            print(f"   [API] Funding: recovered {recovered}/{len(failed)} via REST API")

    return len(failed)


def sync_metrics(symbol, date_range, output_dir, manifest, max_workers=DEFAULT_FETCH_WORKERS):
    """Sync OI metrics. Parallel download + REST API fallback for failures."""
    launch_date = _resolve_launch_date(symbol)
    effective_start = max(launch_date, METRICS_EARLIEST_DATE)
    valid_dates = [d for d in date_range if d >= effective_start]
    if not valid_dates:
        return 0

    # R32+++ pipeline-crawler-CRIT fix: same 1000-prefix ZIP URL bug as funding.
    # Binance Vision metrics path uses 1000SHIBUSDT for low-priced assets.
    fapi_symbol = _resolve_fapi_symbol(symbol)

    existing = get_existing_files(output_dir)
    tasks = []
    skipped_missing = 0

    print(f"   [SCAN] Scanning Metrics ({len(existing)} exists)...", end="\r")
    for date in valid_dates:
        d_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}-metrics-{d_str}.parquet"
        if filename in existing:
            continue
        if is_confirmed_missing(manifest, "metrics", d_str, _RECHECK_STALE_DAYS):
            skipped_missing += 1
            continue
        url = f"https://data.binance.vision/data/futures/um/daily/metrics/{fapi_symbol}/{fapi_symbol}-metrics-{d_str}.zip"
        tasks.append({"date": date, "url": url, "path": output_dir / filename, "d_str": d_str})

    if not tasks:
        skip_note = f" ({skipped_missing} confirmed missing)" if skipped_missing else ""
        _fetch_pl("OK", f"Metrics: Up to date.{skip_note}")
        return 0
    if skipped_missing:
        _fetch_pl("SKIP", f"Metrics: {skipped_missing} dates confirmed missing from Binance")
    _fetch_pl("DL", f"Metrics: Syncing {len(tasks)} files (parallel={max_workers})...")

    n_ok, _, failed = _run_parallel_downloads(
        tasks, _parse_metrics_zip, "metrics", manifest,
        label=f"METRICS {symbol}", max_workers=max_workers,
    )

    if failed:
        for t in failed:
            try:
                manifest["confirmed_missing"]["metrics"].pop(t["d_str"], None)
            except Exception:
                pass
        recovered = 0
        api_session = get_retry_session()
        for t in failed:
            df_api = fetch_api_metrics(symbol, t["date"], api_session)
            if df_api is not None and len(df_api) > 0:
                _atomic_write_parquet(df_api, t["path"])
                recovered += 1
            else:
                mark_missing(manifest, "metrics",
                             t["d_str"], (t.get("reason", "") + "+api_empty").strip("+"))
        if recovered:
            print(f"   [API] Metrics: recovered {recovered}/{len(failed)} via REST API")

    return len(failed)


def process_asset(symbol_pair, start_date, reverse=False, recheck_missing=False,
                   recheck_stale_days=None, max_workers=DEFAULT_FETCH_WORKERS,
                   trade_mode: str = "aggtrades"):
    """trade_mode in {'aggtrades', 'klines', 'both'}:
        aggtrades: tick-precision aggTrades only (default; needed for dollar bars + DIB)
        klines:    1-min OHLCV only (~60x smaller; for tier-C tail assets)
        both:      both (rare; useful for transitional analysis)
    """
    global API_DISABLED, _RECHECK_STALE_DAYS
    _RECHECK_STALE_DAYS = recheck_stale_days
    clean_sym = symbol_pair.replace("/", "").upper()
    direction_str = "REVERSE (Newest -> Oldest)" if reverse else "FORWARD (Oldest -> Newest)"
    print(f"\n[ASSET] PROCESSING: {clean_sym} [{direction_str}]")
    asset_dir = RAW_DIR / clean_sym

    # Reset API_DISABLED per asset (a 403 on one asset shouldn't block others)
    API_DISABLED = False

    # Load manifest (tracks confirmed-missing dates)
    manifest = load_manifest(asset_dir)
    if recheck_missing:
        n_before = sum(count_missing(manifest, dt) for dt in ("aggTrades", "funding", "metrics"))
        clear_missing(manifest)
        print(f"   [RECHECK] Cleared {n_before} confirmed-missing entries for re-verification")
    elif recheck_stale_days is not None:
        # Count how many stale entries will be rechecked
        n_stale = 0
        for dtype in ("aggTrades", "funding", "metrics"):
            for d_str, entry in manifest["confirmed_missing"].get(dtype, {}).items():
                confirmed_at = entry.get("confirmed_at", "2020-01-01")
                try:
                    age = (datetime.now() - datetime.strptime(confirmed_at, "%Y-%m-%d")).days
                    if age >= recheck_stale_days:
                        n_stale += 1
                except ValueError:
                    n_stale += 1
        if n_stale > 0:
            print(f"   [RECHECK-STALE] {n_stale} entries older than {recheck_stale_days} days will be re-verified")

    # Check if we need to purge invalid trades
    dir_trades = asset_dir / "aggTrades"
    if dir_trades.exists():
        files = list(dir_trades.glob("*.parquet"))
        if files:
            try:
                # If timestamp is small integer, it's a trade ID -> PURGE
                sample = pl.read_parquet(files[0])["timestamp"][0]
                if sample < 10_000_000_000:
                    _fetch_pl("WARN", f"DETECTED CORRUPTED DATA (Trade IDs in Timestamp). Purging {len(files)} files...")
                    for f in files: f.unlink()
            except (pl.ComputeError, OSError, IndexError) as _e:
                pass  # non-critical: corrupted file detection is best-effort

    try: start = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    except ValueError: start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.now() - timedelta(days=1)

    full_date_range = [start + timedelta(days=x) for x in range((end - start).days + 1)]

    # [FEATURE] Reverse Mechanism
    if reverse:
        full_date_range.reverse()

    new_m_trades = 0
    new_m_klines = 0
    if trade_mode in ("aggtrades", "both"):
        new_m_trades = sync_trades(clean_sym, full_date_range, asset_dir / "aggTrades",
                                    manifest, max_workers=max_workers)
    if trade_mode in ("klines", "both"):
        new_m_klines = sync_klines(clean_sym, full_date_range, asset_dir / "klines_1m",
                                    manifest, max_workers=max_workers)
    new_m_funding = sync_funding(clean_sym, full_date_range, asset_dir / "funding",
                                 manifest, max_workers=max_workers)
    new_m_metrics = sync_metrics(clean_sym, full_date_range, asset_dir / "metrics",
                                 manifest, max_workers=max_workers)

    # Persist manifest
    save_manifest(asset_dir, manifest)

    # Summary report
    total_missing = {
        "aggTrades": count_missing(manifest, "aggTrades"),
        "funding": count_missing(manifest, "funding"),
        "metrics": count_missing(manifest, "metrics"),
    }
    total_new = new_m_trades + new_m_funding + new_m_metrics
    total_known = sum(total_missing.values())

    if total_new > 0:
        print(f"   [REPORT] {clean_sym}: {total_new} newly confirmed missing "
              f"(total: {total_known} missing across all types)")
        for dtype, n in total_missing.items():
            if n > 0:
                print(f"      {dtype}: {n} dates confirmed missing")
    elif total_known > 0:
        print(f"   [REPORT] {clean_sym}: {total_known} dates confirmed missing (no new)")

    # R32++ pipeline-audit HIGH fix: cross-file count parity check.
    # SHIB-class bug: 1831 aggTrades + 0 funding silently passed before; only
    # surfaced at chimera's norm_funding dead-feature gate. Catches the
    # _resolve_fapi_symbol bug class at fetch time.
    # R32+++ pipeline-audit HIGH fix: parity verdict now WRITES TO A SENTINEL
    # FILE that the caller (or downstream jobs) can read for rc=1 promotion.
    # Previously the parity check printed but had `except: pass` and didn't
    # influence the asset run's outcome - silent failure.
    parity_failed = False
    parity_reasons: list[str] = []
    try:
        from pathlib import Path as _Path
        agg_dir = _Path(asset_dir) / "aggTrades"
        fund_dir = _Path(asset_dir) / "funding"
        metr_dir = _Path(asset_dir) / "metrics"
        n_agg = len(list(agg_dir.glob("*.parquet"))) if agg_dir.exists() else 0
        n_fund = len(list(fund_dir.glob("*.parquet"))) if fund_dir.exists() else 0
        n_metr = len(list(metr_dir.glob("*.parquet"))) if metr_dir.exists() else 0
        if n_agg > 100 and n_fund == 0:
            print(f"   [FAIL-PARITY] {clean_sym}: {n_agg} aggTrades but 0 funding files "
                  f"-- likely 1000-prefix API bug; check _resolve_fapi_symbol", flush=True)
            parity_failed = True
            parity_reasons.append(f"funding=0 (agg={n_agg})")
        elif n_agg > 100 and n_fund < int(0.5 * n_agg):
            print(f"   [WARN-PARITY] {clean_sym}: funding/aggTrades ratio "
                  f"{n_fund}/{n_agg} = {100*n_fund/n_agg:.0f}% (< 50%); investigate",
                  flush=True)
            parity_failed = True
            parity_reasons.append(f"funding/agg ratio {n_fund}/{n_agg}")
        if n_agg > 100 and n_metr == 0:
            print(f"   [FAIL-PARITY] {clean_sym}: {n_agg} aggTrades but 0 metrics files "
                  f"-- likely 1000-prefix API bug for openInterestHist", flush=True)
            parity_failed = True
            parity_reasons.append(f"metrics=0 (agg={n_agg})")
        # Write sentinel so downstream jobs (refresh.py / pre_train_gate) can
        # surface this as rc=1 instead of treating fetch as fully green.
        if parity_failed:
            sentinel = _Path(asset_dir) / "_parity_failed.json"
            import json as _json
            sentinel.write_text(_json.dumps({
                "asset": clean_sym,
                "n_aggTrades": n_agg, "n_funding": n_fund, "n_metrics": n_metr,
                "reasons": parity_reasons,
                # was dt.datetime.utcnow() guarded by `"dt" in dir()` -- `dt` was
                # never defined, so checked_at was ALWAYS None. Use the imported
                # `datetime` directly.
                "checked_at": datetime.utcnow().isoformat() + "Z",
            }, indent=2), encoding="utf-8")
        else:
            # Clear stale sentinel if parity now passes
            sentinel = _Path(asset_dir) / "_parity_failed.json"
            if sentinel.exists():
                try:
                    sentinel.unlink()
                except OSError:
                    pass
    except Exception as e:
        print(f"   [WARN] {clean_sym}: parity check itself crashed: {e}", flush=True)

    if parity_failed:
        print(f"   [DONE-WITH-PARITY-FAIL] Asset Complete: {clean_sym} -- "
              f"see _parity_failed.json sentinel", flush=True)
    else:
        print(f"   [DONE] Asset Complete: {clean_sym}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binance Data Fetcher (Parallel capable)")
    parser.add_argument("-r", "--reverse", action="store_true",
                        help="Reverse BOTH date order (Newest->Oldest) AND asset list order. "
                             "Run two instances (normal + -r) to fetch from both directions.")
    parser.add_argument("--force", action="store_true",
                        help="Alias for --recheck-missing: re-attempt all confirmed-missing "
                             "dates. Useful for forcing a fresh fetch after Binance backfills "
                             "or to invalidate cached miss markers.")
    parser.add_argument("--recheck-missing", action="store_true",
                        help="Re-attempt all previously confirmed-missing dates (for Binance backfills)")
    parser.add_argument("--recheck-stale", type=int, default=None, metavar="DAYS",
                        help="Re-attempt confirmed-missing dates older than N days "
                             "(e.g., --recheck-stale 7 rechecks entries >7 days old)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Override start date (e.g., '2026-03-01'). Default: from config.")
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Specific asset pairs to fetch (e.g., SUI/USDT PEPE/USDT). "
                             "Assets not in config use default $100K bar size.")
    parser.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                        help="Resolve asset list from config/universes/<u>.yaml "
                             "(overrides --assets when both given is an error). u10/u50/u100.")
    parser.add_argument("--from-screener", action="store_true",
                        help="Fetch assets from universe_screener.py output "
                             "(data/prod_state/universe.json)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Run universe screener inline and fetch top N assets "
                             "(e.g., --top-n 30)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove raw data folders for assets not in current "
                             "universe and not in config (frees disk space)")
    parser.add_argument("--workers", type=int, default=DEFAULT_FETCH_WORKERS,
                        help=f"Per-asset parallel download workers (default: {DEFAULT_FETCH_WORKERS}; "
                             "set 1 for serial, 32-48 for fast networks). data.binance.vision "
                             "is on S3 and tolerates high concurrency.")
    parser.add_argument("--asset-workers", type=int, default=1,
                        help="Outer-loop assets in parallel (default: 1 = sequential). "
                             "Use with care; combined with --workers, total threads = "
                             "asset_workers * workers.")
    parser.add_argument("--trade-mode", choices=["aggtrades", "klines", "both"],
                        default="aggtrades",
                        help="Which trade source to fetch. 'aggtrades' (default) = "
                             "tick-precision, ~60x larger, needed for dollar bars + DIB. "
                             "'klines' = 1-min OHLCV, sufficient for cross-sectional "
                             "ranker / breadth signals on tier-C tail assets. "
                             "'both' downloads both (transitional / audit only).")
    args = parser.parse_args()

    try: conf = load_config()
    except Exception as e: print(f"[ERROR] CONFIG ERROR: {e}"); sys.exit(1)

    start_dt = args.start_date or conf['data']['start_date']

    # Determine asset list from various sources
    asset_pairs = None

    if args.top_n:
        # Run screener inline and use top N
        print(f"[SCREENER] Fetching top {args.top_n} assets by volume/volatility...")
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from prod.universe_screener import run_screener
            results = run_screener(top_n=args.top_n, min_volume=5.0)
            asset_pairs = [r["symbol"].upper().replace("USDT", "/USDT")
                          if "/" not in r["symbol"] else r["symbol"]
                          for r in results]
            print(f"[SCREENER] Selected {len(asset_pairs)} assets")
        except Exception as e:
            print(f"[ERROR] Screener failed: {e}")
            print("[FALLBACK] Using config assets")

    elif args.from_screener:
        # Load from universe.json
        universe_path = Path(__file__).resolve().parent.parent.parent / "data" / "prod_state" / "universe.json"
        if universe_path.exists():
            import json as _json
            with open(universe_path) as f:
                universe = _json.load(f)
            asset_pairs = [a["symbol"] for a in universe.get("assets", [])]
            print(f"[SCREENER] Loaded {len(asset_pairs)} assets from {universe_path.name}")
        else:
            print(f"[ERROR] No universe.json found at {universe_path}")
            print("[INFO] Run first: python src/prod/universe_screener.py --top 30")
            sys.exit(1)

    elif args.universe:
        # Resolve from config/universes/<universe>.yaml.
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from universe_loader import UniverseLoader as _UL
            syms = _UL.load().list(args.universe)
            asset_pairs = [s.replace("USDT", "/USDT") if not s.endswith("/USDT") else s
                           for s in syms]
            print(f"[UNIVERSE] {args.universe}: {len(asset_pairs)} assets resolved")
        except Exception as e:
            print(f"[ERROR] universe={args.universe} load failed: {e}")
            sys.exit(1)

    elif args.assets:
        asset_pairs = args.assets

    def _normalize_pair(pair):
        """Normalize asset name to BASE/QUOTE form (e.g., 'BTC' or 'BTCUSDT' -> 'BTC/USDT').

        2026-05-17 fix: previously bare 'BTC' fell through and was treated as
        asset name 'BTC' downstream -- process_asset built asset_dir = RAW_DIR/'BTC'
        which doesn't exist; sync_trades reported (0 exists) and tried to download
        all 2327 days of BTC into data/raw/BTC/ instead of data/raw/BTCUSDT/.
        Now: bare 'BTC' -> 'BTC/USDT'; 'BTCUSDT' -> 'BTC/USDT'; already-formed
        'BTC/USDT' passes through. Default quote = USDT (consistent with universe yamls).
        """
        if "/" in pair:
            return pair.upper()
        pair = pair.upper()
        # Already a known pair-suffix?
        for quote in ("USDT", "USDC", "BUSD", "BTC", "ETH"):
            if pair.endswith(quote) and len(pair) > len(quote):
                return pair[:-len(quote)] + "/" + quote
        # Bare symbol -> default to /USDT
        return pair + "/USDT"

    def _run_one(pair):
        pair = _normalize_pair(pair)
        print(f"\n[FETCH] {pair} (from {start_dt}, workers={args.workers}, mode={args.trade_mode})")
        process_asset(pair, start_dt, reverse=args.reverse,
                      recheck_missing=args.recheck_missing or args.force,
                      recheck_stale_days=args.recheck_stale,
                      max_workers=args.workers,
                      trade_mode=args.trade_mode)

    def _run_assets_parallel(pairs):
        if args.asset_workers <= 1:
            for p in pairs:
                _run_one(p)
        else:
            print(f"\n[ASSET-PARALLEL] {len(pairs)} assets x {args.asset_workers} concurrent "
                  f"(per-asset workers={args.workers})")
            with ThreadPoolExecutor(max_workers=args.asset_workers) as ex:
                list(ex.map(_run_one, pairs))

    # 2026-05-15 R33: defensive filter against u100 excluded_assets.
    # Without this, the legacy config['assets'] fallback (config/data_config.yaml)
    # processed 24 stale entries (QIUSDT, RSRUSDT, etc.) marked excluded in
    # u100.yaml. Drift caught: legacy config not auto-synced with universe yaml.
    def _u100_excluded_set() -> set[str]:
        try:
            import yaml as _yaml
            up = Path(__file__).resolve().parents[2] / "config" / "universes" / "u100.yaml"
            if not up.exists():
                return set()
            with up.open() as f:
                u100 = _yaml.safe_load(f)
            return set(u100.get("excluded_assets") or [])
        except Exception:
            return set()

    def _filter_excluded(pairs):
        if not pairs:
            return pairs
        excluded = _u100_excluded_set()
        if not excluded:
            return pairs
        kept = []
        dropped = []
        for p in pairs:
            normalized = p.replace("/", "").upper()
            if normalized in excluded:
                dropped.append(p)
            else:
                kept.append(p)
        if dropped:
            print(f"[EXCLUDE] filtered {len(dropped)} excluded_assets per u100.yaml: "
                  f"{dropped[:10]}{' ...' if len(dropped) > 10 else ''}")
        return kept

    if asset_pairs:
        # Fetch specific assets (may include ones not in config). Filter excluded.
        if args.reverse:
            asset_pairs = list(reversed(asset_pairs))
        asset_pairs = _filter_excluded(asset_pairs)
        _run_assets_parallel(asset_pairs)
    elif 'assets' in conf:
        asset_items = list(conf['assets'].items())
        if args.reverse:
            asset_items = list(reversed(asset_items))
            print(f"[REVERSE] Asset order reversed ({len(asset_items)} assets)")
        active_pairs = [pair for pair, settings in asset_items if settings.get('is_active', True)]
        active_pairs = _filter_excluded(active_pairs)
        _run_assets_parallel(active_pairs)

    # Cleanup: remove raw data for assets not in universe or config
    if args.cleanup:
        import shutil
        raw_dir = Path(conf['data']['raw_dir'])
        if not raw_dir.is_absolute():
            raw_dir = Path(__file__).resolve().parent.parent.parent / raw_dir

        # Build keep set: config assets + universe assets + explicitly fetched
        keep = set()
        if 'assets' in conf:
            for pair in conf['assets']:
                keep.add(pair.replace("/", "").upper())
        if asset_pairs:
            for pair in asset_pairs:
                keep.add(pair.replace("/", "").upper())
        # Also load universe.json if exists
        universe_path = Path(__file__).resolve().parent.parent.parent / "data" / "prod_state" / "universe.json"
        if universe_path.exists():
            import json as _json
            with open(universe_path) as _f:
                _u = _json.load(_f)
            for a in _u.get("assets", []):
                keep.add(a["asset"].upper())

        if raw_dir.exists() and keep:
            removed = []
            for d in sorted(raw_dir.iterdir()):
                if d.is_dir() and d.name.upper() not in keep:
                    size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
                    print(f"  [CLEANUP] Removing {d.name} ({size_mb:.0f} MB)")
                    shutil.rmtree(d)
                    removed.append(d.name)
                    # Also remove chimera (post-2026-04-26: dated files in chimera_legacy/)
                    legacy_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "chimera_legacy"
                    if legacy_dir.exists():
                        sym_l = d.name.lower()
                        for chim in legacy_dir.glob(f"{sym_l}_v50_chimera*.parquet"):
                            chim.unlink()
                            print(f"  [CLEANUP] Removed chimera: {chim.name}")

            if removed:
                print(f"\n  [CLEANUP] Removed {len(removed)} assets: {', '.join(removed)}")
            else:
                print(f"\n  [CLEANUP] Nothing to remove ({len(keep)} assets in keep list)")

    print("\n[DONE] ALL JOBS DONE.")
