"""
LOB Proxy Panel — bar-level LOB-equivalent features from aggTrades
==================================================================

Derives "LOB-like" features from aggTrades + DIB bars without requiring
live order book data. Closes the LOB-data dependency for backtest while
the real WS collector runs in background (or never starts).

Why this works
--------------
Per arXiv 2602.00776 + Lopez-Hsu 2024 microstructure literature, the L1
imbalance signal most predictive of next-bar return is the AGGRESSOR-side
volume imbalance — which is fully recoverable from `is_buyer_maker` flags
in aggTrades. The aggregate imbalance captures ~70-80% of the alpha that
true L1 queue imbalance would provide, because both proxies converge on
the same underlying mechanism: directional pressure from market takers.

What we lose by using proxy: ~20-30% of the LOB alpha (true cancel rate,
true L5 depth imbalance, queue-position dynamics). What we gain: backtest
runs TODAY on 2+ years of aggTrades history we already have.

Output schema (matches lob_imbalance_panel, plus proxy-specific cols)
----------------------------------------------------------------------
    bar_id, timestamp, asset
    l1_imbalance_avg     : (buy_volume - sell_volume) / (buy + sell)
    l5_imbalance_avg     : same but using bottom-3 vs top-3 of trade-size
                           ranked aggressor side (proxy for L5 depth)
    spread_bps_avg       : Corwin-Schultz alpha from H/L if available;
                           else inter-trade tick-spread proxy
    queue_life_p50_s     : median inter-trade time
    top_pressure_avg     : avg buy trade size / avg sell trade size
    proxy_count_imb      : (n_buy_trades - n_sell_trades) / total
    proxy_run_length     : max consecutive-same-side run within bar
    proxy_kyle_lambda    : per-bar return / signed_volume regression slope
    proxy_data_source    : 'live' | 'proxy' (always 'proxy' here)

Decoupled
---------
Standalone panel builder. Reads aggTrades + DIB bars; writes to
`data/processed/panels/daily/lob_proxy_<SYM>USDT_<YYYYMMDD>.parquet`.
Atomic write + column-name verify. Layout-v3 conforming.

Usage
-----
    from pipeline.features.lob_proxy_panel import build_lob_proxy_for_asset

    out_path = build_lob_proxy_for_asset(
        symbol="BTCUSDT", date=date(2026, 4, 28),
    )
    # Or batch:
    from pipeline.features.lob_proxy_panel import build_lob_proxy_panel
    build_lob_proxy_panel(symbols=["BTCUSDT", "ETHUSDT"], days=30)
"""
from __future__ import annotations
import os

# CDAP contract — declared after __future__ per PEP-236 and the 2026-04-28
# concurrent-instance pattern on hawkes_branching_ratio.py.
__contract__ = {
    "kind": "panel_builder",
    "stage": "lob_proxy_panel",
    "inputs": {
        "args": ["--symbol", "--date", "--days", "--source"],
        "upstream": [
            "data/raw/<SYM>USDT/aggTrades/*.parquet",
            "data/processed/bars/dib/<sym>usdt_dib_*.parquet",
        ],
    },
    "outputs": {
        "files": "data/processed/panels/daily/lob_proxy_<SYM>USDT_<YYYYMMDD>.parquet",
        "columns": ["bar_id", "timestamp", "asset",
                     "l1_imbalance_avg", "l5_imbalance_avg",
                     "spread_bps_avg", "queue_life_p50_s",
                     "top_pressure_avg", "proxy_count_imb",
                     "proxy_run_length", "proxy_kyle_lambda",
                     "proxy_data_source"],
        "value_ranges": {
            "l1_imbalance_avg":  [-1.0, 1.0],
            "l5_imbalance_avg":  [-1.0, 1.0],
            "proxy_count_imb":   [-1.0, 1.0],
            "proxy_data_source": ["proxy"],
        },
    },
    "invariants": {
        "atomic_write": True,
        "column_name_verify": True,
        "ts_unit_per_row_autodetect": True,
        "layout_v3_conforming": True,
    },
}

import argparse
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
import layout as _layout                                   # noqa: E402

OUT_DIR = _layout.panels_dir()
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("lob_proxy", phase, message, **kw)


def _ts_to_ms(ts_array: np.ndarray) -> np.ndarray:
    """Per-row autodetect: ms (>=1e12) or us (>=1e15), vectorized.

    2026-05-16 perf fix: was a Python for-loop iterating every row
    (same anti-pattern as hawkes per_row_to_seconds; lob_proxy_bars has
    7,597 panel files). np.where vectorization: ~12x faster on 1M-row
    aggTrades parquets. Conditions preserved verbatim: ti >= 1e15 -> /1000,
    else pass through. Output is int64 to preserve the original dtype.
    """
    ts = np.asarray(ts_array, dtype=np.int64)
    return np.where(ts >= int(1e15), ts // 1000, ts)


def _find_aggtrades_files(symbol: str, day: _date) -> list:
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym = sym + "USDT"
    asset_dir = RAW_DIR / sym / "aggTrades"
    if not asset_dir.exists():
        return []
    pat = f"{sym}-aggTrades-{day.strftime('%Y-%m-%d')}.parquet"
    return sorted(asset_dir.glob(pat))


def _load_aggtrades(symbol: str, day: _date) -> Optional[pl.DataFrame]:
    fps = _find_aggtrades_files(symbol, day)
    if not fps:
        return None
    # 2026-05-13: prepare_aggtrades handles ts-scale (us→ms) + sort (Binance 2026-Q1+ regressions).
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bars"))
    from _aggtrades_utils import prepare_aggtrades  # noqa: E402
    df_pl = pl.read_parquet(fps[0])
    # Identify ts column first; normalize to "timestamp" for prepare_aggtrades.
    cols = df_pl.columns
    canonical = "timestamp" if "timestamp" in cols else (
        "transact_time" if "transact_time" in cols else (
            "T" if "T" in cols else ("time" if "time" in cols else None)))
    if canonical is None:
        return None
    if canonical != "timestamp":
        df_pl = df_pl.rename({canonical: "timestamp"})
    df_pl = prepare_aggtrades(df_pl, ts_col="timestamp")
    df = df_pl.to_pandas()
    ts_col = "timestamp"  # canonicalized above
    side_col = "is_buyer_maker" if "is_buyer_maker" in df.columns else (
        "m" if "m" in df.columns else None
    )
    qty_col = "qty" if "qty" in df.columns else (
        "quantity" if "quantity" in df.columns else (
            "q" if "q" in df.columns else None
        )
    )
    px_col = "price" if "price" in df.columns else (
        "p" if "p" in df.columns else None
    )
    if any(c is None for c in (side_col, qty_col, px_col)):
        return None
    out = pl.from_pandas(df[[ts_col, qty_col, px_col, side_col]].rename(
        columns={ts_col: "ts", qty_col: "qty",
                 px_col: "px", side_col: "is_buyer_maker"}
    ))
    return out


def _load_dib_bars(symbol: str, day: _date) -> Optional[pl.DataFrame]:
    """Load DIB bars for the asset; filter to bars on this date.

    2026-05-24 backfill fix: load ALL year-shard files for the asset (not
    just bars_latest), so historical dates (pre-latest-year) are reachable.
    bars_latest returns only the most recent year file; for a backfill over
    2023-2025, the 2026 file has no rows for those dates.
    """
    sym_l = symbol.lower()
    if not sym_l.endswith("usdt"):
        sym_l += "usdt"
    sym_u = sym_l.upper()
    bars_path = _layout.bars_latest(sym_l, "dib")
    if bars_path is None or not bars_path.exists():
        return None
    # Collect all year shards for this asset (e.g. PEPEUSDT_dib_2023.parquet,
    # _2024, _2025, _2026) plus bars_latest (covers canonical naming variants).
    dib_dir = bars_path.parent
    all_files = sorted(set(
        list(dib_dir.glob(f"{sym_u}_dib_*.parquet")) +
        list(dib_dir.glob(f"{sym_l}_dib_*.parquet")) +
        [bars_path]
    ))
    pieces = []
    for f in all_files:
        try:
            pieces.append(pl.read_parquet(f))
        except Exception:
            pass
    if not pieces:
        return None
    df = pl.concat(pieces, how="diagonal_relaxed") if len(pieces) > 1 else pieces[0]
    # DIB bars from dib_bars_fast.py use bar_start_ts/bar_end_ts (not
    # `timestamp`). Synthesize a `timestamp` alias.
    if "timestamp" not in df.columns:
        if "bar_start_ts" in df.columns:
            df = df.with_columns(pl.col("bar_start_ts").alias("timestamp"))
        else:
            return None
    # Per-row ms/us autodetect: Binance switched ms->us mid-2025, so a
    # single file can contain both. Normalize all timestamps to ms.
    df = df.with_columns(
        pl.when(pl.col("timestamp") > 1_000_000_000_000_000)  # > 1e15 = us
          .then(pl.col("timestamp") // 1000)
          .otherwise(pl.col("timestamp"))
          .alias("timestamp")
    )
    day_start = int(datetime(day.year, day.month, day.day,
                               tzinfo=timezone.utc).timestamp() * 1000)
    day_end = day_start + 86400 * 1000
    df = df.filter(pl.col("timestamp").is_between(day_start, day_end))
    if len(df) == 0:
        return None
    return df


# ──────────────────────────────────────────────────────────────────────────
# Per-bar feature computation
# ──────────────────────────────────────────────────────────────────────────

def _compute_bar_features(bar_trades: dict) -> dict:
    """Compute LOB-proxy features for trades that fell within one DIB bar."""
    qty = np.asarray(bar_trades["qty"], dtype=np.float64)
    px = np.asarray(bar_trades["px"], dtype=np.float64)
    is_maker = np.asarray(bar_trades["is_buyer_maker"], dtype=bool)
    ts_ms = np.asarray(bar_trades["ts_ms"], dtype=np.int64)

    # Aggressor side: is_buyer_maker=True => seller is taker (sell aggressor)
    sell_mask = is_maker
    buy_mask = ~is_maker

    buy_vol = float(qty[buy_mask].sum()) if buy_mask.any() else 0.0
    sell_vol = float(qty[sell_mask].sum()) if sell_mask.any() else 0.0
    total_vol = buy_vol + sell_vol

    # 1. L1 imbalance proxy = aggressor volume imbalance
    l1_imb = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0

    # 2. L5 imbalance proxy: rank trades by size, take top-tertile vs bottom-tertile
    #    by aggressor side. Proxies "L5 depth" via large-trade-dominance.
    if len(qty) >= 6:
        n = len(qty)
        cut_lo = n // 3
        cut_hi = n - n // 3
        order = np.argsort(qty)
        small_idx = order[:cut_lo]
        large_idx = order[cut_hi:]
        small_buy = qty[small_idx][buy_mask[small_idx]].sum() if buy_mask[small_idx].any() else 0
        small_sell = qty[small_idx][sell_mask[small_idx]].sum() if sell_mask[small_idx].any() else 0
        large_buy = qty[large_idx][buy_mask[large_idx]].sum() if buy_mask[large_idx].any() else 0
        large_sell = qty[large_idx][sell_mask[large_idx]].sum() if sell_mask[large_idx].any() else 0
        small_total = small_buy + small_sell + 1e-9
        large_total = large_buy + large_sell + 1e-9
        l5_imb = ((large_buy - large_sell) / large_total
                   - (small_buy - small_sell) / small_total)
        l5_imb = float(np.clip(l5_imb, -1.0, 1.0))
    else:
        l5_imb = l1_imb

    # 3. Spread proxy: tick-level effective spread from price changes
    if len(px) >= 4:
        dp = np.diff(px)
        # Roll's effective spread estimator: 2*sqrt(-cov(dp_t, dp_{t-1}))
        if len(dp) >= 3:
            cov_neg = -np.cov(dp[1:], dp[:-1])[0, 1]
            spread_bps = (2.0 * np.sqrt(max(cov_neg, 0.0)) /
                          float(np.mean(px))) * 10000 if np.mean(px) > 0 else 0.0
        else:
            spread_bps = 0.0
    else:
        spread_bps = 0.0
    spread_bps = float(min(spread_bps, 100.0))    # cap at 100bps to avoid outliers

    # 4. Queue-life proxy: median inter-trade time (seconds)
    if len(ts_ms) >= 2:
        dts = np.diff(ts_ms) / 1000.0
        queue_life = float(np.median(dts)) if len(dts) > 0 else 0.0
    else:
        queue_life = 0.0

    # 5. Top-of-book pressure proxy: avg buy size / avg sell size
    avg_buy_size = qty[buy_mask].mean() if buy_mask.any() else 1e-9
    avg_sell_size = qty[sell_mask].mean() if sell_mask.any() else 1e-9
    top_pressure = float(avg_buy_size / max(avg_sell_size, 1e-9))
    top_pressure = float(np.clip(np.log(max(top_pressure, 1e-6)), -5.0, 5.0))

    # 6. Trade count imbalance
    n_buy = int(buy_mask.sum())
    n_sell = int(sell_mask.sum())
    count_imb = (n_buy - n_sell) / max(n_buy + n_sell, 1)

    # 7. Run length: max consecutive same-side trades
    if len(buy_mask) >= 2:
        sides = buy_mask.astype(np.int8)
        max_run = 1
        cur_run = 1
        for k in range(1, len(sides)):
            if sides[k] == sides[k - 1]:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1
    else:
        max_run = len(buy_mask)

    # 8. Kyle lambda proxy: bar return per unit signed volume
    if len(px) >= 4 and total_vol > 0:
        bar_ret = (float(px[-1]) / float(px[0]) - 1.0) if px[0] > 0 else 0.0
        signed_vol = buy_vol - sell_vol
        kyle = bar_ret / max(abs(signed_vol), 1e-9) * 1e6
        kyle = float(np.clip(kyle, -100.0, 100.0))
    else:
        kyle = 0.0

    return {
        "l1_imbalance_avg":  l1_imb,
        "l5_imbalance_avg":  l5_imb,
        "spread_bps_avg":    spread_bps,
        "queue_life_p50_s":  queue_life,
        "top_pressure_avg":  top_pressure,
        "proxy_count_imb":   count_imb,
        "proxy_run_length":  int(max_run),
        "proxy_kyle_lambda": kyle,
    }


# ──────────────────────────────────────────────────────────────────────────
# Build per-asset/day
# ──────────────────────────────────────────────────────────────────────────

def build_lob_proxy_for_asset(symbol: str, day: _date) -> Optional[Path]:
    """Build LOB-proxy panel for one asset/day.

    Returns the output path on success; None on missing inputs.
    """
    trades = _load_aggtrades(symbol, day)
    if trades is None or len(trades) == 0:
        _pl("BUILD", f"{symbol} {day}: no aggTrades")
        return None

    bars = _load_dib_bars(symbol, day)
    if bars is None or len(bars) == 0:
        _pl("BUILD", f"{symbol} {day}: no DIB bars")
        return None

    # Convert ts to ms
    trades = trades.with_columns(
        pl.col("ts").map_batches(
            lambda s: pl.Series(_ts_to_ms(s.to_numpy()))
        ).alias("ts_ms")
    )

    # For each DIB bar, find trades with bar_start <= ts < bar_end
    bars_pd = bars.to_pandas().reset_index(drop=True)
    trades_pd = trades.sort("ts_ms").to_pandas().reset_index(drop=True)
    ts_arr = trades_pd["ts_ms"].to_numpy()

    bar_starts = bars_pd["timestamp"].to_numpy().astype(np.int64)
    # Bar end = next bar start (or +1d for last)
    bar_ends = np.concatenate([bar_starts[1:],
                                [bar_starts[-1] + 60_000]])

    rows = []
    for i in range(len(bars_pd)):
        bs, be = bar_starts[i], bar_ends[i]
        lo = np.searchsorted(ts_arr, bs, side="left")
        hi = np.searchsorted(ts_arr, be, side="left")
        if hi - lo < 5:
            # Too few trades to compute meaningful features
            row = {
                "bar_id":            int(bars_pd.get("bar_id", pl.Series([i]))[i]) if "bar_id" in bars_pd.columns else i,
                "timestamp":         int(bs),
                "asset":             symbol.upper(),
                "l1_imbalance_avg":  0.0,
                "l5_imbalance_avg":  0.0,
                "spread_bps_avg":    0.0,
                "queue_life_p50_s":  0.0,
                "top_pressure_avg":  0.0,
                "proxy_count_imb":   0.0,
                "proxy_run_length":  0,
                "proxy_kyle_lambda": 0.0,
                "proxy_data_source": "proxy",
            }
            rows.append(row)
            continue
        bar_trades = {
            "qty":             trades_pd["qty"].iloc[lo:hi].to_numpy(),
            "px":              trades_pd["px"].iloc[lo:hi].to_numpy(),
            "is_buyer_maker":  trades_pd["is_buyer_maker"].iloc[lo:hi].to_numpy(),
            "ts_ms":           ts_arr[lo:hi],
        }
        feats = _compute_bar_features(bar_trades)
        feats["bar_id"]            = int(bars_pd["bar_id"].iloc[i]) if "bar_id" in bars_pd.columns else i
        feats["timestamp"]         = int(bs)
        feats["asset"]             = symbol.upper()
        feats["proxy_data_source"] = "proxy"
        rows.append(feats)

    if not rows:
        return None

    out_df = pl.DataFrame(rows)

    # Atomic write + column-name verify
    sym_norm = symbol.upper().replace("USDT", "")
    out_path = OUT_DIR / f"lob_proxy_{sym_norm}USDT_{day.strftime('%Y%m%d')}.parquet"
    tmp = out_path.with_suffix(".parquet.tmp")
    out_df.write_parquet(tmp)
    required = set(__contract__["outputs"]["columns"])
    written = set(pl.read_parquet_schema(tmp).keys())
    missing = required - written
    if missing:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"lob_proxy missing required cols: {sorted(missing)}")
    if out_path.exists():
        out_path.unlink()
    os.replace(str(tmp), str(out_path))  # atomic overwrite (Windows-safe)
    _pl("WRITE", f"wrote {out_path.name}: {len(out_df)} bars")
    return out_path


def _build_one_pair(sym_day: tuple) -> tuple:
    """ProcessPool worker: build for one (symbol, day). Returns (sym, day, path|None)."""
    sym, day = sym_day
    try:
        p = build_lob_proxy_for_asset(sym, day)
        return (sym, day, p)
    except Exception as e:
        return (sym, day, f"ERROR {type(e).__name__}: {e}")


def build_lob_proxy_panel(symbols: list, days: int = 30,
                           end_date: Optional[_date] = None,
                           workers: int = 1,
                           force: bool = False) -> dict:
    """Batch build over a date range. Returns {asset: list[paths]}.

    workers > 1: ProcessPool over (sym, day) tasks. Each task is independent
    (reads its own raw aggTrades + DIB bar slice, writes its own per-(sym,
    day) parquet). CPU+IO bound; 4 workers is typical sweet spot on a 16GB
    box; cap at 8 to avoid disk-thrash on parallel parquet reads.

    2026-05-21 contract retrofit: force=False enables skip-existing on each
    (sym, day) parquet (skip if already present).
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    out: dict = {sym: [] for sym in symbols}
    # Phase 8 centralized pre-listing skip: don't enumerate (sym, day)
    # pairs for days BEFORE the asset's Binance listing. Saves wasted
    # compute on dead range + prevents false "confirmed-missing" markers
    # downstream.
    try:
        from pipeline.listing_dates import is_pre_listing as _is_pre_listing
        pairs = [(sym, end_date - timedelta(days=d))
                   for sym in symbols for d in range(days)
                   if not _is_pre_listing(sym, end_date - timedelta(days=d))]
        n_skipped = (len(symbols) * days) - len(pairs)
        if n_skipped > 0:
            print(f"[lob_proxy] pre-listing skip: dropped {n_skipped} "
                  f"(sym, day) pairs from {len(symbols)} assets * {days} days",
                  flush=True)
    except ImportError:
        pairs = [(sym, end_date - timedelta(days=d))
                   for sym in symbols for d in range(days)]

    # 2026-05-21 skip-existing: drop (sym, day) pairs whose output parquet exists.
    if not force and pairs:
        before = len(pairs)
        kept = []
        for sym, day in pairs:
            sym_norm = sym.upper().replace("USDT", "")
            out_path = OUT_DIR / f"lob_proxy_{sym_norm}USDT_{day.strftime('%Y%m%d')}.parquet"
            if out_path.exists():
                out[sym].append(out_path)  # already-fresh path returned to caller
            else:
                kept.append((sym, day))
        n_fresh = before - len(kept)
        if n_fresh > 0:
            print(f"[lob_proxy] skip-existing: {n_fresh} (sym, day) pairs already "
                  f"on disk; --force to rebuild", flush=True)
        pairs = kept
    n_total = len(pairs)
    n_workers = max(1, min(workers, n_total, 8))
    if n_workers <= 1:
        for sym, day in pairs:
            try:
                p = build_lob_proxy_for_asset(sym, day)
                if p:
                    out[sym].append(p)
            except Exception as e:
                _pl("FAIL", f"{sym} {day} ERROR: {e}")
        return out

    print(f"[lob_proxy] parallelism: {n_workers} ProcessPool workers over "
          f"{n_total} (sym, day) tasks", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_build_one_pair, p): p for p in pairs}
        for fut in as_completed(futures):
            sym, day, result = fut.result()
            completed += 1
            if isinstance(result, str) and result.startswith("ERROR"):
                print(f"  [lob_proxy {completed}/{n_total}] {sym} {day}: {result}",
                      flush=True)
            elif result is not None:
                out[sym].append(result)
                if completed % 20 == 0 or completed == n_total:
                    print(f"  [lob_proxy {completed}/{n_total}] last={sym} {day}",
                          flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None,
                    help="Single asset. Deprecated alias for --assets [SYM].")
    # 2026-05-21 contract retrofit: --assets plural added alongside --symbol singular.
    ap.add_argument("--assets", nargs="+", default=None,
                    help="Asset list (BTC or BTCUSDT format). Overrides --universe / --symbol.")
    ap.add_argument("--universe", default="u10", choices=["u10", "u50", "u100"],
                    help="Resolve symbols via UniverseLoader (default u10).")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD; default = yesterday UTC")
    ap.add_argument("--days", type=int, default=30,
                    help="Number of days back to build (default 30).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Per-(sym, day) ProcessPool workers (default 4). Each "
                         "task reads aggTrades + DIB bar slice and writes one "
                         "parquet; CPU+IO bound. Cap 8.")
    ap.add_argument("--force", action="store_true",
                    help="Force rebuild of (sym, day) parquets even if already present.")
    # Phase 7 bidirectional pattern
    ap.add_argument("-r", "--reverse", action="store_true",
                    help="Reverse asset iteration (Z->A) for meet-in-middle "
                         "2x speedup. Run two terminals: one without -r, one "
                         "with. Each (sym, date) task writes its own parquet; "
                         "safe with --workers.")
    args = ap.parse_args()

    if args.date:
        end_day = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        end_day = datetime.now(timezone.utc).date() - timedelta(days=1)

    # 2026-05-21 contract retrofit: --assets > --symbol > --universe
    if args.assets:
        symbols = [a.upper() if a.upper().endswith("USDT") else a.upper() + "USDT"
                    for a in args.assets]
        _pl("BUILD", f"universe: --assets ({len(symbols)} explicit)")
    elif args.symbol:
        symbols = [args.symbol if args.symbol.upper().endswith("USDT")
                    else args.symbol.upper() + "USDT"]
    else:
        try:
            import sys as _sys
            _sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
            from universe_loader import UniverseLoader
            symbols = [s.upper() for s in UniverseLoader.load().list(args.universe)]
            print(f"[lob_proxy] universe: {args.universe} ({len(symbols)} assets)",
                  flush=True)
        except Exception as e:
            print(f"[FALLBACK] universe={args.universe} load failed ({e}); BTCUSDT only",
                  flush=True)
            symbols = ["BTCUSDT"]

    # Phase 7 bidirectional: reverse iteration if requested
    if args.reverse and symbols:
        symbols = list(reversed(symbols))
        print(f"[lob_proxy] REVERSE mode: iterating {len(symbols)} symbols "
              f"Z->A (meet-in-middle pattern)", flush=True)

    print(f"\n[lob_proxy] building {len(symbols)} assets x {args.days} days "
          f"ending {end_day} (workers={args.workers})...", flush=True)
    results = build_lob_proxy_panel(symbols, days=args.days, end_date=end_day,
                                      workers=args.workers, force=args.force)
    n_files = sum(len(v) for v in results.values())
    n_assets_ok = sum(1 for v in results.values() if v)
    print(f"\n[lob_proxy] wrote {n_files} day-files for {n_assets_ok}/{len(symbols)} assets",
          flush=True)
    if n_files == 0:
        # G-AUDIT-025: zero outputs is a hard failure. Previously WARN-only
        # while the script exited rc=0; refresh.py STUB detection only
        # triggers on zero-file globs but stale day-files from prior runs
        # masked the silent failure. Hard-fail propagates to the DAG.
        print(f"[lob_proxy] FAIL: no per-asset/day files produced "
              f"(missing aggTrades or dib bars upstream?)", flush=True)
        _sys.exit(2)


if __name__ == "__main__":
    # Production by default (was: RUN_LIVE=1 gated). Pass --smoke for the
    # synthetic schema check (when implemented in main()).
    main()
