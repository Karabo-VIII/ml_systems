"""Range Bars — fixed-range price-move bars (Lopez de Prado).

Bar closes when price moves >= range_threshold from bar open.
Pure price-driven; ignores time, volume, ticks.

Natural fit for trend-follow strategies because:
- Every bar has identical range by construction
- Drawdowns bounded by range
- MA crosses / breakout signals are cleaner

Output:
    data/processed/bars/range/<sym>_range_2025.parquet
        columns: bar_start_ts, bar_end_ts, open, high, low, close,
                 volume, signed_usd, buy_usd, sell_usd, tick_count, direction

Threshold policy (2026-04-29 fix after SUI 58M-bar memory hang):
    Per-asset thresholds resolved in this order:
        1. Explicit RANGE_THRESHOLDS table (BTC/ETH/SOL hand-tuned).
        2. Liquidity-tier default ladder via UniverseLoader.liquidity_tier():
             TIER_B     -> 0.5%   (deep majors)
             TIER_C     -> 0.8%
             EVENT_ONLY -> 1.5%   (high-vol low-price; PEPE/SUI-class)
             else / unset -> 1.5% (conservative default)
        3. Hard cap: if estimated bars/day > MAX_BARS_PER_DAY (10K), the
           threshold is auto-widened (2x then 4x base) until it fits. If
           still over at 4x, the asset is SKIPPED.

    Rationale: a uniform 0.5% default produced 58M bars for SUI on 350
    days; the pandas-to-parquet write hit memory pressure across 4
    concurrent bartypes and hung at 23 GB RSS.
"""
from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from numba import njit

# Framework primitives.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parquet_io import atomic_write_parquet, delta_state, append_parquet
from dispatch import run_per_task
from cli import add_standard_args, resolve_assets


# Hot inner loop: stateful streaming range-bar builder.
# 2026-05-16 perf fix: was a Python for-loop iterating over every trade
# (100K-500K per asset-day x 87 assets x 2327 days). The streaming algorithm
# is intrinsically sequential (each bar boundary depends on running min/max
# since last bar open) so it cannot be naively vectorized with numpy. numba
# JIT compiles the Python source to native code, ~50-100x faster than the
# CPython interpreter loop, while preserving exact algorithm semantics.
#
# Output as parallel numpy arrays (numba cannot return list-of-dict).
# The wrapper converts back to dict-per-bar for the original API.
#
# 2026-05-17 cache=False: cache=True caused a "No module named '<dynamic>'"
# error on first run after code changes -- numba's pickle of the cached AOT
# binary references a module name that's no longer resolvable when the
# source's signature/structure changes. The JIT cost is one ~1-2s warmup at
# script startup; trivial vs the 184x runtime speedup. Auto-invalidation
# beats manual cache-clearing (which was today's recovery action).
# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
# NOTE (2026-05-29 fix): this helper is PURE PYTHON and must NOT be @njit -- it
# uses try/except + dynamic imports that numba cannot compile. The @njit was
# previously mis-attached here (so it crashed on first call AND the real hot
# loop ran unaccelerated). The decorator now correctly sits on
# _range_bars_inner below.
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("range", phase, message, **kw)


@njit(cache=False)
def _range_bars_inner(ts: np.ndarray, price: np.ndarray, qty: np.ndarray,
                       is_buyer_maker: np.ndarray, range_pct: float):
    """Numba-jitted hot loop. Same semantics as the prior Python loop."""
    n = len(ts)
    # Pre-allocate worst case (one bar per row). Trim before return.
    bar_start_ts = np.empty(n, dtype=np.int64)
    bar_end_ts = np.empty(n, dtype=np.int64)
    bar_open_arr = np.empty(n, dtype=np.float64)
    bar_hi_arr = np.empty(n, dtype=np.float64)
    bar_lo_arr = np.empty(n, dtype=np.float64)
    bar_close_arr = np.empty(n, dtype=np.float64)
    bar_vol_arr = np.empty(n, dtype=np.float64)
    bar_signed_arr = np.empty(n, dtype=np.float64)
    bar_buy_arr = np.empty(n, dtype=np.float64)
    bar_sell_arr = np.empty(n, dtype=np.float64)
    bar_tick_arr = np.empty(n, dtype=np.int64)
    bar_dir_arr = np.empty(n, dtype=np.int8)

    n_bars = 0
    has_open = False
    cur_open = 0.0
    cur_open_ts = np.int64(0)
    cur_hi = -np.inf
    cur_lo = np.inf
    cur_vol = 0.0
    cur_buy = 0.0
    cur_sell = 0.0
    cur_signed = 0.0
    cur_tick = 0

    for i in range(n):
        p = price[i]
        v = p * qty[i]
        sign = -1.0 if is_buyer_maker[i] else 1.0
        if not has_open:
            cur_open = p
            cur_open_ts = ts[i]
            cur_hi = p
            cur_lo = p
            cur_vol = 0.0
            cur_buy = 0.0
            cur_sell = 0.0
            cur_signed = 0.0
            cur_tick = 0
            has_open = True
        if p > cur_hi:
            cur_hi = p
        if p < cur_lo:
            cur_lo = p
        cur_vol += qty[i]
        cur_signed += sign * v
        if is_buyer_maker[i]:
            cur_sell += v
        else:
            cur_buy += v
        cur_tick += 1

        move = (p - cur_open) / cur_open
        if abs(move) >= range_pct:
            bar_start_ts[n_bars] = cur_open_ts
            bar_end_ts[n_bars] = ts[i]
            bar_open_arr[n_bars] = cur_open
            bar_hi_arr[n_bars] = cur_hi
            bar_lo_arr[n_bars] = cur_lo
            bar_close_arr[n_bars] = p
            bar_vol_arr[n_bars] = cur_vol
            bar_signed_arr[n_bars] = cur_signed
            bar_buy_arr[n_bars] = cur_buy
            bar_sell_arr[n_bars] = cur_sell
            bar_tick_arr[n_bars] = cur_tick
            bar_dir_arr[n_bars] = 1 if move > 0 else -1
            n_bars += 1
            has_open = False

    return (bar_start_ts[:n_bars], bar_end_ts[:n_bars],
            bar_open_arr[:n_bars], bar_hi_arr[:n_bars], bar_lo_arr[:n_bars],
            bar_close_arr[:n_bars], bar_vol_arr[:n_bars],
            bar_signed_arr[:n_bars], bar_buy_arr[:n_bars],
            bar_sell_arr[:n_bars], bar_tick_arr[:n_bars],
            bar_dir_arr[:n_bars])

# Shared aggTrades utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _aggtrades_utils import prepare_aggtrades

# Explicit src/ on sys.path so `from pipeline.bidirectional import ...`
# resolves regardless of cwd. Sibling bars (dib_bars_fast/runs_bars/
# adaptive_vol_bars) get this transitively via importing _thresholds; we
# don't import _thresholds here so we set it explicitly. Tested 2026-05-16
# wave-2 rebuild — without this the script crashes at the `from
# pipeline.bidirectional import iter_assets` line in main().
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _date_from_aggtrades_path(fp):
    """BTCUSDT-aggTrades-2024-01-01.parquet -> date(2024, 1, 1)."""
    parts = fp.stem.split("-")
    if len(parts) < 5:
        return None
    try:
        return pd.Timestamp("-".join(parts[-3:])).date()
    except Exception:
        return None

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed" / "bars" / "range"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hand-tuned thresholds for the 3 calibrated majors.
RANGE_THRESHOLDS = {
    "BTCUSDT": 0.005,
    "ETHUSDT": 0.008,
    "SOLUSDT": 0.012,
}

# Tier-based default ladder (used when not in RANGE_THRESHOLDS).
TIER_DEFAULT_RANGE_PCT = {
    "TIER_B":     0.005,    # 0.5%
    "TIER_C":     0.008,    # 0.8%
    "EVENT_ONLY": 0.015,    # 1.5%
    "":           0.015,    # un-tiered
}

# Hard caps to prevent runaway memory.
MAX_BARS_PER_DAY = 10_000
MAX_THRESHOLD_MULTIPLIER = 4
SAMPLE_DAYS_FOR_PROJECTION = 5


def _resolve_threshold(symbol: str) -> float:
    """Pick range_pct via explicit table > tier ladder > default."""
    if symbol in RANGE_THRESHOLDS:
        return RANGE_THRESHOLDS[symbol]
    try:
        import sys
        sys.path.insert(0, str(ROOT / "src" / "pipeline"))
        from universe_loader import UniverseLoader
        loader = UniverseLoader.load()
        tier = (loader.liquidity_tier(symbol, "u100")
                or loader.liquidity_tier(symbol, "u50"))
        return TIER_DEFAULT_RANGE_PCT.get(tier or "", TIER_DEFAULT_RANGE_PCT[""])
    except Exception:
        return TIER_DEFAULT_RANGE_PCT[""]


def build_range_bars_day(fp: Path, range_pct: float) -> list[dict]:
    """Build range bars from one aggTrades file."""
    try:
        # Normalize ts scale + sort (Binance us-scale + unsort issues, 2025+)
        df_pl = pl.read_parquet(fp)
        df_pl = prepare_aggtrades(df_pl, ts_col="timestamp")
        df = df_pl.to_pandas()
    except Exception:
        return []
    if len(df) == 0:
        return []

    # Coerce dtypes for numba (njit signatures are strict).
    ts = np.asarray(df["timestamp"].values, dtype=np.int64)
    price = np.asarray(df["price"].values, dtype=np.float64)
    qty = np.asarray(df["qty"].values, dtype=np.float64)
    is_buyer_maker = np.asarray(df["is_buyer_maker"].values, dtype=np.bool_)

    # numba-jitted streaming bar builder; ~50-100x faster than the prior
    # Python loop on 100K-500K-trade days.
    (bs_ts, be_ts, b_open, b_hi, b_lo, b_close, b_vol, b_signed,
     b_buy, b_sell, b_tick, b_dir) = _range_bars_inner(
        ts, price, qty, is_buyer_maker, float(range_pct))

    # Materialize as list-of-dict to preserve the original return contract.
    bars = [
        {
            "bar_start_ts": int(bs_ts[i]),
            "bar_end_ts": int(be_ts[i]),
            "open": float(b_open[i]),
            "high": float(b_hi[i]),
            "low": float(b_lo[i]),
            "close": float(b_close[i]),
            "volume": float(b_vol[i]),
            "signed_usd": float(b_signed[i]),
            "buy_usd": float(b_buy[i]),
            "sell_usd": float(b_sell[i]),
            "tick_count": int(b_tick[i]),
            "direction": int(b_dir[i]),
        }
        for i in range(len(bs_ts))
    ]
    return bars


def project_bars_per_day(fps: list, range_pct: float, sample_days: int) -> float:
    """Estimate bars/day by running build_range_bars_day on first N files."""
    sample = fps[:sample_days]
    if not sample:
        return 0.0
    total = 0
    for fp in sample:
        total += len(build_range_bars_day(fp, range_pct))
    return total / max(len(sample), 1)


def adaptive_threshold(symbol: str, fps: list) -> tuple:
    """Resolve threshold + auto-widen if projection > MAX_BARS_PER_DAY.

    Returns (range_pct, reason). range_pct < 0 means SKIP this asset.
    """
    base = _resolve_threshold(symbol)
    proj = project_bars_per_day(fps, base, SAMPLE_DAYS_FOR_PROJECTION)
    if proj <= MAX_BARS_PER_DAY:
        return base, ""
    # Auto-widen: try 2x, 4x — bail if still over.
    last_proj = proj
    for mult in (2, MAX_THRESHOLD_MULTIPLIER):
        widened = base * mult
        proj_w = project_bars_per_day(fps, widened, SAMPLE_DAYS_FOR_PROJECTION)
        if proj_w <= MAX_BARS_PER_DAY:
            return widened, (f"auto-widened {base*100:.2f}% -> {widened*100:.2f}% "
                             f"(projected {proj:.0f} -> {proj_w:.0f} bars/day; "
                             f"cap {MAX_BARS_PER_DAY})")
        last_proj = proj_w
    return -1.0, (f"SKIPPED: {proj:.0f} bars/day @ {base*100:.2f}% "
                  f"and {last_proj:.0f} bars/day @ {base*MAX_THRESHOLD_MULTIPLIER*100:.2f}%; "
                  f"both exceed cap {MAX_BARS_PER_DAY}")


def _build_one_asset(symbol: str, fps_to_process: list, range_pct: float,
                      out_path: str, mode: str) -> dict:
    """ProcessPool worker: build range bars for one asset.
    mode='rebuild' -> full overwrite; mode='append' -> append-by-date semantics.
    """
    import time
    t0 = time.time()
    all_bars: list[dict] = []
    for fp in fps_to_process:
        all_bars.extend(build_range_bars_day(Path(fp), range_pct))
    if not all_bars:
        # Legitimate empty result: in append mode, the input dates simply
        # produced no bar closes (low-vol day where price never moved
        # range_pct from the bar open). Existing rows are unchanged; this
        # is success, not failure. In rebuild mode an entirely empty asset
        # IS a real fail (no data at all).
        is_real_failure = (mode == "rebuild")
        return {"status": "empty" if is_real_failure else "ok",
                "symbol": symbol, "n_bars": 0,
                "mode": mode,
                "elapsed_s": round(time.time() - t0, 1),
                "note": "no bars produced (low-vol days)"}
    df_pl = pl.DataFrame(all_bars).with_columns(
        pl.when(pl.col("bar_end_ts") > 1e15)
          .then(pl.col("bar_end_ts") // 1000)
          .otherwise(pl.col("bar_end_ts")).alias("ts_ms")
    ).with_columns(
        pl.from_epoch(pl.col("ts_ms"), time_unit="ms").alias("ts_datetime")
    ).drop("ts_ms").with_columns(
        # Explicit date column for delta-rebuild semantics.
        pl.col("ts_datetime").dt.date().alias("date")
    )

    if mode == "append" and Path(out_path).exists():
        append_parquet(out_path, df_pl, date_col="date", sort_col="ts_datetime")
        n_bars = len(pl.read_parquet(out_path, columns=["date"]))
    else:
        atomic_write_parquet(df_pl, out_path)
        n_bars = len(df_pl)

    return {"status": "ok", "symbol": symbol, "n_bars": n_bars,
            "n_new_days": len(fps_to_process), "mode": mode,
            "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    add_standard_args(ap, default_workers=1)
    global MAX_BARS_PER_DAY
    ap.add_argument("--max-bars-per-day", type=int, default=MAX_BARS_PER_DAY,
                    help=f"Hard cap on projected bars/day (default {MAX_BARS_PER_DAY}). "
                         f"If exceeded at 4x base threshold, asset is SKIPPED.")
    ap.add_argument("--burn-from-first-gap", action="store_true",
                    help="On mid-stream gap, rebuild every input from gap forward.")
    # Phase 7 bidirectional pattern
    ap.add_argument("-r", "--reverse", action="store_true",
                    help="Reverse asset iteration (Z->A) for meet-in-middle "
                         "2x speedup. Two terminals: one without -r, one with.")
    args = ap.parse_args()
    MAX_BARS_PER_DAY = args.max_bars_per_day

    symbols = resolve_assets(args, default=["BTCUSDT", "ETHUSDT"], stage_name="range")
    from pipeline.bidirectional import iter_assets
    symbols = list(iter_assets(symbols, reverse=args.reverse))
    if args.reverse:
        print(f"[range] REVERSE mode: iterating {len(symbols)} assets Z->A",
              flush=True)

    tasks: list[tuple] = []
    n_skipped = n_appends = n_rebuilds = 0
    for symbol in symbols:
        fps = sorted(glob.glob(str(RAW / symbol / "aggTrades" / f"{symbol}-aggTrades-*.parquet")))
        fps_filt = [Path(fp) for fp in fps
                    if args.start <= "-".join(Path(fp).stem.split("-")[-3:]) < args.end]
        if not fps_filt:
            print(f"[range] {symbol} no aggTrades in [{args.start}, {args.end}); skip",
                  flush=True)
            continue
        year_tag = args.start[:4]
        out = OUT_DIR / f"{symbol}_range_{year_tag}.parquet"
        delta = delta_state(out, fps_filt, force=args.force,
                             date_from_filename=_date_from_aggtrades_path,
                             burn_from_first_gap=args.burn_from_first_gap,
                             required_cols={"date", "open", "close", "signed_usd"},
                             max_null_rate={"close": 0.01, "signed_usd": 0.01})
        if delta["mode"] == "fresh":
            _pl("SKIP", f"{symbol} SKIP (fresh: {out.name})")
            n_skipped += 1
            continue
        # Threshold resolution can fail (data too volatile); skip if so.
        range_pct, note = adaptive_threshold(symbol, fps_filt)
        if range_pct < 0:
            _pl("BUILD", f"{symbol} {note}")
            n_skipped += 1
            continue
        if note:
            _pl("BUILD", f"{symbol} {note}")
        if delta["mode"] == "append":
            n_appends += 1
        else:
            n_rebuilds += 1
        print(f"[range] {symbol} {delta['mode']}: range={range_pct*100:.2f}%, "
              f"{len(delta['new_inputs'])} files | {delta['reason'][:80]}",
              flush=True)
        tasks.append((symbol, [str(p) for p in delta["new_inputs"]],
                       range_pct, str(out), delta["mode"]))

    if args.dry_run:
        print(f"[range] dry-run: {n_appends} appends + {n_rebuilds} rebuilds + "
              f"{n_skipped} fresh-skips")
        return

    if not tasks:
        _pl("SKIP", f"nothing to build ({n_skipped} skipped)")
        return

    run_per_task(tasks, _build_one_asset,
                  workers=args.workers, mode="process",
                  stage_name="range",
                  progress_summary_keys=["mode", "n_new_days", "n_bars", "elapsed_s"])


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
