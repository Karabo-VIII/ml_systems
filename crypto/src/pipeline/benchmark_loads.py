"""Pipeline load-performance benchmark.

Measures cold-load and warm-load times for chimera v51 + cadence views.
Helps strategies pick the right cadence (faster than re-running on dollar bars
when 4h is sufficient) and surfaces parquet-IO regressions early.

Run:
  python src/pipeline/benchmark_loads.py --asset BTCUSDT
  python src/pipeline/benchmark_loads.py --asset BTCUSDT --cols close,norm_return_1
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import polars as pl

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from chimera_loader import ChimeraLoader  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def time_load(loader: ChimeraLoader, sym: str, cadence: str,
              cols: list[str] | None, n_runs: int = 3) -> dict:
    times = []
    rows = 0
    n_cols = 0
    bytes_read = 0
    for i in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        df = loader.load(sym, cadence=cadence, features=cols)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        rows = len(df)
        n_cols = len(df.columns)
        bytes_read = df.estimated_size("mb")
    return {
        "cadence": cadence,
        "rows": rows,
        "cols": n_cols,
        "first_load_ms": round(times[0], 1),
        "warm_load_ms_avg": round(sum(times[1:]) / max(len(times) - 1, 1), 1) if n_runs > 1 else None,
        "warm_load_ms_min": round(min(times[1:]), 1) if n_runs > 1 else None,
        "size_mb": round(bytes_read, 1),
        "throughput_mb_per_s": round(bytes_read / max(times[-1] / 1000, 0.001), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTCUSDT")
    ap.add_argument("--cols", default=None,
                    help="Comma-sep column subset to load (faster). Default: all.")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    sym = args.asset.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"

    cols = None
    if args.cols:
        cols = [c.strip() for c in args.cols.split(",")]

    loader = ChimeraLoader()

    print(f"[bench] {sym}, n_runs={args.runs}, cols={'all' if cols is None else len(cols)}")
    print()

    results = []
    for cadence in ("dollar", "1d", "4h", "1h", "15m"):
        try:
            r = time_load(loader, sym, cadence, cols, n_runs=args.runs)
            results.append(r)
            warm = f"{r['warm_load_ms_min']:>5}ms" if r['warm_load_ms_min'] is not None else " --  "
            print(f"  {cadence:>6}  {r['rows']:>10,}r  {r['cols']:>4}c  "
                  f"first={r['first_load_ms']:>6}ms  warm_min={warm}  "
                  f"size={r['size_mb']:>6}MB  thrpt={r['throughput_mb_per_s']:>6}MB/s")
        except FileNotFoundError as e:
            print(f"  {cadence:>6}  SKIP (no file)")

    print()
    print("[bench] tip: cadence views give 100x speedup over dollar for daily strategies.")


if __name__ == "__main__":
    main()
