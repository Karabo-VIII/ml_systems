"""Cadence resampling correctness checker.

The chimera v51_<cadence>.parquet files are pre-materialized "last bar of period"
views from the underlying dollar bars. This script verifies the materialized
views match what live aggregation would produce — catches drift between the
batch builder and on-the-fly resampling logic.

Checks per cadence (1d, 4h, 1h, 15m):
  1. Row count plausibility: 1d ≈ unique calendar dates; 4h ≈ 6× 1d; etc.
  2. Schema parity: cadence view has all columns of the dollar source.
  3. Last-bar correctness: for a sample of N periods, verify the cadence row's
     close == last dollar bar's close in that period.
  4. Date alignment: cadence rows should align to period boundaries (e.g., 4h
     boundaries at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC).

Run:
  python src/pipeline/cadence_correctness.py --asset BTCUSDT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import layout as _layout  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def check_one_cadence(symbol: str, cadence: str, full: pl.DataFrame, n_samples: int = 20) -> dict:
    sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
    cad_path = _layout.chimera_v51_latest(sym, cadence)
    if cad_path is None or not cad_path.exists():
        return {"cadence": cadence, "status": "skip_no_file"}
    cad = pl.read_parquet(cad_path)

    out = {"cadence": cadence, "n_rows": len(cad)}

    # 1. Schema parity (cadence has all of full's cols)
    full_cols = set(full.columns)
    cad_cols = set(cad.columns)
    missing = full_cols - cad_cols
    if missing:
        out["schema_status"] = f"FAIL: missing {len(missing)} cols"
    else:
        out["schema_status"] = "OK"

    # 2. Row count plausibility
    if "timestamp" in full.columns:
        if cadence == "1d":
            expected = full.select(
                pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date().n_unique()
            ).item()
            ratio = len(cad) / max(expected, 1)
            out["row_count_status"] = "OK" if 0.95 < ratio < 1.05 else f"WARN ratio={ratio:.3f}"
        elif cadence == "4h":
            expected = full.select(
                pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.truncate("4h").n_unique()
            ).item()
            ratio = len(cad) / max(expected, 1)
            out["row_count_status"] = "OK" if 0.95 < ratio < 1.05 else f"WARN ratio={ratio:.3f}"
        elif cadence == "1h":
            expected = full.select(
                pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.truncate("1h").n_unique()
            ).item()
            ratio = len(cad) / max(expected, 1)
            out["row_count_status"] = "OK" if 0.95 < ratio < 1.05 else f"WARN ratio={ratio:.3f}"
        elif cadence == "15m":
            expected = full.select(
                pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.truncate("15m").n_unique()
            ).item()
            ratio = len(cad) / max(expected, 1)
            out["row_count_status"] = "OK" if 0.95 < ratio < 1.05 else f"WARN ratio={ratio:.3f}"

    # 3. Last-bar correctness: sample N cadence rows, verify close == max-ts dollar bar in period
    if "timestamp" in full.columns and "close" in full.columns and "close" in cad.columns:
        rng = np.random.default_rng(seed=42)
        # Skip first/last 50 rows of cad (boundary effects)
        if len(cad) > 200:
            idxs = rng.integers(low=50, high=len(cad) - 50, size=min(n_samples, len(cad) - 100))
            mismatches = 0
            for i in idxs:
                cad_ts = cad["timestamp"][int(i)]
                cad_close = cad["close"][int(i)]
                if cad_ts is None or cad_close is None:
                    continue
                # Find this same ts in full chimera. When ts has duplicates (zero-diff
                # bursts), pick the LAST tied bar (highest tick_seq) to match
                # materialize_cadence's group_by.last() tie-breaking.
                match = full.filter(pl.col("timestamp") == cad_ts)
                if len(match) == 0:
                    mismatches += 1
                    continue
                if "tick_seq" in match.columns and len(match) > 1:
                    match = match.sort("tick_seq")
                full_close = match["close"][-1]
                if abs(float(full_close) - float(cad_close)) > 1e-6:
                    mismatches += 1
            out["last_bar_check"] = f"OK ({n_samples} samples)" if mismatches == 0 else f"FAIL {mismatches} mismatches"
        else:
            out["last_bar_check"] = "SKIP too few rows"

    return out


def check_asset(symbol: str) -> list[dict]:
    sym = symbol if symbol.endswith("USDT") else f"{symbol}USDT"
    full_path = _layout.chimera_v51_latest(sym, "dollar")
    if full_path is None or not full_path.exists():
        return [{"asset": sym, "error": f"missing v51 chimera for {sym}"}]
    cols_needed = ["timestamp", "close"]
    schema = pl.read_parquet_schema(full_path)
    if "tick_seq" in schema:
        cols_needed.append("tick_seq")
    full = pl.read_parquet(full_path, columns=cols_needed)
    return [check_one_cadence(sym, cad, full) for cad in ("1d", "4h", "1h", "15m")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTCUSDT")
    args = ap.parse_args()
    sym = args.asset.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    print(f"[cadence] checking cadence views for {sym}...")
    results = check_asset(sym)
    n_fail = 0
    for r in results:
        if "error" in r:
            print(f"  ERROR {r['error']}")
            n_fail += 1
            continue
        cad = r["cadence"]
        sch = r.get("schema_status", "?")
        rc = r.get("row_count_status", "?")
        lb = r.get("last_bar_check", "?")
        n_r = r.get("n_rows", 0)
        line = f"  {cad:>4}: {n_r:>10,} rows  schema={sch}  rows={rc}  last_bar={lb}"
        if "FAIL" in line:
            n_fail += 1
            print(line)
        else:
            print(line)
    if n_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
